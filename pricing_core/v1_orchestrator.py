from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data_adapter import build_daily_panel_from_transactions
from recommendation import build_business_recommendation

from pricing_core.v1_features import build_v1_one_step_features, build_v1_panel_feature_matrix, derive_v1_feature_spec
from pricing_core.v1_forecast import predict_v1_demand, recursive_v1_demand_forecast, train_v1_demand_model
from pricing_core.v1_optimizer import recommend_v1_price_horizon, simulate_v1_horizon_profit

V1_ANALYSIS_ENGINE = "v1_universal"
V1_ANALYSIS_ENGINE_VERSION = "v1_universal_panel_xgboost_poisson"


def _eval_wape(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.abs(actual).sum())
    return 100.0 if denom <= 1e-9 else float(np.abs(actual - pred).sum() / denom * 100.0)


def _metrics(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(actual, errors="coerce").fillna(0.0)
    p = pd.to_numeric(pred, errors="coerce").fillna(0.0)
    actual_sum = float(a.sum())
    pred_sum = float(p.sum())
    mae = float(np.abs(a - p).mean()) if len(a) else 0.0
    rmse = float(np.sqrt(np.mean((a - p) ** 2))) if len(a) else 0.0
    sum_ratio = float(pred_sum / actual_sum) if abs(actual_sum) > 1e-9 else 0.0
    bias_pct = float((pred_sum - actual_sum) / actual_sum) if abs(actual_sum) > 1e-9 else 0.0
    return {"forecast_wape": _eval_wape(a, p), "mae": mae, "rmse": rmse, "sum_ratio": sum_ratio, "bias_pct": bias_pct}


def _summarize_diag(diag: pd.DataFrame, key: str) -> pd.DataFrame:
    if len(diag) == 0 or key not in diag.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for val, g in diag.groupby(key, dropna=False):
        m = _metrics(g["actual_sales"], g["pred_sales_main"])
        m[key] = val
        rows.append(m)
    return pd.DataFrame(rows)


def _build_base_ctx(target_history: pd.DataFrame, feature_spec: Dict[str, Any], target_category: str, target_sku: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {"product_id": target_sku, "category": target_category}
    num_fields = ["price", "cost", "discount", "promotion", "stock", "freight_value", "review_score", "reviews_count"] + feature_spec.get("user_numeric_features", [])
    cat_fields = ["region", "channel", "segment"] + feature_spec.get("user_categorical_features", [])
    for c in num_fields:
        s = pd.to_numeric(target_history.get(c, np.nan), errors="coerce")
        base[c] = float(s.dropna().iloc[-1]) if s.notna().any() else 0.0
    for c in cat_fields:
        s = target_history.get(c, pd.Series(dtype=str)).dropna().astype(str)
        base[c] = str(s.iloc[-1]) if not s.empty else "unknown"
    return base


def run_v1_recursive_holdout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trained_models: Dict[str, Any],
    feature_spec: dict[str, Any],
    calibration_factor: float = 1.0,
    forecast_mode: str = "strong_signal",
) -> pd.DataFrame:
    history = train_df.copy().sort_values("date")
    rows: List[Dict[str, Any]] = []
    hdays = int((history["date"].max() - history["date"].min()).days + 1) if len(history) else 1
    for _, tr in test_df.sort_values("date").iterrows():
        step_ctx: Dict[str, Any] = {}
        for c in feature_spec.get("scenario_features", []):
            step_ctx[c] = float(pd.to_numeric(pd.Series([tr.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        for c in feature_spec.get("categorical_demand_features", []):
            step_ctx[c] = str(tr.get(c, "unknown"))
        step = build_v1_one_step_features(history, pd.Timestamp(tr["date"]), step_ctx, hdays, feature_spec)
        pred_row = predict_v1_demand(step, trained_models, feature_spec, forecast_mode=forecast_mode).iloc[0].to_dict()
        pred_row = {k: max(0.0, float(v) * float(calibration_factor)) for k, v in pred_row.items()}
        rows.append(
            {
                "date": pd.Timestamp(tr["date"]),
                "actual_sales": float(tr["sales"]),
                "pred_sales": pred_row["pred_sales"],
                "pred_sales_main": pred_row["pred_sales_main"],
                "pred_sales_fallback": pred_row["pred_sales_fallback"],
                "month": str(pd.Timestamp(tr["date"]).to_period("M")),
                "dow_name": pd.Timestamp(tr["date"]).day_name(),
            }
        )
        history = pd.concat([history, pd.DataFrame([{"date": pd.Timestamp(tr["date"]), "sales": pred_row["pred_sales"], **step_ctx}])], ignore_index=True)

    out = pd.DataFrame(rows)
    if len(out):
        out["residual"] = out["pred_sales"] - out["actual_sales"]
        out["abs_error"] = out["residual"].abs()
    return out


def assess_factor_identifiability(target_history: pd.DataFrame, feature_spec: dict) -> dict:
    hist = target_history.copy().sort_values("date")
    price = pd.to_numeric(hist.get("price", np.nan), errors="coerce").dropna()
    cur_price = float(price.iloc[-1]) if len(price) else 1.0
    history_ok = len(hist) >= 84
    price_variation_ok = bool(price.nunique() >= 5 and cur_price > 0 and ((float(price.max()) - float(price.min())) / cur_price >= 0.08))
    # initial direction proxy, later refined by model perturbation test.
    corr = np.corrcoef(price.values, pd.to_numeric(hist.loc[price.index, "sales"], errors="coerce").fillna(0.0).values)[0, 1] if len(price) > 5 else 0.0
    price_direction_ok = bool(np.isfinite(corr) and corr <= 0.10)
    price_signal_ok = bool(history_ok and price_variation_ok and price_direction_ok)

    weak, strong, strength = [], [], {}
    for f in feature_spec.get("user_numeric_features", []):
        s = pd.to_numeric(hist.get(f, np.nan), errors="coerce")
        ok = bool(s.notna().mean() >= 0.6 and s.nunique(dropna=True) > 1 and (s.max() - s.min()) != 0)
        strength[f] = "strong" if ok else "weak"
        (strong if ok else weak).append(f)

    return {
        "price_signal_ok": price_signal_ok,
        "price_variation_ok": price_variation_ok,
        "price_direction_ok": price_direction_ok,
        "weak_factors": weak,
        "strong_factors": strong,
        "user_factor_strength": strength,
    }


def _run_price_perturbation_test(
    trained_models: Dict[str, Any],
    seed_history: pd.DataFrame,
    calibration_slice: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> Dict[str, Any]:
    if len(seed_history) < 28 or len(calibration_slice) == 0:
        return {"price_direction_ok": False, "avg_reaction": 0.0}
    hdays = int((seed_history["date"].max() - seed_history["date"].min()).days + 1)
    checks = []
    tail = calibration_slice.sort_values("date").tail(min(12, len(calibration_slice)))
    history = seed_history.copy().sort_values("date")
    for _, row in tail.iterrows():
        step_ctx = dict(base_ctx)
        for c in feature_spec.get("scenario_features", []):
            step_ctx[c] = float(pd.to_numeric(pd.Series([row.get(c, step_ctx.get(c, 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        for c in feature_spec.get("categorical_demand_features", []):
            step_ctx[c] = str(row.get(c, step_ctx.get(c, "unknown")))
        step = build_v1_one_step_features(history, pd.Timestamp(row["date"]), step_ctx, hdays, feature_spec)
        base_pred = float(predict_v1_demand(step, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales_main"].iloc[0])
        step_plus = step.copy()
        step_minus = step.copy()
        if "price" in step_plus.columns:
            step_plus["price"] = pd.to_numeric(step_plus["price"], errors="coerce").fillna(0.0) * 1.05
            step_minus["price"] = pd.to_numeric(step_minus["price"], errors="coerce").fillna(0.0) * 0.95
        plus_pred = float(predict_v1_demand(step_plus, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales_main"].iloc[0])
        minus_pred = float(predict_v1_demand(step_minus, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales_main"].iloc[0])
        checks.append((plus_pred <= base_pred, minus_pred >= base_pred, base_pred, plus_pred, minus_pred))
        history = pd.concat([history, pd.DataFrame([{"date": pd.Timestamp(row["date"]), "sales": base_pred, **step_ctx}])], ignore_index=True)
    if not checks:
        return {"price_direction_ok": False, "avg_reaction": 0.0}
    dir_ok_share = float(np.mean([1.0 if (a and b) else 0.0 for a, b, *_ in checks]))
    reactions = [abs((mn - pl) / max(bs, 1e-9)) for _, _, bs, pl, mn in checks]
    avg_reaction = float(np.mean(reactions)) if reactions else 0.0
    return {"price_direction_ok": bool(dir_ok_share >= 0.6 and avg_reaction >= 0.03), "avg_reaction": avg_reaction}


def _run_model_price_effect_sign_test(
    trained_models: Dict[str, Any],
    seed_history: pd.DataFrame,
    calibration_slice: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> Dict[str, Any]:
    if len(seed_history) < 28 or len(calibration_slice) == 0:
        return {"main_sign": 0, "fallback_sign": 0, "agree": False}
    hdays = int((seed_history["date"].max() - seed_history["date"].min()).days + 1)
    tail = calibration_slice.sort_values("date").tail(min(12, len(calibration_slice)))
    history = seed_history.copy().sort_values("date")
    main_deltas: List[float] = []
    fallback_deltas: List[float] = []
    for _, row in tail.iterrows():
        step_ctx = dict(base_ctx)
        for c in feature_spec.get("scenario_features", []):
            step_ctx[c] = float(pd.to_numeric(pd.Series([row.get(c, step_ctx.get(c, 0.0))]), errors="coerce").fillna(0.0).iloc[0])
        for c in feature_spec.get("categorical_demand_features", []):
            step_ctx[c] = str(row.get(c, step_ctx.get(c, "unknown")))
        step = build_v1_one_step_features(history, pd.Timestamp(row["date"]), step_ctx, hdays, feature_spec)
        step_plus = step.copy()
        step_minus = step.copy()
        if "price" in step_plus.columns:
            step_plus["price"] = pd.to_numeric(step_plus["price"], errors="coerce").fillna(0.0) * 1.05
            step_minus["price"] = pd.to_numeric(step_minus["price"], errors="coerce").fillna(0.0) * 0.95
        plus_pred = predict_v1_demand(step_plus, trained_models, feature_spec, forecast_mode="strong_signal").iloc[0].to_dict()
        minus_pred = predict_v1_demand(step_minus, trained_models, feature_spec, forecast_mode="strong_signal").iloc[0].to_dict()
        main_deltas.append(float(minus_pred["pred_sales_main"] - plus_pred["pred_sales_main"]))
        fallback_deltas.append(float(minus_pred["pred_sales_fallback"] - plus_pred["pred_sales_fallback"]))
        base_pred = float(predict_v1_demand(step, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales"].iloc[0])
        history = pd.concat([history, pd.DataFrame([{"date": pd.Timestamp(row["date"]), "sales": base_pred, **step_ctx}])], ignore_index=True)
    main_sign = int(np.sign(np.mean(main_deltas))) if main_deltas else 0
    fallback_sign = int(np.sign(np.mean(fallback_deltas))) if fallback_deltas else 0
    agree = bool(main_sign != 0 and fallback_sign != 0 and main_sign == fallback_sign)
    return {"main_sign": main_sign, "fallback_sign": fallback_sign, "agree": agree}


def _assess_weak_user_factors(
    trained_models: Dict[str, Any],
    target_train_history: pd.DataFrame,
    feature_spec: Dict[str, Any],
    base_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    hist = target_train_history.copy().sort_values("date")
    tail = hist.tail(min(14, len(hist))).copy()
    seed = hist.iloc[:-len(tail)].copy() if len(hist) > len(tail) else hist.head(0).copy()
    weak: List[str] = []
    diagnostics: Dict[str, Any] = {}
    if len(seed) < 14 or len(tail) == 0:
        return {"weak_factors": list(feature_spec.get("user_numeric_features", [])), "diagnostics": {}}
    hdays = int((seed["date"].max() - seed["date"].min()).days + 1)
    for f in feature_spec.get("user_numeric_features", []):
        s = pd.to_numeric(hist.get(f, np.nan), errors="coerce")
        non_null_share = float(s.notna().mean()) if len(s) else 0.0
        variability_ok = bool(s.nunique(dropna=True) > 1)
        if not variability_ok or non_null_share < 0.60:
            weak.append(f)
            diagnostics[f] = {"non_null_share": non_null_share, "variability_ok": variability_ok, "avg_reaction": 0.0}
            continue
        lo = float(s.quantile(0.25))
        hi = float(s.quantile(0.75))
        local_hist = seed.copy()
        reactions: List[float] = []
        for _, row in tail.iterrows():
            step_ctx = dict(base_ctx)
            for c in feature_spec.get("scenario_features", []):
                step_ctx[c] = float(pd.to_numeric(pd.Series([row.get(c, step_ctx.get(c, 0.0))]), errors="coerce").fillna(0.0).iloc[0])
            step = build_v1_one_step_features(local_hist, pd.Timestamp(row["date"]), step_ctx, hdays, feature_spec)
            low_step = step.copy()
            high_step = step.copy()
            low_step[f] = lo
            high_step[f] = hi
            low_pred = float(predict_v1_demand(low_step, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales_main"].iloc[0])
            high_pred = float(predict_v1_demand(high_step, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales_main"].iloc[0])
            reactions.append(abs(high_pred - low_pred) / max(abs(low_pred), 1e-9))
            base_pred = float(predict_v1_demand(step, trained_models, feature_spec, forecast_mode="strong_signal")["pred_sales"].iloc[0])
            local_hist = pd.concat([local_hist, pd.DataFrame([{"date": pd.Timestamp(row["date"]), "sales": base_pred, **step_ctx}])], ignore_index=True)
        avg_reaction = float(np.mean(reactions)) if reactions else 0.0
        if avg_reaction < 0.01:
            weak.append(f)
        diagnostics[f] = {"non_null_share": non_null_share, "variability_ok": variability_ok, "avg_reaction": avg_reaction}
    return {"weak_factors": sorted(set(weak)), "diagnostics": diagnostics}


def _build_ood_flags(base_ctx: Dict[str, Any], target_history: pd.DataFrame, feature_spec: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    p = pd.to_numeric(target_history.get("price", np.nan), errors="coerce").dropna()
    if len(p) > 10:
        if base_ctx.get("price", 0.0) > float(p.quantile(0.99)):
            flags.append("price_above_p99")
        elif base_ctx.get("price", 0.0) > float(p.quantile(0.95)):
            flags.append("price_above_p95")
        if base_ctx.get("price", 0.0) < float(p.quantile(0.01)):
            flags.append("price_below_p01")
        elif base_ctx.get("price", 0.0) < float(p.quantile(0.05)):
            flags.append("price_below_p05")
    for f in feature_spec.get("user_numeric_features", []):
        s = pd.to_numeric(target_history.get(f, np.nan), errors="coerce").dropna()
        if len(s) < 10:
            continue
        fv = float(base_ctx.get(f, 0.0))
        if fv < float(s.quantile(0.01)) or fv > float(s.quantile(0.99)):
            flags.append(f"{f}_ood")
    return sorted(set(flags))


def run_full_pricing_analysis_universal_v1(normalized_txn: pd.DataFrame, target_category: str, target_sku: str, objective_mode: str = "maximize_profit", horizon_days: int = 30, risk_lambda: float = 0.7, analysis_route: str = "", ui_load_mode: str = "") -> Dict[str, Any]:
    panel_daily = build_daily_panel_from_transactions(normalized_txn.copy())
    panel_features = build_v1_panel_feature_matrix(panel_daily)
    target_history = panel_features[(panel_features["product_id"].astype(str) == str(target_sku)) & (panel_features["category"].astype(str) == str(target_category))].copy().sort_values("date")
    if len(target_history) < 10:
        raise ValueError("Слишком мало данных для target SKU/category.")

    unique_dates = sorted(panel_features["date"].dropna().unique().tolist())
    holdout_len = max(1, int(len(unique_dates) * 0.2))
    holdout_start = pd.Timestamp(unique_dates[-holdout_len])
    panel_train = panel_features[panel_features["date"] < holdout_start].copy()
    panel_test = panel_features[panel_features["date"] >= holdout_start].copy()
    feature_spec = derive_v1_feature_spec(panel_train)

    trained_models_bt = train_v1_demand_model(panel_train, feature_spec, small_mode=len(panel_train) < 200)
    trained_models_final = train_v1_demand_model(panel_features, feature_spec, small_mode=len(panel_features) < 200)

    target_train_history = target_history[target_history["date"] < holdout_start].copy()
    target_test_history = target_history[target_history["date"] >= holdout_start].copy()

    cal_slice = target_train_history.tail(28).copy()
    cal_factor = 1.0
    if len(cal_slice) >= 5:
        seed_hist = target_train_history.iloc[:-len(cal_slice)].copy()
        if len(seed_hist) >= 2:
            base_ctx_cal = _build_base_ctx(seed_hist, feature_spec, target_category, target_sku)
            cal_diag = run_v1_recursive_holdout(seed_hist, cal_slice, trained_models_bt, feature_spec, calibration_factor=1.0, forecast_mode="strong_signal")
            actual_sum = float(pd.to_numeric(cal_slice["sales"], errors="coerce").fillna(0.0).sum())
            pred_sum = float(pd.to_numeric(cal_diag["pred_sales_main"], errors="coerce").fillna(0.0).sum())
            if pred_sum > 1e-9:
                cal_factor = float(np.clip(actual_sum / pred_sum, 0.90, 1.20))

    holdout_diag = run_v1_recursive_holdout(target_train_history, target_test_history, trained_models_bt, feature_spec, calibration_factor=cal_factor, forecast_mode="strong_signal") if len(target_test_history) else pd.DataFrame(columns=["date", "actual_sales", "pred_sales", "pred_sales_main", "pred_sales_fallback", "month", "dow_name"])
    metrics = _metrics(holdout_diag.get("actual_sales", pd.Series(dtype=float)), holdout_diag.get("pred_sales_main", pd.Series(dtype=float)))

    holdout_by_month = _summarize_diag(holdout_diag, "month")
    holdout_by_dow = _summarize_diag(holdout_diag, "dow_name")

    factor_diag = assess_factor_identifiability(target_train_history, feature_spec)
    train_base_ctx = _build_base_ctx(target_train_history, feature_spec, target_category, target_sku)
    perturb = _run_price_perturbation_test(trained_models_bt, target_train_history.iloc[:-min(28, len(target_train_history))].copy(), target_train_history.tail(min(28, len(target_train_history))).copy(), train_base_ctx, feature_spec)
    factor_diag["price_direction_ok"] = bool(perturb["price_direction_ok"])
    factor_diag["price_avg_reaction"] = float(perturb["avg_reaction"])
    factor_diag["price_signal_ok"] = bool(factor_diag["price_variation_ok"] and factor_diag["price_direction_ok"] and len(target_train_history) >= 84 and factor_diag.get("price_avg_reaction", 0.0) >= 0.03)
    weak_eval = _assess_weak_user_factors(trained_models_bt, target_train_history, feature_spec, train_base_ctx)
    factor_diag["weak_factors"] = weak_eval["weak_factors"]
    factor_diag["user_factor_diagnostics"] = weak_eval["diagnostics"]
    base_ctx = _build_base_ctx(target_history, feature_spec, target_category, target_sku)
    future_dates = pd.DataFrame({"date": pd.date_range(pd.Timestamp(target_history["date"].max()) + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")})

    ood_flags = _build_ood_flags(base_ctx, target_history, feature_spec)

    recommendation_reasons: List[str] = []
    can_recommend = True
    if metrics["forecast_wape"] > 35:
        can_recommend = False
        recommendation_reasons.append("high_wape")
    if not (0.9 <= metrics["sum_ratio"] <= 1.1):
        can_recommend = False
        recommendation_reasons.append("sum_ratio_outside_bounds")
    if not factor_diag["price_signal_ok"]:
        can_recommend = False
        recommendation_reasons.append("price_signal_weak")
    hp = pd.to_numeric(target_history["price"], errors="coerce").dropna()
    if len(hp) > 10 and not (float(hp.quantile(0.05)) <= float(base_ctx["price"]) <= float(hp.quantile(0.95))):
        can_recommend = False
        recommendation_reasons.append("current_price_outside_supported_range")
    if ood_flags:
        can_recommend = False
        recommendation_reasons.append("ood")

    hist_days = int((target_history["date"].max() - target_history["date"].min()).days + 1)
    if hist_days < 45 or target_history["sales"].isna().mean() > 0.3 or pd.to_numeric(target_history["price"], errors="coerce").nunique(dropna=True) < 3:
        forecast_mode = "insufficient_data"
    elif can_recommend:
        forecast_mode = "strong_signal"
    else:
        forecast_mode = "weak_signal"

    disagreement = False
    sign_diag = _run_model_price_effect_sign_test(
        trained_models_bt,
        target_train_history.iloc[:-min(28, len(target_train_history))].copy(),
        target_train_history.tail(min(28, len(target_train_history))).copy(),
        train_base_ctx,
        feature_spec,
    )
    disagreement = not bool(sign_diag.get("agree", False))
    if disagreement:
        can_recommend = False
        recommendation_reasons.append("main_fallback_disagreement")

    latest_row = dict(base_ctx)
    rec = recommend_v1_price_horizon(latest_row, trained_models_final, target_history, base_ctx, feature_spec, n_days=int(horizon_days), objective_mode=objective_mode, risk_lambda=float(risk_lambda), can_recommend=can_recommend, price_signal_ok=factor_diag["price_signal_ok"], calibration_factor=cal_factor, forecast_mode=forecast_mode)

    current_sim = simulate_v1_horizon_profit(latest_row, base_ctx["price"], future_dates, trained_models_final, target_history, base_ctx, feature_spec, risk_lambda=float(risk_lambda), calibration_factor=cal_factor, forecast_mode=forecast_mode)
    optimal_price = float(rec.get("best_price", base_ctx["price"]))
    optimal_sim = simulate_v1_horizon_profit(latest_row, optimal_price, future_dates, trained_models_final, target_history, base_ctx, feature_spec, risk_lambda=float(risk_lambda), calibration_factor=cal_factor, forecast_mode=forecast_mode)

    biz_rec = build_business_recommendation(current_price=float(base_ctx["price"]), recommended_price=float(optimal_price if can_recommend else base_ctx["price"]), current_profit=float(current_sim["total_profit"]), recommended_profit=float(optimal_sim["total_profit"] if can_recommend else current_sim["total_profit"]), current_revenue=float(current_sim["total_revenue"]), recommended_revenue=float(optimal_sim["total_revenue"] if can_recommend else current_sim["total_revenue"]), current_volume=float(current_sim["total_volume"]), recommended_volume=float(optimal_sim["total_volume"] if can_recommend else current_sim["total_volume"]), confidence=0.7 if can_recommend else 0.4, elasticity=-1.0, history_days=hist_days, data_quality={"label": forecast_mode, "issues": recommendation_reasons}, base_ctx=base_ctx, reason_hints={})

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        target_history.to_excel(writer, sheet_name="history", index=False)
        current_sim["daily"].to_excel(writer, sheet_name="baseline", index=False)
        optimal_sim["daily"].to_excel(writer, sheet_name="optimal", index=False)
        holdout_diag.to_excel(writer, sheet_name="holdout_diag", index=False)
        holdout_by_month.to_excel(writer, sheet_name="holdout_by_month", index=False)
        holdout_by_dow.to_excel(writer, sheet_name="holdout_by_dow", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="metrics", index=False)
    excel_buffer.seek(0)

    holdout_metrics = dict(metrics)
    holdout_metrics.update({"price_signal_ok": factor_diag["price_signal_ok"], "weak_factors": factor_diag["weak_factors"], "ood_flags": ood_flags, "forecast_mode": forecast_mode, "can_recommend_price": can_recommend})

    return {
        "daily": target_history,
        "recommendation": rec,
        "forecast_current": current_sim["daily"],
        "forecast_optimal": optimal_sim["daily"],
        "profit_curve": pd.DataFrame(rec.get("results", [])),
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "current_price": float(base_ctx["price"]),
        "best_price": float(base_ctx["price"] if not can_recommend else optimal_price),
        "recommended_price": float(base_ctx["price"] if not can_recommend else optimal_price),
        "current_profit": float(current_sim["total_profit"]),
        "best_profit": float(current_sim["total_profit"] if not can_recommend else optimal_sim["total_profit"]),
        "current_profit_adjusted": float(current_sim["adjusted_profit"]),
        "best_profit_adjusted": float(current_sim["adjusted_profit"] if not can_recommend else optimal_sim["adjusted_profit"]),
        "current_revenue": float(current_sim["total_revenue"]),
        "best_revenue": float(current_sim["total_revenue"] if not can_recommend else optimal_sim["total_revenue"]),
        "current_volume": float(current_sim["total_volume"]),
        "best_volume": float(current_sim["total_volume"] if not can_recommend else optimal_sim["total_volume"]),
        "data_quality": {"label": forecast_mode, "issues": recommendation_reasons},
        "excel_buffer": excel_buffer,
        "business_recommendation": biz_rec,
        "analysis_engine": V1_ANALYSIS_ENGINE,
        "analysis_engine_version": V1_ANALYSIS_ENGINE_VERSION,
        "analysis_route": analysis_route,
        "ui_load_mode": ui_load_mode,
        "_trained_bundle": {
            "trained_models": trained_models_final,
            "trained_models_backtest": trained_models_bt,
            "feature_spec": feature_spec,
            "panel_daily": panel_daily,
            "target_history": target_history,
            "daily_base": target_history,
            "base_ctx": base_ctx,
            "future_dates": future_dates,
            "calibration_factor": cal_factor,
            "factor_diagnostics": factor_diag,
            "model_price_effect_sign": sign_diag,
            "data_quality": {"label": forecast_mode, "issues": recommendation_reasons},
            "forecast_mode": forecast_mode,
            "can_recommend_price": can_recommend,
            "recommendation_reasons": recommendation_reasons,
            "risk_lambda": float(risk_lambda),
        },
    }
