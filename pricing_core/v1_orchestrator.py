from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_adapter import build_daily_from_transactions_scoped
from recommendation import build_business_recommendation

from pricing_core.config import CONFIG
from pricing_core.quality import assess_data_quality
from pricing_core.v1_features import build_v1_feature_matrix, derive_v1_feature_spec, build_v1_one_step_features
from pricing_core.v1_forecast import train_v1_demand_model, predict_v1_demand, recursive_v1_demand_forecast
from pricing_core.v1_optimizer import recommend_v1_price_horizon, simulate_v1_horizon_profit

V1_ANALYSIS_ENGINE = "v1_universal"
V1_ANALYSIS_ENGINE_VERSION = "v1_universal_2026_04_route_check"


def _safe_split_sizes(n: int) -> Tuple[int, int]:
    train_end = max(3, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85))
    return train_end, min(val_end, n)


def _eval_wape(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.abs(actual).sum())
    if denom <= 1e-9:
        return 100.0
    return float(np.abs(actual - pred).sum() / denom * 100.0)


def _summarize_holdout_diag(diag: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if len(diag) == 0 or group_col not in diag.columns:
        return pd.DataFrame()
    out = diag.groupby(group_col, dropna=False).agg(actual_mean=("actual_sales", "mean"), pred_mean=("pred_sales", "mean"), abs_error_mean=("abs_error", "mean"), days=("actual_sales", "size")).reset_index()
    out["bias_pct"] = np.where(out["actual_mean"] > 0, (out["pred_mean"] - out["actual_mean"]) / out["actual_mean"], np.nan)
    return out.sort_values(group_col)


def _compute_confidence(wape: float, quality_cap: float, can_recommend: bool) -> float:
    raw_conf = float(np.exp(-max(float(wape), 0.0) / 30.0))
    signal_cap = 1.0 if can_recommend else 0.55
    return float(np.clip(raw_conf * float(quality_cap) * signal_cap, 0.0, 1.0))


def _last_valid(series: pd.Series, default: float) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.dropna().iloc[-1]) if s.notna().any() else float(default)


def _build_base_ctx(daily_base: pd.DataFrame, target_category: str, target_sku: str) -> Dict[str, Any]:
    base = {
        "price": _last_valid(daily_base.get("price", pd.Series(dtype=float)), 0.0),
        "cost": _last_valid(daily_base.get("cost", pd.Series(dtype=float)), 0.0),
        "discount": _last_valid(daily_base.get("discount", daily_base.get("discount_rate", pd.Series(dtype=float))), 0.0),
        "promotion": _last_valid(daily_base.get("promotion", pd.Series(dtype=float)), 0.0),
        "stock": _last_valid(daily_base.get("stock", pd.Series(dtype=float)), 0.0),
        "freight_value": _last_valid(daily_base.get("freight_value", pd.Series(dtype=float)), 0.0),
        "review_score": _last_valid(daily_base.get("review_score", pd.Series(dtype=float)), 4.5),
        "reviews_count": _last_valid(daily_base.get("reviews_count", pd.Series(dtype=float)), 0.0),
        "category": target_category,
        "product_id": target_sku,
    }
    for c in daily_base.columns:
        if str(c).startswith("user_factor__"):
            base[c] = _last_valid(daily_base[c], 0.0)
    return base


def run_v1_recursive_holdout(train_df: pd.DataFrame, test_df: pd.DataFrame, demand_models: list[Any], feature_spec: dict[str, Any]) -> pd.DataFrame:
    history = train_df.copy()
    rows: List[Dict[str, Any]] = []
    scenario_cols = ["price", "discount", "promotion", "stock", "freight_value", "review_score", "reviews_count"] + [c for c in feature_spec.get("scenario_features", []) if str(c).startswith("user_factor__")]
    hdays = int((history["date"].max() - history["date"].min()).days + 1) if len(history) else 1
    for _, tr in test_df.sort_values("date").iterrows():
        step_ctx = {c: float(pd.to_numeric(pd.Series([tr.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0]) for c in scenario_cols}
        step = build_v1_one_step_features(history, pd.Timestamp(tr["date"]), step_ctx, hdays, feature_spec)
        pred_sales = float(predict_v1_demand(step, demand_models, feature_spec)[0])
        rows.append({"date": pd.Timestamp(tr["date"]), "actual_sales": float(tr["sales"]), "pred_sales": pred_sales, "month": str(pd.Timestamp(tr["date"]).to_period("M")), "dow_name": pd.Timestamp(tr["date"]).day_name()})
        history = pd.concat([history, pd.DataFrame([{"date": pd.Timestamp(tr["date"]), "sales": pred_sales, **step_ctx}])], ignore_index=True)
    out = pd.DataFrame(rows)
    if len(out):
        out["residual"] = out["pred_sales"] - out["actual_sales"]
        out["abs_error"] = out["residual"].abs()
    return out


def run_full_pricing_analysis_universal_v1(normalized_txn: pd.DataFrame, target_category: str, target_sku: str, objective_mode: str = "maximize_profit", horizon_days: int = 30, risk_lambda: float = 0.7, analysis_route: str = "", ui_load_mode: str = "") -> Dict[str, Any]:
    daily_base = build_v1_feature_matrix(build_daily_from_transactions_scoped(normalized_txn.copy(), target_sku, target_category)).dropna(subset=["sales", "price"]).reset_index(drop=True)
    if len(daily_base) < 10:
        raise ValueError("Слишком мало данных для v1 анализа.")
    feature_spec = derive_v1_feature_spec(daily_base)

    n = len(daily_base)
    train_end, val_end = _safe_split_sizes(n)
    train_df = daily_base.iloc[:train_end].copy()
    test_df = daily_base.iloc[val_end:].copy()

    demand_models_bt = train_v1_demand_model(train_df, feature_spec, small_mode=len(train_df) < 120)
    demand_models_final = train_v1_demand_model(daily_base, feature_spec, small_mode=len(daily_base) < 120)

    holdout_diag = run_v1_recursive_holdout(train_df, test_df, demand_models_bt, feature_spec) if len(test_df) else pd.DataFrame(columns=["date", "actual_sales", "pred_sales", "month", "dow_name"])
    forecast_wape = _eval_wape(holdout_diag["actual_sales"], holdout_diag["pred_sales"]) if len(holdout_diag) else 100.0
    holdout_metrics = {"forecast_wape": forecast_wape, "e2e_wape": forecast_wape, "wape": forecast_wape}
    holdout_by_month = _summarize_holdout_diag(holdout_diag, "month")
    holdout_by_dow = _summarize_holdout_diag(holdout_diag, "dow_name")

    history_days = int((daily_base["date"].max() - daily_base["date"].min()).days + 1)
    data_quality = assess_data_quality(history_days, len(daily_base), float(normalized_txn.isna().mean().mean()) if len(normalized_txn) else 1.0, forecast_wape)
    can_rec = bool(data_quality.get("can_recommend", True))
    rec_reasons = [] if can_rec else ["data_quality_blocks_recommendation"]

    base_ctx = _build_base_ctx(daily_base, target_category, target_sku)
    latest_row = dict(base_ctx)
    rec = recommend_v1_price_horizon(latest_row, demand_models_final, daily_base, base_ctx, feature_spec, n_days=int(horizon_days), objective_mode=objective_mode, risk_lambda=float(risk_lambda), can_recommend=can_rec)

    future_dates = pd.DataFrame({"date": pd.date_range(pd.Timestamp(daily_base["date"].max()) + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")})
    current_sim = simulate_v1_horizon_profit(latest_row, base_ctx["price"], future_dates, demand_models_final, daily_base, base_ctx, feature_spec, risk_lambda=float(risk_lambda))
    optimal_price = float(rec.get("best_price", base_ctx["price"]))
    optimal_sim = simulate_v1_horizon_profit(latest_row, optimal_price, future_dates, demand_models_final, daily_base, base_ctx, feature_spec, risk_lambda=float(risk_lambda))

    confidence = _compute_confidence(forecast_wape, float(data_quality.get("confidence_cap", 0.5)), can_rec)
    biz_rec = build_business_recommendation(current_price=float(base_ctx["price"]), recommended_price=float(optimal_price if can_rec else base_ctx["price"]), current_profit=float(current_sim["total_profit"]), recommended_profit=float(optimal_sim["total_profit"] if can_rec else current_sim["total_profit"]), current_revenue=float(current_sim["total_revenue"]), recommended_revenue=float(optimal_sim["total_revenue"] if can_rec else current_sim["total_revenue"]), current_volume=float(current_sim["total_volume"]), recommended_volume=float(optimal_sim["total_volume"] if can_rec else current_sim["total_volume"]), confidence=confidence, elasticity=-1.0, history_days=history_days, data_quality=data_quality, base_ctx=base_ctx, reason_hints={})

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        daily_base.to_excel(writer, sheet_name="history", index=False)
        current_sim["daily"].to_excel(writer, sheet_name="baseline", index=False)
        optimal_sim["daily"].to_excel(writer, sheet_name="optimal", index=False)
        holdout_diag.to_excel(writer, sheet_name="holdout_diag", index=False)
        holdout_by_month.to_excel(writer, sheet_name="holdout_by_month", index=False)
        holdout_by_dow.to_excel(writer, sheet_name="holdout_by_dow", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="metrics", index=False)
    excel_buffer.seek(0)

    current_profit_raw = float(current_sim["total_profit"])
    best_profit_raw = float(current_sim["total_profit"] if not can_rec else optimal_sim["total_profit"])
    return {
        "daily": daily_base,
        "recommendation": rec,
        "forecast_current": current_sim["daily"],
        "forecast_optimal": optimal_sim["daily"],
        "profit_curve": pd.DataFrame(rec.get("results", [])),
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "current_price": float(base_ctx["price"]),
        "best_price": float(base_ctx["price"] if not can_rec else optimal_price),
        "current_profit": current_profit_raw,
        "best_profit": best_profit_raw,
        "current_profit_adjusted": float(current_sim["adjusted_profit"]),
        "best_profit_adjusted": float(current_sim["adjusted_profit"] if not can_rec else optimal_sim["adjusted_profit"]),
        "current_revenue": float(current_sim["total_revenue"]),
        "best_revenue": float(current_sim["total_revenue"] if not can_rec else optimal_sim["total_revenue"]),
        "current_volume": float(current_sim["total_volume"]),
        "best_volume": float(current_sim["total_volume"] if not can_rec else optimal_sim["total_volume"]),
        "profit_lift_pct": float(((best_profit_raw - current_profit_raw) / max(abs(current_profit_raw), 1e-9) * 100.0) if can_rec else 0.0),
        "data_quality": data_quality,
        "excel_buffer": excel_buffer,
        "business_recommendation": biz_rec,
        "analysis_engine": V1_ANALYSIS_ENGINE,
        "analysis_engine_version": V1_ANALYSIS_ENGINE_VERSION,
        "analysis_route": analysis_route,
        "ui_load_mode": ui_load_mode,
        "_trained_bundle": {
            "demand_models": demand_models_final,
            "demand_models_backtest": demand_models_bt,
            "feature_spec": feature_spec,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "price_signal": {},
            "confidence": confidence,
            "data_quality": data_quality,
            "can_recommend_price": can_rec,
            "recommendation_reasons": rec_reasons,
            "risk_lambda": float(risk_lambda),
        },
    }
