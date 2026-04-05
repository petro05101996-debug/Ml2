from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from calc_engine import compute_daily_unit_economics
from data_adapter import build_daily_panel_from_transactions
from pricing_core.baseline_features import build_baseline_feature_matrix, derive_baseline_feature_spec
from pricing_core.baseline_model import (
    build_baseline_oof_predictions,
    build_weekday_profile,
    build_weekly_baseline_oof_predictions,
    disaggregate_weekly_to_daily,
    forecast_weekly_baseline_by_strategy,
    forecast_baseline_by_strategy,
    aggregate_daily_to_weekly,
    recursive_baseline_forecast,
    run_baseline_rolling_backtest,
    select_best_baseline_plan,
    train_baseline_model,
    week_start,
)
from pricing_core.factor_features import build_factor_feature_matrix, build_factor_target, derive_factor_feature_spec
from pricing_core.factor_model import compute_factor_contributions, run_factor_rolling_backtest, train_factor_model
from pricing_core.model_registry import build_model_bundle
from pricing_core.scenario_engine import compute_scenario_delta, run_scenario_forecast
from pricing_core.uncertainty import (
    build_baseline_confidence_state,
    build_factor_confidence_state,
    build_shock_confidence_state,
    combine_confidence_states,
)


def _tiny_baseline_fallback(target_history: pd.DataFrame, future_dates: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(target_history.get("sales", 0.0), errors="coerce").fillna(0.0)
    if len(s) == 0:
        base = 0.0
    elif len(s) < 7:
        base = float(s.mean())
    else:
        base = float(s.tail(7).median())
    return pd.DataFrame(
        {
            "date": pd.to_datetime(future_dates["date"]),
            "baseline_pred": max(0.0, base),
            "baseline_lower": np.nan,
            "baseline_upper": np.nan,
        }
    )


def _assess_factor_trainability(factor_train: pd.DataFrame, factor_feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    n_rows = int(len(factor_train))
    cont = factor_feature_spec.get("controllable_features", [])
    variative = [
        c
        for c in cont
        if c in factor_train.columns and pd.to_numeric(factor_train[c], errors="coerce").nunique(dropna=True) > 1
    ]
    price_unique_count = int(pd.to_numeric(factor_train.get("price", np.nan), errors="coerce").nunique(dropna=True))
    variative_count = int(len(variative))
    trainable = n_rows >= 45 and variative_count >= 1 and (price_unique_count >= 3 or variative_count >= 2)
    reason_codes = []
    if n_rows < 45:
        reason_codes.append("rows_lt_45")
    if variative_count < 1:
        reason_codes.append("no_variative_controllable")
    if price_unique_count < 3 and variative_count < 2:
        reason_codes.append("weak_price_and_control_variation")
    return {
        "trainable": bool(trainable),
        "n_rows": n_rows,
        "price_unique_count": price_unique_count,
        "variative_controllable_count": variative_count,
        "reason_codes": reason_codes,
    }


def _build_empty_baseline_rolling() -> Dict[str, Any]:
    return {
        "rolling_diag": pd.DataFrame(),
        "rolling_metrics": pd.DataFrame(),
        "rolling_summary": {
            "n_valid_windows": 0,
            "median_wape": np.nan,
            "median_bias_pct": np.nan,
            "median_sum_ratio": np.nan,
            "max_wape": np.nan,
        },
    }


def _build_pooled_factor_train(panel_train: pd.DataFrame, target_category: str, max_skus: int = 8) -> Dict[str, Any]:
    rows = []
    used_skus: List[str] = []
    scope_panel = panel_train[panel_train["category"].astype(str) == str(target_category)].copy()
    if scope_panel.empty:
        return {"factor_train": pd.DataFrame(), "pooled_skus_used": [], "pooled_rows_used": 0}

    sku_rank = (
        scope_panel.groupby("product_id", dropna=False)
        .agg(
            n_rows=("date", "size"),
            price_unique=("price", lambda s: pd.to_numeric(s, errors="coerce").nunique(dropna=True)),
            non_na_share=("sales", lambda s: float(pd.Series(s).notna().mean())),
        )
        .reset_index()
    )
    sku_rank = sku_rank.sort_values(["price_unique", "n_rows", "non_na_share"], ascending=False)

    for sku in sku_rank["product_id"].astype(str).dropna().head(max_skus):
        selected = select_best_baseline_plan(scope_panel, target_category, sku)
        sku_strategy = str(selected.get("selected_strategy", "xgb_recursive"))
        sku_granularity = str(selected.get("granularity", "daily"))
        sku_hist = scope_panel[scope_panel["product_id"].astype(str) == str(sku)].copy()
        if sku_granularity == "weekly":
            oof = build_weekly_baseline_oof_predictions(sku_hist, strategy=sku_strategy)
        else:
            oof = build_baseline_oof_predictions(scope_panel, target_category, sku, strategy=sku_strategy)
        if oof.empty:
            continue
        merged = sku_hist.merge(oof[["date", "baseline_oof"]], on="date", how="left")
        if merged["baseline_oof"].notna().sum() < 10:
            continue
        rows.append(merged)
        used_skus.append(str(sku))

    factor_train = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return {"factor_train": factor_train, "pooled_skus_used": used_skus, "pooled_rows_used": int(len(factor_train))}


def run_full_pricing_analysis_v2(normalized_txn: pd.DataFrame, target_category: str, target_sku: str, horizon_days: int = 30, scenario_overrides: Dict[str, Any] | None = None, shocks: List[Dict[str, Any]] | None = None, objective_mode: str = "maximize_profit") -> Dict[str, Any]:
    panel_daily = build_daily_panel_from_transactions(normalized_txn)
    panel_features_baseline = build_baseline_feature_matrix(panel_daily)
    target_history = panel_features_baseline[(panel_features_baseline["category"].astype(str) == str(target_category)) & (panel_features_baseline["product_id"].astype(str) == str(target_sku))].copy().sort_values("date")
    if target_history.empty:
        raise ValueError("Target history is empty")

    unique_dates = sorted(panel_features_baseline["date"].dropna().unique())
    holdout_n = max(1, int(np.ceil(len(unique_dates) * 0.2)))
    holdout_start = pd.Timestamp(unique_dates[-holdout_n])
    panel_train = panel_features_baseline[panel_features_baseline["date"] < holdout_start].copy()
    target_train = target_history[target_history["date"] < holdout_start].copy()

    baseline_feature_spec_train = derive_baseline_feature_spec(panel_train)
    baseline_feature_spec_full = derive_baseline_feature_spec(panel_features_baseline)

    future_dates = pd.DataFrame({"date": pd.date_range(pd.Timestamp(target_history["date"].max()) + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")})

    tiny_mode = target_history["date"].nunique() < 14
    trained_baseline_bt = None
    trained_baseline_final = None
    best_baseline_strategy = "xgb_recursive"
    baseline_granularity = "daily"
    baseline_selector_reason = "daily selected: tiny_mode fallback"
    baseline_plan_selection = {"granularity": "daily", "selected_strategy": "median7"}
    baseline_strategy_selection = {"best_strategy": "xgb_recursive", "strategy_metrics": pd.DataFrame(), "strategy_summary": pd.DataFrame()}
    best_daily_strategy = "xgb_recursive"
    best_weekly_strategy = "weekly_median4w"

    if tiny_mode:
        baseline_forecast = _tiny_baseline_fallback(target_history, future_dates)
        baseline_rolling = _build_empty_baseline_rolling()
        baseline_oof = target_train[["date", "sales"]].copy()
        baseline_oof["baseline_oof"] = np.nan
        best_baseline_strategy = "median7"
        baseline_granularity = "daily"
        baseline_selector_reason = "daily selected: tiny_mode fallback"
        baseline_plan_selection = {
            "granularity": "daily",
            "daily_selection": baseline_strategy_selection,
            "weekly_selection": {"strategy_metrics": pd.DataFrame(), "strategy_summary": pd.DataFrame(), "best_strategy": best_weekly_strategy},
            "best_daily_strategy": best_baseline_strategy,
            "best_weekly_strategy": best_weekly_strategy,
            "selected_strategy": best_baseline_strategy,
            "selector_reason": baseline_selector_reason,
        }
        baseline_strategy_selection["best_strategy"] = best_baseline_strategy
    else:
        baseline_plan_selection = select_best_baseline_plan(panel_train, target_category, target_sku)
        baseline_granularity = str(baseline_plan_selection.get("granularity", "daily"))
        best_baseline_strategy = str(baseline_plan_selection.get("selected_strategy", "xgb_recursive"))
        baseline_selector_reason = str(baseline_plan_selection.get("selector_reason", "daily selected"))
        baseline_strategy_selection = baseline_plan_selection.get("daily_selection", baseline_strategy_selection)
        best_daily_strategy = str(baseline_plan_selection.get("best_daily_strategy", baseline_strategy_selection.get("best_strategy", "xgb_recursive")))
        best_weekly_strategy = str(baseline_plan_selection.get("best_weekly_strategy", best_weekly_strategy))

        base_ctx = {
            c: (target_history[c].dropna().astype(str).iloc[-1] if c in ["product_id", "category", "region", "channel", "segment"] and c in target_history else "unknown")
            for c in baseline_feature_spec_full.get("baseline_context_features", [])
        }
        if baseline_granularity == "weekly":
            trained_baseline_bt = None
            trained_baseline_final = None
            weekly_history = aggregate_daily_to_weekly(target_history)
            weekday_profile = build_weekday_profile(target_history)
            future_week_starts = pd.DataFrame({"week_start": sorted(week_start(future_dates["date"]).dropna().unique())})
            weekly_forecast = forecast_weekly_baseline_by_strategy(best_baseline_strategy, weekly_history, future_week_starts)
            baseline_forecast = disaggregate_weekly_to_daily(weekly_forecast, future_dates[["date"]], weekday_profile)
            baseline_oof = build_weekly_baseline_oof_predictions(target_train, strategy=best_baseline_strategy)
            baseline_rolling = baseline_plan_selection.get("weekly_selection", _build_empty_baseline_rolling())
        elif best_baseline_strategy == "xgb_recursive":
            trained_baseline_bt = train_baseline_model(panel_train, baseline_feature_spec_train, small_mode=len(panel_train) < 200, training_profile="backtest")
            trained_baseline_final = train_baseline_model(panel_features_baseline, baseline_feature_spec_full, small_mode=len(panel_features_baseline) < 200, training_profile="final")
            baseline_forecast = recursive_baseline_forecast(trained_baseline_final, target_history, future_dates, base_ctx, baseline_feature_spec_full)
            baseline_rolling = run_baseline_rolling_backtest(panel_train, target_category, target_sku, strategy=best_baseline_strategy)
            baseline_oof = build_baseline_oof_predictions(panel_train, target_category, target_sku, strategy=best_baseline_strategy)
        else:
            trained_baseline_bt = None
            trained_baseline_final = None
            baseline_forecast = forecast_baseline_by_strategy(best_baseline_strategy, None, target_history, future_dates, base_ctx, baseline_feature_spec_full)
            baseline_rolling = run_baseline_rolling_backtest(panel_train, target_category, target_sku, strategy=best_baseline_strategy)
            baseline_oof = build_baseline_oof_predictions(panel_train, target_category, target_sku, strategy=best_baseline_strategy)
        baseline_forecast["baseline_lower"] = np.nan
        baseline_forecast["baseline_upper"] = np.nan

    factor_train_target = target_train.merge(baseline_oof[["date", "baseline_oof"]], on="date", how="left")
    factor_feature_spec_target = derive_factor_feature_spec(factor_train_target)
    factor_train_target = build_factor_feature_matrix(factor_train_target, factor_feature_spec_target)
    factor_train_target = factor_train_target[factor_train_target["baseline_oof"].notna()].copy()
    factor_train_target["factor_target"] = build_factor_target(factor_train_target)
    target_assess = _assess_factor_trainability(factor_train_target, factor_feature_spec_target)

    use_target_direct = (
        target_assess["n_rows"] >= 120
        and target_assess["price_unique_count"] >= 6
        and target_assess["variative_controllable_count"] >= 2
    )

    factor_train_scope = "none"
    factor_train = pd.DataFrame()
    factor_feature_spec = factor_feature_spec_target
    pooled_skus_used: List[str] = []
    pooled_rows_used = 0

    if target_assess["trainable"] and use_target_direct:
        factor_train_scope = "target"
        factor_train = factor_train_target
    else:
        pooled_info = _build_pooled_factor_train(panel_train, target_category)
        pooled_candidate = pooled_info.get("factor_train", pd.DataFrame())
        pooled_skus_used = pooled_info.get("pooled_skus_used", [])
        pooled_rows_used = int(pooled_info.get("pooled_rows_used", 0))
        if not pooled_candidate.empty:
            factor_feature_spec = derive_factor_feature_spec(pooled_candidate)
            pooled_candidate = build_factor_feature_matrix(pooled_candidate, factor_feature_spec)
            pooled_candidate = pooled_candidate[pooled_candidate["baseline_oof"].notna()].copy()
            pooled_candidate["factor_target"] = build_factor_target(pooled_candidate)
            pooled_assess = _assess_factor_trainability(pooled_candidate, factor_feature_spec)
            if pooled_assess["trainable"]:
                factor_train_scope = "pooled"
                factor_train = pooled_candidate
            elif target_assess["trainable"]:
                factor_train_scope = "target"
                factor_train = factor_train_target
                factor_feature_spec = factor_feature_spec_target
        elif target_assess["trainable"]:
            factor_train_scope = "target"
            factor_train = factor_train_target
            factor_feature_spec = factor_feature_spec_target

    factor_train_stats = _assess_factor_trainability(factor_train if not factor_train.empty else factor_train_target, factor_feature_spec_target if factor_train.empty else factor_feature_spec)
    factor_model_trained = factor_train_scope in {"target", "pooled"} and len(factor_train) > 0

    if factor_model_trained:
        trained_factor = train_factor_model(factor_train, factor_feature_spec, small_mode=len(factor_train) < 200, training_profile="final")
        factor_backtest = run_factor_rolling_backtest(factor_train, factor_feature_spec)
    else:
        trained_factor = None
        factor_backtest = {"trained": False, "reason": "factor signal insufficient", "n_valid_windows": 0, "factor_ood_share": 0.0}

    factor_backtest.update(
        {
            "n_train_rows": int(len(factor_train)),
            "price_unique_count": int(factor_train_stats["price_unique_count"]),
            "variative_controllable_count": int(factor_train_stats["variative_controllable_count"]),
            "train_scope": factor_train_scope,
            "pooled_skus_used": pooled_skus_used,
            "pooled_rows_used": pooled_rows_used,
        }
    )

    use_baseline_override = tiny_mode or (baseline_granularity == "weekly") or (best_baseline_strategy != "xgb_recursive")
    scenario_result = run_scenario_forecast(
        trained_baseline=trained_baseline_final,
        trained_factor=trained_factor,
        base_history=target_history,
        future_dates_df=future_dates,
        baseline_feature_spec=baseline_feature_spec_full,
        factor_feature_spec=factor_feature_spec if factor_model_trained else None,
        scenario_overrides=scenario_overrides,
        shocks=shocks,
        baseline_override_df=baseline_forecast[["date", "baseline_pred"]] if use_baseline_override else None,
    )
    scenario_forecast = scenario_result["scenario_forecast"].copy()

    base_ctx = scenario_result["base_ctx"]
    scn_ctx = scenario_result["scenario_ctx"]

    b_input = baseline_forecast.copy()
    b_input["price"] = max(float(base_ctx.get("price", target_history.get("price", pd.Series([1.0])).iloc[-1])), 1e-6)
    b_input["discount"] = float(base_ctx.get("discount", target_history.get("discount", pd.Series([0.0])).iloc[-1]))
    b_input["cost"] = float(base_ctx.get("cost", target_history.get("cost", pd.Series([0.0])).iloc[-1]))
    b_input["freight_value"] = float(base_ctx.get("freight_value", target_history.get("freight_value", pd.Series([0.0])).iloc[-1]))
    baseline_economics, _ = compute_daily_unit_economics(b_input, quantity_col="baseline_pred")

    s_input = scenario_forecast.copy()
    s_input["price"] = max(float(scn_ctx.get("price", b_input["price"].iloc[0])), 1e-6)
    s_input["discount"] = float(scn_ctx.get("discount", b_input["discount"].iloc[0]))
    s_input["cost"] = float(scn_ctx.get("cost", b_input["cost"].iloc[0]))
    s_input["freight_value"] = float(scn_ctx.get("freight_value", b_input["freight_value"].iloc[0]))
    scenario_economics, _ = compute_daily_unit_economics(s_input, quantity_col="actual_sales")

    demand_delta = compute_scenario_delta(baseline_forecast, scenario_forecast)
    baseline_total_revenue = float(baseline_economics["total_revenue"].sum())
    scenario_total_revenue = float(scenario_economics["total_revenue"].sum())
    baseline_total_profit = float(baseline_economics["profit"].sum())
    scenario_total_profit = float(scenario_economics["profit"].sum())
    delta_summary = demand_delta.copy()
    delta_summary["baseline_total_revenue"] = baseline_total_revenue
    delta_summary["scenario_total_revenue"] = scenario_total_revenue
    delta_summary["revenue_delta_abs"] = scenario_total_revenue - baseline_total_revenue
    delta_summary["revenue_delta_pct"] = 0.0 if abs(baseline_total_revenue) < 1e-9 else (scenario_total_revenue - baseline_total_revenue) / baseline_total_revenue
    delta_summary["baseline_total_profit"] = baseline_total_profit
    delta_summary["scenario_total_profit"] = scenario_total_profit
    delta_summary["profit_delta_abs"] = scenario_total_profit - baseline_total_profit
    delta_summary["profit_delta_pct"] = 0.0 if abs(baseline_total_profit) < 1e-9 else (scenario_total_profit - baseline_total_profit) / baseline_total_profit

    factor_contributions = compute_factor_contributions(target_history, future_dates, base_ctx, scn_ctx, trained_factor, factor_feature_spec if factor_model_trained else {"controllable_features": []}) if factor_model_trained else pd.DataFrame(columns=["factor_name", "multiplier_delta", "contribution_pct", "confidence", "note"])

    baseline_conf = build_baseline_confidence_state(baseline_rolling.get("rolling_summary", {}))
    factor_conf = build_factor_confidence_state(factor_backtest, ood_flags=scenario_result.get("ood_flags", []))
    shock_conf = build_shock_confidence_state(shocks)
    confidence = combine_confidence_states(baseline_conf, factor_conf, shock_conf, intervals_available=False)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        target_history.to_excel(writer, sheet_name="history", index=False)
        baseline_forecast.to_excel(writer, sheet_name="baseline_forecast", index=False)
        scenario_forecast.to_excel(writer, sheet_name="scenario_forecast", index=False)
        baseline_economics.to_excel(writer, sheet_name="baseline_economics", index=False)
        scenario_economics.to_excel(writer, sheet_name="scenario_economics", index=False)
        delta_summary.to_excel(writer, sheet_name="delta_summary", index=False)
        baseline_rolling.get("rolling_metrics", pd.DataFrame()).to_excel(writer, sheet_name="baseline_rolling_metrics", index=False)
        baseline_rolling.get("rolling_diag", pd.DataFrame()).to_excel(writer, sheet_name="baseline_rolling_diag", index=False)
        (pd.DataFrame([factor_backtest]) if isinstance(factor_backtest, dict) else pd.DataFrame(factor_backtest)).to_excel(writer, sheet_name="factor_backtest", index=False)
        factor_contributions.to_excel(writer, sheet_name="factor_contributions", index=False)
        pd.DataFrame([confidence]).to_excel(writer, sheet_name="confidence", index=False)
    excel_buffer.seek(0)

    mode = scenario_result.get("mode", "baseline_only")
    ood_flags = scenario_result.get("ood_flags", [])
    intervals_available = False


    scenario_feature_spec = {
        "user_numeric_features": [c for c in (factor_feature_spec or {}).get("controllable_features", []) if str(c).startswith("user_factor_num__")],
        "user_categorical_features": [c for c in (factor_feature_spec or {}).get("controllable_features", []) if str(c).startswith("user_factor_cat__")],
        "supported_controls": list((factor_feature_spec or {}).get("controllable_features", [])),
    }

    bundle = build_model_bundle(
        trained_baseline_bt=trained_baseline_bt,
        trained_baseline_final=trained_baseline_final,
        trained_factor=trained_factor,
        baseline_feature_spec_train=baseline_feature_spec_train,
        baseline_feature_spec_full=baseline_feature_spec_full,
        factor_feature_spec=factor_feature_spec,
        baseline_rolling_backtest=baseline_rolling,
        factor_backtest=factor_backtest,
        baseline_oof=baseline_oof,
        target_history=target_history,
        panel_daily=panel_daily,
        future_dates=future_dates,
        confidence=confidence,
        warnings=scenario_result.get("warnings", []) + ood_flags,
        model_backend_info={"baseline_bt": (trained_baseline_bt or {}).get("model_backend"), "baseline_final": (trained_baseline_final or {}).get("model_backend"), "factor": trained_factor.get("model_backend") if trained_factor else None},
        engine_version="v2_decomposed_baseline_factor_shock",
        factor_train_scope=factor_train_scope,
        factor_train_rows=int(len(factor_train)),
        baseline_strategy=best_baseline_strategy,
        baseline_strategy_selection=baseline_strategy_selection,
        baseline_granularity=baseline_granularity,
        baseline_plan_selection=baseline_plan_selection,
        best_daily_strategy=best_daily_strategy,
        best_weekly_strategy=best_weekly_strategy,
        baseline_selector_reason=baseline_selector_reason,
        mode=mode,
        ood_flags=ood_flags,
        intervals_available=intervals_available,
        pooled_skus_used=pooled_skus_used,
        pooled_rows_used=pooled_rows_used,
        base_ctx=base_ctx,
        baseline_forecast=baseline_forecast[["date", "baseline_pred", "baseline_lower", "baseline_upper"]].copy(),
        scenario_feature_spec=scenario_feature_spec,
        feature_spec=scenario_feature_spec,
        current_price=float(base_ctx.get("price", target_history.get("price", pd.Series([1.0])).iloc[-1])),
        forecast_horizon_days=int(horizon_days),
        objective_mode=str(objective_mode),
    )

    return {
        "history": target_history,
        "baseline_forecast": baseline_forecast[["date", "baseline_pred", "baseline_lower", "baseline_upper"]],
        "scenario_forecast": scenario_forecast,
        "baseline_economics": baseline_economics,
        "scenario_economics": scenario_economics,
        "delta_summary": delta_summary,
        "factor_contributions": factor_contributions,
        "confidence": confidence,
        "excel_buffer": excel_buffer,
        "analysis_engine": "v2_decomposed_baseline_factor_shock",
        "factor_model_trained": bool(factor_model_trained),
        "factor_train_scope": factor_train_scope,
        "factor_train_rows": int(len(factor_train)),
        "baseline_strategy": best_baseline_strategy,
        "baseline_granularity": baseline_granularity,
        "baseline_plan_selection": baseline_plan_selection,
        "best_daily_strategy": best_daily_strategy,
        "best_weekly_strategy": best_weekly_strategy,
        "baseline_selector_reason": baseline_selector_reason,
        "baseline_strategy_selection": baseline_strategy_selection,
        "mode": mode,
        "ood_flags": ood_flags,
        "intervals_available": intervals_available,
        "_trained_bundle": bundle,
    }
