from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from calc_engine import compute_daily_unit_economics
from data_adapter import SERIES_SCOPE_COLS, build_baseline_data_quality_summary, build_daily_panel_from_transactions, build_series_id
from pricing_core.baseline_features import build_baseline_feature_matrix, derive_baseline_feature_spec
from pricing_core.baseline_model import (
    build_baseline_oof_predictions,
    build_baseline_quality_summary,
    build_weekday_profile,
    build_weekly_baseline_oof_predictions,
    disaggregate_weekly_to_daily,
    forecast_weekly_baseline_by_strategy,
    forecast_baseline_by_strategy,
    aggregate_daily_to_weekly,
    recursive_baseline_forecast,
    run_baseline_benchmark_suite,
    run_baseline_rolling_backtest,
    select_baseline_from_benchmark_suite,
    select_best_baseline_plan,
    train_baseline_model,
    week_start,
)
from pricing_core.factor_features import build_factor_feature_matrix, build_factor_target_frame, derive_factor_feature_spec
from pricing_core.factor_model import (
    compute_current_state_contributions,
    compute_scenario_delta_contributions,
    run_factor_rolling_backtest,
    train_factor_model,
)
from pricing_core.model_registry import build_model_bundle
from pricing_core.scenario_engine import run_scenario_forecast
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
    trainable_for_advisory = n_rows >= 60 and variative_count >= 1 and (price_unique_count >= 3 or variative_count >= 2)
    trainable_for_production = n_rows >= 120 and variative_count >= 2 and price_unique_count >= 5
    reason_codes = []
    if n_rows < 60:
        reason_codes.append("rows_lt_60")
    if variative_count < 1:
        reason_codes.append("no_variative_controllable")
    if price_unique_count < 3 and variative_count < 2:
        reason_codes.append("weak_price_and_control_variation")
    if trainable_for_production:
        factor_role = "production"
    elif trainable_for_advisory:
        factor_role = "advisory_only"
    else:
        factor_role = "unavailable"
    return {
        "trainable": bool(trainable_for_advisory),
        "trainable_for_advisory": bool(trainable_for_advisory),
        "trainable_for_production": bool(trainable_for_production),
        "factor_role": factor_role,
        "n_rows": n_rows,
        "price_unique_count": price_unique_count,
        "variative_controllable_count": variative_count,
        "reason_codes": reason_codes,
    }


def _build_empty_baseline_rolling() -> Dict[str, Any]:
    return {
        "rolling_diag": pd.DataFrame(),
        "rolling_metrics": pd.DataFrame(
            columns=[
                "window_id",
                "window_start",
                "window_end",
                "forecast_wape",
                "mae",
                "rmse",
                "bias_pct",
                "sum_ratio",
                "pred_std",
                "actual_std",
                "std_ratio",
                "pred_nunique",
                "actual_nunique",
                "is_flat_forecast",
                "weekday_shape_error",
            ]
        ),
        "rolling_summary": {
            "n_valid_windows": 0,
            "median_wape": np.nan,
            "median_bias_pct": np.nan,
            "median_sum_ratio": np.nan,
            "max_wape": np.nan,
        },
    }


def _build_pooled_factor_train(panel_train: pd.DataFrame, target_category: str, max_series: int = 8) -> Dict[str, Any]:
    rows = []
    used_series: List[str] = []
    scope_panel = panel_train[panel_train["category"].astype(str) == str(target_category)].copy()
    if scope_panel.empty:
        return {"factor_train": pd.DataFrame(), "pooled_series_used": [], "pooled_rows_used": 0}

    sku_rank = (
        scope_panel.groupby("series_id", dropna=False)
        .agg(
            n_rows=("date", "size"),
            price_unique=("price", lambda s: pd.to_numeric(s, errors="coerce").nunique(dropna=True)),
            non_na_share=("sales", lambda s: float(pd.Series(s).notna().mean())),
        )
        .reset_index()
    )
    sku_rank = sku_rank.sort_values(["price_unique", "n_rows", "non_na_share"], ascending=False)

    for series_id in sku_rank["series_id"].astype(str).dropna().head(max_series):
        series_hist = scope_panel[scope_panel["series_id"].astype(str) == str(series_id)].copy()
        if series_hist.empty:
            continue
        sku = str(series_hist["product_id"].astype(str).iloc[0])
        selected = select_best_baseline_plan(scope_panel, target_category, sku, target_series_id=str(series_id))
        sku_strategy = str(selected.get("selected_strategy", "xgb_recursive"))
        sku_granularity = str(selected.get("granularity", "daily"))
        sku_hist = series_hist.copy()
        if sku_granularity == "weekly":
            oof = build_weekly_baseline_oof_predictions(sku_hist, strategy=sku_strategy)
        else:
            oof = build_baseline_oof_predictions(scope_panel, target_category, sku, target_series_id=str(series_id), strategy=sku_strategy)
        if oof.empty:
            continue
        merged = sku_hist.merge(oof[["date", "baseline_oof"]], on="date", how="left")
        if merged["baseline_oof"].notna().sum() < 10:
            continue
        rows.append(merged)
        used_series.append(str(series_id))

    factor_train = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return {"factor_train": factor_train, "pooled_series_used": used_series, "pooled_rows_used": int(len(factor_train))}


def run_full_pricing_analysis_v2(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    horizon_days: int = 30,
    scenario_overrides: Dict[str, Any] | None = None,
    shocks: List[Dict[str, Any]] | None = None,
    objective_mode: str = "maximize_profit",
    target_series_id: str | None = None,
    unit_price_input_type: str = "net",
    economics_mode: str = "net_price",
) -> Dict[str, Any]:
    panel_daily = build_daily_panel_from_transactions(normalized_txn)
    if "series_id" not in panel_daily.columns:
        panel_daily["series_id"] = build_series_id(panel_daily)
    panel_features_baseline = build_baseline_feature_matrix(panel_daily)
    if target_series_id is None:
        selector = panel_features_baseline[
            (panel_features_baseline["category"].astype(str) == str(target_category))
            & (panel_features_baseline["product_id"].astype(str) == str(target_sku))
        ].copy()
        if selector.empty:
            raise ValueError("Target history is empty")
        target_series_id = str(selector["series_id"].astype(str).mode().iloc[0])
    target_history = panel_features_baseline[
        panel_features_baseline["series_id"].astype(str) == str(target_series_id)
    ].copy().sort_values("date")
    if target_history.empty:
        raise ValueError("Target history is empty")
    target_category = str(target_history["category"].astype(str).iloc[0]) if "category" in target_history.columns else str(target_category)
    target_sku = str(target_history["product_id"].astype(str).iloc[0]) if "product_id" in target_history.columns else str(target_sku)
    baseline_data_quality = build_baseline_data_quality_summary(target_history)

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
    baseline_benchmark_suite = pd.DataFrame()
    baseline_quality_gate: Dict[str, Any] = {}
    best_daily_strategy = "xgb_recursive"
    best_weekly_strategy = "weekly_median4w"
    baseline_selection_result: Dict[str, Any] = {}
    plan_selected_strategy = "median7"
    plan_selected_granularity = "daily"

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
        plan_selected_strategy = str(baseline_plan_selection.get("selected_strategy", best_baseline_strategy))
        plan_selected_granularity = str(baseline_plan_selection.get("granularity", baseline_granularity))
        baseline_strategy_selection["best_strategy"] = best_baseline_strategy
        baseline_benchmark_suite = pd.DataFrame(
            [
                {
                    "strategy": best_baseline_strategy,
                    "granularity": "daily",
                    "composite_score": np.nan,
                    "goal_wape_median_le_25": False,
                    "goal_wape_max_le_35": False,
                    "goal_abs_bias_le_7pct": False,
                    "goal_sum_ratio_in_range": False,
                    "goal_std_ratio_ge_055": False,
                    "acceptance_pass": False,
                }
            ]
        )
        baseline_selection_result = {
            "best_available_strategy": best_baseline_strategy,
            "best_available_granularity": baseline_granularity,
            "baseline_meets_quality_gate": False,
            "baseline_rejection_reason": "tiny_history",
            "runner_up_strategy": "",
            "runner_up_score": np.nan,
        }
    else:
        baseline_plan_selection = select_best_baseline_plan(panel_train, target_category, target_sku, target_series_id=str(target_series_id))
        plan_selected_strategy = str(baseline_plan_selection.get("selected_strategy", "xgb_recursive"))
        plan_selected_granularity = str(baseline_plan_selection.get("granularity", "daily"))
        baseline_granularity = str(baseline_plan_selection.get("granularity", "daily"))
        best_baseline_strategy = str(baseline_plan_selection.get("selected_strategy", "xgb_recursive"))
        baseline_selector_reason = str(baseline_plan_selection.get("selector_reason", "daily selected"))
        baseline_strategy_selection = baseline_plan_selection.get("daily_selection", baseline_strategy_selection)
        best_daily_strategy = str(baseline_plan_selection.get("best_daily_strategy", baseline_strategy_selection.get("best_strategy", "xgb_recursive")))
        best_weekly_strategy = str(baseline_plan_selection.get("best_weekly_strategy", best_weekly_strategy))
        baseline_benchmark_suite = run_baseline_benchmark_suite(
            panel_train=panel_train,
            target_category=target_category,
            target_sku=target_sku,
            target_series_id=str(target_series_id),
        )
        baseline_selection_result = select_baseline_from_benchmark_suite(baseline_benchmark_suite)
        best_baseline_strategy = str(baseline_selection_result.get("best_available_strategy", best_baseline_strategy))
        baseline_granularity = str(baseline_selection_result.get("best_available_granularity", baseline_granularity))
        baseline_selector_reason = (
            f"final selected via benchmark_suite ({best_baseline_strategy}/{baseline_granularity}); "
            f"plan suggested ({plan_selected_strategy}/{plan_selected_granularity})"
        )
        if not baseline_benchmark_suite.empty:
            baseline_benchmark_suite["winner_scope"] = "rejected"
            best_mask = (
                (baseline_benchmark_suite["strategy"].astype(str) == best_baseline_strategy)
                & (baseline_benchmark_suite["granularity"].astype(str) == baseline_granularity)
            )
            baseline_benchmark_suite.loc[best_mask, "winner_scope"] = "best_available"
            runner_up_name = str(baseline_selection_result.get("runner_up_strategy", "") or "")
            if runner_up_name:
                ru = (baseline_benchmark_suite["strategy"].astype(str) == runner_up_name) & (~best_mask)
                baseline_benchmark_suite.loc[ru, "winner_scope"] = "runner_up"

        base_ctx = {
            c: (target_history[c].dropna().astype(str).iloc[-1] if c in ["series_id", "product_id", "category", "region", "channel", "segment"] and c in target_history else "unknown")
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
            baseline_rolling = run_baseline_rolling_backtest(panel_train, target_category, target_sku, target_series_id=str(target_series_id), strategy=best_baseline_strategy)
            baseline_oof = build_baseline_oof_predictions(panel_train, target_category, target_sku, target_series_id=str(target_series_id), strategy=best_baseline_strategy)
        else:
            trained_baseline_bt = None
            trained_baseline_final = None
            baseline_forecast = forecast_baseline_by_strategy(best_baseline_strategy, None, target_history, future_dates, base_ctx, baseline_feature_spec_full)
            baseline_rolling = run_baseline_rolling_backtest(panel_train, target_category, target_sku, target_series_id=str(target_series_id), strategy=best_baseline_strategy)
            baseline_oof = build_baseline_oof_predictions(panel_train, target_category, target_sku, target_series_id=str(target_series_id), strategy=best_baseline_strategy)
        baseline_forecast["baseline_lower"] = np.nan
        baseline_forecast["baseline_upper"] = np.nan

    baseline_plan_selection["plan_selected_strategy"] = plan_selected_strategy
    baseline_plan_selection["plan_selected_granularity"] = plan_selected_granularity
    baseline_plan_selection["final_selected_strategy"] = best_baseline_strategy
    baseline_plan_selection["final_selected_granularity"] = baseline_granularity
    baseline_plan_selection["final_selection_source"] = "benchmark_suite_selection"

    bench_match = pd.DataFrame()
    if not baseline_benchmark_suite.empty:
        bench_match = baseline_benchmark_suite[
            (baseline_benchmark_suite["strategy"].astype(str) == str(best_baseline_strategy))
            & (baseline_benchmark_suite["granularity"].astype(str) == str(baseline_granularity))
        ].copy()
        if bench_match.empty:
            bench_match = baseline_benchmark_suite[baseline_benchmark_suite["strategy"].astype(str) == str(best_baseline_strategy)].copy()
    if not bench_match.empty:
        r = bench_match.iloc[0]
        baseline_quality_gate = {
            "baseline_meets_quality_gate": bool(r.get("acceptance_pass", False)),
            "baseline_goal_wape_median_le_25": bool(r.get("goal_wape_median_le_25", False)),
            "baseline_goal_wape_max_le_35": bool(r.get("goal_wape_max_le_35", False)),
            "baseline_goal_abs_bias_le_7pct": bool(r.get("goal_abs_bias_le_7pct", False)),
            "baseline_goal_sum_ratio_in_range": bool(r.get("goal_sum_ratio_in_range", False)),
            "baseline_goal_std_ratio_ge_055": bool(r.get("goal_std_ratio_ge_055", False)),
            "baseline_rejection_reason": "" if bool(r.get("acceptance_pass", False)) else str(baseline_selection_result.get("baseline_rejection_reason", "quality_gate_failed")),
        }
    else:
        baseline_quality_gate = {
            "baseline_meets_quality_gate": False,
            "baseline_goal_wape_median_le_25": False,
            "baseline_goal_wape_max_le_35": False,
            "baseline_goal_abs_bias_le_7pct": False,
            "baseline_goal_sum_ratio_in_range": False,
            "baseline_goal_std_ratio_ge_055": False,
            "baseline_rejection_reason": str(baseline_selection_result.get("baseline_rejection_reason", "no_benchmark_match")),
        }

    factor_train_target = target_train.merge(baseline_oof[["date", "baseline_oof"]], on="date", how="left")
    factor_feature_spec_target = derive_factor_feature_spec(factor_train_target)
    factor_train_target = build_factor_feature_matrix(factor_train_target, factor_feature_spec_target)
    factor_train_target = factor_train_target[factor_train_target["baseline_oof"].notna()].copy()
    target_frame = build_factor_target_frame(factor_train_target)
    factor_train_target = pd.concat([factor_train_target, target_frame], axis=1)
    factor_train_target = factor_train_target[factor_train_target["factor_target_valid"].astype(bool)].copy()
    target_assess = _assess_factor_trainability(factor_train_target, factor_feature_spec_target)

    use_target_direct = (
        target_assess["n_rows"] >= 120
        and target_assess["price_unique_count"] >= 6
        and target_assess["variative_controllable_count"] >= 2
    )

    factor_train_scope = "none"
    factor_train = pd.DataFrame()
    factor_feature_spec = factor_feature_spec_target
    pooled_series_used: List[str] = []
    pooled_rows_used = 0

    if target_assess["trainable"] and use_target_direct:
        factor_train_scope = "target"
        factor_train = factor_train_target
    else:
        pooled_info = _build_pooled_factor_train(panel_train, target_category)
        pooled_candidate = pooled_info.get("factor_train", pd.DataFrame())
        pooled_series_used = pooled_info.get("pooled_series_used", [])
        pooled_rows_used = int(pooled_info.get("pooled_rows_used", 0))
        if not pooled_candidate.empty:
            factor_feature_spec = derive_factor_feature_spec(pooled_candidate)
            pooled_candidate = build_factor_feature_matrix(pooled_candidate, factor_feature_spec)
            pooled_candidate = pooled_candidate[pooled_candidate["baseline_oof"].notna()].copy()
            pooled_frame = build_factor_target_frame(pooled_candidate)
            pooled_candidate = pd.concat([pooled_candidate, pooled_frame], axis=1)
            pooled_candidate = pooled_candidate[pooled_candidate["factor_target_valid"].astype(bool)].copy()
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
            "pooled_series_used": pooled_series_used,
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
        factor_backtest_summary=factor_backtest,
    )
    neutral_baseline_forecast = scenario_result["neutral_baseline_forecast"].copy()
    as_is_forecast = scenario_result["as_is_forecast"].copy()
    scenario_forecast = scenario_result["scenario_forecast"].copy()

    neutral_ctx = scenario_result["neutral_ctx"]
    current_ctx = scenario_result["current_ctx"]
    scn_ctx = scenario_result["scenario_ctx"]

    n_input = neutral_baseline_forecast.copy()
    n_input["price"] = max(float(neutral_ctx.get("price", target_history.get("price", pd.Series([1.0])).iloc[-1])), 1e-6)
    n_input["discount"] = float(neutral_ctx.get("discount", target_history.get("discount", pd.Series([0.0])).iloc[-1]))
    n_input["cost"] = float(neutral_ctx.get("cost", target_history.get("cost", pd.Series([0.0])).iloc[-1]))
    n_input["freight_value"] = float(neutral_ctx.get("freight_value", target_history.get("freight_value", pd.Series([0.0])).iloc[-1]))
    neutral_baseline_economics, _ = compute_daily_unit_economics(
        n_input,
        quantity_col="baseline_pred",
        unit_price_input_type=unit_price_input_type,
        economics_mode=economics_mode,
    )

    a_input = as_is_forecast.copy()
    a_input["price"] = max(float(current_ctx.get("price", n_input["price"].iloc[0])), 1e-6)
    a_input["discount"] = float(current_ctx.get("discount", n_input["discount"].iloc[0]))
    a_input["cost"] = float(current_ctx.get("cost", n_input["cost"].iloc[0]))
    a_input["freight_value"] = float(current_ctx.get("freight_value", n_input["freight_value"].iloc[0]))
    as_is_economics, _ = compute_daily_unit_economics(
        a_input,
        quantity_col="actual_sales",
        unit_price_input_type=unit_price_input_type,
        economics_mode=economics_mode,
    )

    s_input = scenario_forecast.copy()
    s_input["price"] = max(float(scn_ctx.get("price", a_input["price"].iloc[0])), 1e-6)
    s_input["discount"] = float(scn_ctx.get("discount", a_input["discount"].iloc[0]))
    s_input["cost"] = float(scn_ctx.get("cost", a_input["cost"].iloc[0]))
    s_input["freight_value"] = float(scn_ctx.get("freight_value", a_input["freight_value"].iloc[0]))
    scenario_economics, _ = compute_daily_unit_economics(
        s_input,
        quantity_col="actual_sales",
        unit_price_input_type=unit_price_input_type,
        economics_mode=economics_mode,
    )

    as_is_total_demand = float(pd.to_numeric(as_is_forecast["actual_sales"], errors="coerce").fillna(0.0).sum())
    scenario_total_demand = float(pd.to_numeric(scenario_forecast["actual_sales"], errors="coerce").fillna(0.0).sum())
    demand_delta_abs = scenario_total_demand - as_is_total_demand
    demand_delta_pct = demand_delta_abs / max(as_is_total_demand, 1e-9)
    baseline_total_revenue = float(as_is_economics["total_revenue"].sum())
    scenario_total_revenue = float(scenario_economics["total_revenue"].sum())
    baseline_total_profit = float(as_is_economics["profit"].sum())
    scenario_total_profit = float(scenario_economics["profit"].sum())
    delta_summary_current_vs_scenario = pd.DataFrame(
        [
            {
                "as_is_total_demand": as_is_total_demand,
                "scenario_total_demand": scenario_total_demand,
                "demand_delta_abs": demand_delta_abs,
                "demand_delta_pct": demand_delta_pct,
                "as_is_total_revenue": baseline_total_revenue,
                "scenario_total_revenue": scenario_total_revenue,
                "revenue_delta_abs": scenario_total_revenue - baseline_total_revenue,
                "revenue_delta_pct": 0.0 if abs(baseline_total_revenue) < 1e-9 else (scenario_total_revenue - baseline_total_revenue) / baseline_total_revenue,
                "as_is_total_profit": baseline_total_profit,
                "scenario_total_profit": scenario_total_profit,
                "profit_delta_abs": scenario_total_profit - baseline_total_profit,
                "profit_delta_pct": 0.0 if abs(baseline_total_profit) < 1e-9 else (scenario_total_profit - baseline_total_profit) / baseline_total_profit,
            }
        ]
    )
    neutral_total_demand = float(pd.to_numeric(neutral_baseline_forecast["baseline_pred"], errors="coerce").fillna(0.0).sum())
    neutral_total_revenue = float(neutral_baseline_economics["total_revenue"].sum())
    neutral_total_profit = float(neutral_baseline_economics["profit"].sum())
    delta_summary_neutral_vs_current = pd.DataFrame(
        [
            {
                "neutral_total_demand": neutral_total_demand,
                "as_is_total_demand": as_is_total_demand,
                "demand_delta_abs": as_is_total_demand - neutral_total_demand,
                "demand_delta_pct": 0.0 if abs(neutral_total_demand) < 1e-9 else (as_is_total_demand - neutral_total_demand) / neutral_total_demand,
                "neutral_total_revenue": neutral_total_revenue,
                "as_is_total_revenue": baseline_total_revenue,
                "revenue_delta_abs": baseline_total_revenue - neutral_total_revenue,
                "revenue_delta_pct": 0.0 if abs(neutral_total_revenue) < 1e-9 else (baseline_total_revenue - neutral_total_revenue) / neutral_total_revenue,
                "neutral_total_profit": neutral_total_profit,
                "as_is_total_profit": baseline_total_profit,
                "profit_delta_abs": baseline_total_profit - neutral_total_profit,
                "profit_delta_pct": 0.0 if abs(neutral_total_profit) < 1e-9 else (baseline_total_profit - neutral_total_profit) / neutral_total_profit,
            }
        ]
    )

    current_state_contributions = compute_current_state_contributions(target_history, future_dates, neutral_ctx, current_ctx, trained_factor, factor_feature_spec if factor_model_trained else {"controllable_features": []}) if factor_model_trained else pd.DataFrame(columns=["factor_name", "from_value", "to_value", "multiplier_delta", "contribution_pct", "confidence", "note"])
    scenario_delta_contributions = compute_scenario_delta_contributions(target_history, future_dates, current_ctx, scn_ctx, trained_factor, factor_feature_spec if factor_model_trained else {"controllable_features": []}) if factor_model_trained else pd.DataFrame(columns=["factor_name", "from_value", "to_value", "multiplier_delta", "contribution_pct", "confidence", "note"])

    baseline_conf = build_baseline_confidence_state(baseline_rolling.get("rolling_summary", {}))
    factor_role = factor_train_stats.get("factor_role", "unavailable")
    if factor_train_scope == "pooled":
        factor_role = "advisory_only"
    if not bool(baseline_quality_gate.get("baseline_meets_quality_gate", False)):
        factor_role = "advisory_only"
    if baseline_conf.get("level") != "high" and factor_role == "production":
        factor_role = "advisory_only"
    factor_conf = build_factor_confidence_state(
        factor_backtest,
        ood_flags=scenario_result.get("ood_flags", []),
        baseline_level=baseline_conf.get("level", "low"),
        scenario_outside_factor_backtest_range=("scenario_outside_factor_backtest_range" in scenario_result.get("warnings", [])),
    )
    shock_conf = build_shock_confidence_state(shocks)
    has_current_effect = any(neutral_ctx.get(c) != current_ctx.get(c) for c in (factor_feature_spec or {}).get("controllable_features", []))
    has_override_effect = any(current_ctx.get(c) != scn_ctx.get(c) for c in (factor_feature_spec or {}).get("controllable_features", []))
    explainability_available = True
    if has_current_effect and current_state_contributions.empty:
        explainability_available = False
    if has_override_effect and scenario_delta_contributions.empty:
        explainability_available = False

    confidence = combine_confidence_states(
        baseline_conf,
        factor_conf,
        shock_conf,
        intervals_available=False,
        factor_role=factor_role,
        scenario_outside_factor_backtest_range=("scenario_outside_factor_backtest_range" in scenario_result.get("warnings", [])),
        scenario_equals_current_but_delta_nonzero=("scenario_equals_current_but_delta_nonzero" in scenario_result.get("warnings", [])),
        explainability_available=explainability_available,
    )
    confidence.update(baseline_quality_gate)

    confidence_flat = pd.DataFrame(
        [
            {
                "overall_confidence": confidence.get("overall_confidence"),
                "issues": "; ".join([str(x) for x in confidence.get("issues", [])]),
                "intervals_available": bool(confidence.get("intervals_available", False)),
                "baseline_confidence_level": confidence.get("baseline_confidence", {}).get("level"),
                "factor_confidence_level": confidence.get("factor_confidence", {}).get("level"),
                "shock_confidence_level": confidence.get("shock_confidence", {}).get("level"),
                "factor_role": factor_role,
                **baseline_quality_gate,
            }
        ]
    )

    baseline_quality_summary = build_baseline_quality_summary(
        baseline_benchmark_suite,
        {
            "best_available_strategy": best_baseline_strategy,
            "best_available_granularity": baseline_granularity,
            "baseline_meets_quality_gate": baseline_quality_gate.get("baseline_meets_quality_gate", False),
            "runner_up_strategy": baseline_selection_result.get("runner_up_strategy", ""),
            "runner_up_score": baseline_selection_result.get("runner_up_score", np.nan),
        },
        baseline_selector_reason,
    )

    runner_up_note = "n/a"
    if not baseline_benchmark_suite.empty:
        bench_rank = baseline_benchmark_suite.copy()
        if "composite_score" not in bench_rank.columns:
            bench_rank["composite_score"] = np.nan
        if "median_wape" not in bench_rank.columns:
            bench_rank["median_wape"] = np.nan
        top2 = bench_rank.sort_values(["composite_score", "median_wape"]).head(2).copy()
        if len(top2) >= 2:
            r = top2.iloc[1]
            runner_up_note = f"{r.get('strategy')} ({r.get('granularity')}) score={float(r.get('composite_score', np.nan)):.2f}"
    diagnostic_summary = pd.DataFrame(
        [
            {"item": "selected_baseline", "value": f"{best_baseline_strategy} ({baseline_granularity})"},
            {"item": "selector_reason", "value": baseline_selector_reason},
            {"item": "plan_selected_baseline", "value": f"{plan_selected_strategy} ({plan_selected_granularity})"},
            {"item": "final_baseline_source", "value": "benchmark_suite_selection"},
            {"item": "runner_up", "value": runner_up_note},
            {"item": "overall_confidence", "value": str(confidence.get("overall_confidence", "low"))},
            {"item": "confidence_issues", "value": "; ".join([str(x) for x in confidence.get("issues", [])])},
            {"item": "scenario_equals_as_is_demand", "value": str(abs(demand_delta_pct) <= 1e-9)},
            {"item": "scenario_demand_delta_pct", "value": f"{float(demand_delta_pct) * 100.0:.2f}%"},
            {"item": "baseline_meets_quality_gate", "value": str(bool(baseline_quality_gate.get("baseline_meets_quality_gate", False)))},
            {"item": "baseline_rejection_reason", "value": str(baseline_quality_gate.get("baseline_rejection_reason", ""))},
        ]
    )
    scenario_inputs_echo = pd.DataFrame(
        [
            {
                "control_name": c,
                "current_value": current_ctx.get(c),
                "scenario_value": scn_ctx.get(c),
                "changed_flag": current_ctx.get(c) != scn_ctx.get(c),
                "supported_flag": True,
            }
            for c in (factor_feature_spec or {}).get("controllable_features", [])
        ]
    )
    if scenario_overrides:
        known_controls = set((factor_feature_spec or {}).get("controllable_features", []))
        for k, v in scenario_overrides.items():
            if k not in known_controls:
                scenario_inputs_echo = pd.concat(
                    [
                        scenario_inputs_echo,
                        pd.DataFrame([{"control_name": k, "current_value": current_ctx.get(k), "scenario_value": v, "changed_flag": False, "supported_flag": False}]),
                    ],
                    ignore_index=True,
                )

    scenario_changed_any = bool(scenario_inputs_echo["changed_flag"].any()) if not scenario_inputs_echo.empty else False
    unsupported_override_count = int((scenario_inputs_echo["supported_flag"] == False).sum()) if not scenario_inputs_echo.empty else 0  # noqa: E712
    scenario_delta_zero_reason = ""
    if abs(float(demand_delta_pct)) <= 1e-9:
        if not scenario_overrides:
            scenario_delta_zero_reason = "no_overrides"
        elif unsupported_override_count > 0 and not scenario_changed_any:
            scenario_delta_zero_reason = "unsupported_overrides"
        elif not scenario_changed_any:
            scenario_delta_zero_reason = "scenario_equals_current_state"
        else:
            scenario_delta_zero_reason = "no_material_effect"

    diagnostic_summary = pd.concat(
        [
            diagnostic_summary,
            pd.DataFrame(
                [
                    {"item": "scenario_controls_changed", "value": str(scenario_changed_any)},
                    {"item": "scenario_unsupported_overrides", "value": str(unsupported_override_count)},
                    {"item": "scenario_delta_zero_reason", "value": scenario_delta_zero_reason},
                ]
            ),
        ],
        ignore_index=True,
    )

    rolling_metrics_export = baseline_rolling.get("rolling_metrics", pd.DataFrame()).copy()
    rolling_diag_export = baseline_rolling.get("rolling_diag", pd.DataFrame()).copy()
    required_roll_cols = [
        "window_id",
        "window_start",
        "window_end",
        "forecast_wape",
        "mae",
        "rmse",
        "bias_pct",
        "sum_ratio",
        "pred_std",
        "actual_std",
        "std_ratio",
        "pred_nunique",
        "actual_nunique",
        "is_flat_forecast",
        "weekday_shape_error",
    ]
    for c in required_roll_cols:
        if c not in rolling_metrics_export.columns:
            rolling_metrics_export[c] = np.nan
    required_diag_cols = ["date", "sales", "baseline_pred", "window_id", "window_start", "window_end"]
    for c in required_diag_cols:
        if c not in rolling_diag_export.columns:
            rolling_diag_export[c] = np.nan

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        target_history.to_excel(writer, sheet_name="history", index=False)
        neutral_baseline_forecast.to_excel(writer, sheet_name="neutral_baseline_forecast", index=False)
        as_is_forecast.to_excel(writer, sheet_name="as_is_forecast", index=False)
        scenario_forecast.to_excel(writer, sheet_name="scenario_forecast", index=False)
        neutral_baseline_economics.to_excel(writer, sheet_name="neutral_baseline_economics", index=False)
        as_is_economics.to_excel(writer, sheet_name="as_is_economics", index=False)
        scenario_economics.to_excel(writer, sheet_name="scenario_economics", index=False)
        delta_summary_current_vs_scenario.to_excel(writer, sheet_name="delta_summary_current_vs_scenario", index=False)
        delta_summary_neutral_vs_current.to_excel(writer, sheet_name="delta_summary_neutral_vs_current", index=False)
        rolling_metrics_export[required_roll_cols].to_excel(writer, sheet_name="baseline_rolling_metrics", index=False)
        rolling_diag_export[required_diag_cols].to_excel(writer, sheet_name="baseline_rolling_diag", index=False)
        (pd.DataFrame([factor_backtest]) if isinstance(factor_backtest, dict) else pd.DataFrame(factor_backtest)).to_excel(writer, sheet_name="factor_backtest", index=False)
        current_state_contributions.to_excel(writer, sheet_name="current_state_contributions", index=False)
        scenario_delta_contributions.to_excel(writer, sheet_name="scenario_delta_contributions", index=False)
        baseline_quality_summary.to_excel(writer, sheet_name="baseline_quality_summary", index=False)
        baseline_data_quality.to_excel(writer, sheet_name="baseline_data_quality", index=False)
        scenario_inputs_echo.to_excel(writer, sheet_name="scenario_inputs_echo", index=False)
        diagnostic_summary.to_excel(writer, sheet_name="diagnostic_summary", index=False)
        baseline_benchmark_suite.to_excel(writer, sheet_name="baseline_benchmark_suite", index=False)
        pd.DataFrame([confidence]).to_excel(writer, sheet_name="confidence", index=False)
        confidence_flat.to_excel(writer, sheet_name="confidence_flat", index=False)
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
        baseline_benchmark_suite=baseline_benchmark_suite,
        baseline_quality_gate=baseline_quality_gate,
        baseline_runner_up_strategy=str(baseline_selection_result.get("runner_up_strategy", "")),
        baseline_runner_up_score=float(baseline_selection_result.get("runner_up_score", np.nan)),
        final_baseline_strategy=best_baseline_strategy,
        final_baseline_granularity=baseline_granularity,
        final_baseline_source="benchmark_suite_selection",
        scenario_controls_changed=scenario_changed_any,
        scenario_delta_zero_reason=scenario_delta_zero_reason,
        mode=mode,
        ood_flags=ood_flags,
        intervals_available=intervals_available,
        pooled_series_used=pooled_series_used,
        pooled_rows_used=pooled_rows_used,
        neutral_ctx=neutral_ctx,
        current_ctx=current_ctx,
        base_ctx=current_ctx,
        scenario_mode=scenario_result.get("scenario_mode", mode),
        scenario_effect_source=scenario_result.get("scenario_effect_source", mode),
        target_series_id=str(target_series_id),
        series_scope={c: str(target_history[c].iloc[-1]) if c in target_history.columns else "unknown" for c in SERIES_SCOPE_COLS},
        unit_price_input_type=str(unit_price_input_type),
        economics_mode=str(economics_mode),
        neutral_baseline_forecast=neutral_baseline_forecast.copy(),
        as_is_forecast=as_is_forecast.copy(),
        factor_role=factor_role,
        decision_validity=(
            "production_candidate"
            if (confidence.get("overall_confidence") == "high" and bool(baseline_quality_gate.get("baseline_meets_quality_gate", False)))
            else "advisory_only"
        ),
        baseline_forecast=neutral_baseline_forecast.copy(),
        scenario_feature_spec=scenario_feature_spec,
        feature_spec=scenario_feature_spec,
        current_price=float(current_ctx.get("price", target_history.get("price", pd.Series([1.0])).iloc[-1])),
        forecast_horizon_days=int(horizon_days),
        objective_mode=str(objective_mode),
    )

    return {
        "history": target_history,
        "neutral_baseline_forecast": neutral_baseline_forecast,
        "as_is_forecast": as_is_forecast,
        "baseline_forecast": neutral_baseline_forecast,
        "scenario_forecast": scenario_forecast,
        "neutral_baseline_economics": neutral_baseline_economics,
        "as_is_economics": as_is_economics,
        "baseline_economics": neutral_baseline_economics,
        "scenario_economics": scenario_economics,
        "delta_summary_current_vs_scenario": delta_summary_current_vs_scenario,
        "delta_summary_neutral_vs_current": delta_summary_neutral_vs_current,
        "delta_summary": delta_summary_current_vs_scenario,
        "current_state_contributions": current_state_contributions,
        "scenario_delta_contributions": scenario_delta_contributions,
        "factor_contributions": scenario_delta_contributions,
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
        "baseline_benchmark_suite": baseline_benchmark_suite,
        "baseline_quality_gate": baseline_quality_gate,
        "baseline_runner_up_strategy": str(baseline_selection_result.get("runner_up_strategy", "")),
        "baseline_runner_up_score": float(baseline_selection_result.get("runner_up_score", np.nan)),
        "final_baseline_strategy": best_baseline_strategy,
        "final_baseline_granularity": baseline_granularity,
        "final_baseline_source": "benchmark_suite_selection",
        "baseline_data_quality": baseline_data_quality,
        "scenario_inputs_echo": scenario_inputs_echo,
        "scenario_controls_changed": scenario_changed_any,
        "scenario_delta_zero_reason": scenario_delta_zero_reason,
        "mode": mode,
        "scenario_mode": scenario_result.get("scenario_mode", mode),
        "scenario_effect_source": scenario_result.get("scenario_effect_source", mode),
        "factor_role": factor_role,
        "ood_flags": ood_flags,
        "intervals_available": intervals_available,
        "target_series_id": str(target_series_id),
        "unit_price_input_type": str(unit_price_input_type),
        "economics_mode": str(economics_mode),
        "_trained_bundle": bundle,
    }
