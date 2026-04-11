from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from calc_engine import compute_daily_unit_economics
from pricing_core.baseline_model import disaggregate_weekly_to_daily
from pricing_core.scenario_engine import run_weekly_scenario_forecast


def _confidence_map(label: str) -> float:
    return {"high": 0.8, "medium": 0.6, "low": 0.35}.get(str(label).lower(), 0.35)


def run_v2_what_if_projection(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: int = 30,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float | None = None,
    scenario: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    target_history = trained_bundle.get("target_history", pd.DataFrame())
    future_dates = trained_bundle.get("future_dates", pd.DataFrame())
    if horizon_days > 0 and len(future_dates):
        future_dates = future_dates.head(int(horizon_days)).copy()
    factors = (scenario or {}).get("factors", {})
    current_ctx = dict(trained_bundle.get("current_ctx", {})) or ({**target_history.tail(1).to_dict("records")[0]} if len(target_history) else {})
    overrides = {
        "price": float(factors.get("price", manual_price)),
        "discount": float(factors.get("discount", current_ctx.get("discount", 0.0))) * float(discount_multiplier),
        "promotion": float(factors.get("promotion", current_ctx.get("promotion", 0.0))),
        "cost": float(current_ctx.get("cost", 0.0)) * float(cost_multiplier),
        "freight_value": float(current_ctx.get("freight_value", 0.0)) * float(freight_multiplier),
    }
    if stock_cap is None:
        overrides["use_stock_cap"] = False
    else:
        overrides["use_stock_cap"] = True
        overrides["stock_total_horizon"] = float(stock_cap)

    for k, v in factors.items():
        if str(k).startswith("user_factor_num__") or str(k).startswith("user_factor_cat__"):
            overrides[k] = v

    weekly_baseline = trained_bundle.get("weekly_baseline_forecast_native", trained_bundle.get("weekly_baseline_forecast", pd.DataFrame())).copy()
    if not weekly_baseline.empty and "sales" in weekly_baseline.columns:
        weekly_baseline = weekly_baseline.rename(columns={"sales": "baseline_pred_weekly"})
    if weekly_baseline.empty:
        # fallback for legacy bundles
        baseline_override = trained_bundle.get("neutral_baseline_forecast", trained_bundle.get("baseline_forecast", pd.DataFrame()))
        if isinstance(baseline_override, pd.DataFrame) and not baseline_override.empty:
            from pricing_core.baseline_model import aggregate_daily_to_weekly

            weekly_baseline = aggregate_daily_to_weekly(baseline_override.rename(columns={"baseline_pred": "sales"})).rename(columns={"sales": "baseline_pred_weekly"})

    weekly_result = run_weekly_scenario_forecast(
        weekly_baseline_forecast=weekly_baseline[["week_start", "baseline_pred_weekly"]],
        trained_weekly_factor=trained_bundle.get("trained_weekly_factor"),
        current_ctx=current_ctx,
        scenario_ctx=overrides,
        shocks=(scenario or {}).get("shocks"),
        confidence_level=str((trained_bundle.get("confidence", {}) or {}).get("overall_confidence", "medium")),
    )
    weekday_profile = trained_bundle.get("weekday_profile")
    if weekday_profile is None:
        weekday_profile = pd.Series([1.0 / 7.0] * 7, index=range(7), dtype=float)
    as_is_forecast = disaggregate_weekly_to_daily(
        weekly_result["weekly_as_is_forecast"][["week_start", "actual_as_is"]].rename(columns={"actual_as_is": "baseline_pred_weekly"}),
        future_dates[["date"]],
        weekday_profile,
    ).rename(columns={"baseline_pred": "actual_sales"})
    as_is_lost = disaggregate_weekly_to_daily(
        weekly_result["weekly_as_is_forecast"][["week_start", "lost_as_is"]].rename(columns={"lost_as_is": "baseline_pred_weekly"}),
        future_dates[["date"]],
        weekday_profile,
    ).rename(columns={"baseline_pred": "lost_sales"})
    sf = disaggregate_weekly_to_daily(
        weekly_result["weekly_scenario_forecast"][["week_start", "actual_scenario"]].rename(columns={"actual_scenario": "baseline_pred_weekly"}),
        future_dates[["date"]],
        weekday_profile,
    ).rename(columns={"baseline_pred": "actual_sales"})
    sf_lost = disaggregate_weekly_to_daily(
        weekly_result["weekly_scenario_forecast"][["week_start", "lost_scenario"]].rename(columns={"lost_scenario": "baseline_pred_weekly"}),
        future_dates[["date"]],
        weekday_profile,
    ).rename(columns={"baseline_pred": "lost_sales"})
    as_is_forecast = as_is_forecast.merge(as_is_lost, on="date", how="left")
    sf = sf.merge(sf_lost, on="date", how="left")
    sf["scenario_demand_raw"] = pd.to_numeric(sf["actual_sales"], errors="coerce").fillna(0.0) + pd.to_numeric(sf["lost_sales"], errors="coerce").fillna(0.0)
    as_is_forecast["scenario_demand_raw"] = pd.to_numeric(as_is_forecast["actual_sales"], errors="coerce").fillna(0.0) + pd.to_numeric(as_is_forecast["lost_sales"], errors="coerce").fillna(0.0)
    week_map = future_dates[["date"]].copy()
    week_map["week_start"] = pd.to_datetime(week_map["date"], errors="coerce").dt.to_period("W-SUN").dt.start_time
    as_is_mult = week_map.merge(
        weekly_result["weekly_as_is_forecast"][["week_start", "factor_effect_as_is", "shock_effect"]],
        on="week_start",
        how="left",
    )[["date", "factor_effect_as_is", "shock_effect"]]
    scn_mult = week_map.merge(
        weekly_result["weekly_scenario_forecast"][["week_start", "factor_effect_scenario", "shock_effect"]],
        on="week_start",
        how="left",
    )[["date", "factor_effect_scenario", "shock_effect"]]

    sf["price"] = max(float(overrides.get("price", manual_price)), 1e-6)
    sf["discount"] = float(overrides.get("discount", 0.0))
    sf["cost"] = float(overrides.get("cost", 0.0))
    sf["freight_value"] = float(overrides.get("freight_value", 0.0))
    as_is_input = as_is_forecast.copy()
    as_is_input["scenario_demand_raw"] = as_is_input["scenario_demand_raw"]
    as_is_input["lost_sales"] = pd.to_numeric(as_is_input["lost_sales"], errors="coerce").fillna(0.0)
    as_is_input["price"] = max(float(current_ctx.get("price", manual_price)), 1e-6)
    as_is_input["discount"] = float(current_ctx.get("discount", 0.0))
    as_is_input["cost"] = float(current_ctx.get("cost", 0.0))
    as_is_input["freight_value"] = float(current_ctx.get("freight_value", 0.0))
    daily_final_as_is = as_is_input[["date", "actual_sales", "lost_sales"]].copy()
    daily_final_as_is["baseline_pred"] = disaggregate_weekly_to_daily(
        weekly_result["weekly_neutral_baseline_forecast"][["week_start", "baseline_pred_weekly"]],
        future_dates[["date"]],
        weekday_profile,
    )["baseline_pred"].values
    daily_final_as_is = daily_final_as_is.merge(as_is_mult, on="date", how="left")
    daily_final_as_is["factor_effect"] = pd.to_numeric(daily_final_as_is["factor_effect_as_is"], errors="coerce").fillna(1.0)
    daily_final_as_is["shock_effect"] = pd.to_numeric(daily_final_as_is["shock_effect"], errors="coerce").fillna(1.0)
    daily_final_as_is = daily_final_as_is.drop(columns=["factor_effect_as_is"])
    daily_final_as_is["price"] = as_is_input["price"]
    daily_final_as_is["discount"] = as_is_input["discount"]
    daily_final_as_is["cost"] = as_is_input["cost"]
    daily_final_as_is["freight_value"] = as_is_input["freight_value"]
    as_is_eco, _ = compute_daily_unit_economics(
        daily_final_as_is,
        quantity_col="actual_sales",
        unit_price_input_type=str(trained_bundle.get("unit_price_input_type", "net")),
        economics_mode=str(trained_bundle.get("economics_mode", "net_price")),
    )
    daily_final_scenario = sf[["date", "actual_sales", "lost_sales"]].copy()
    daily_final_scenario["baseline_pred"] = daily_final_as_is["baseline_pred"].values
    daily_final_scenario = daily_final_scenario.merge(scn_mult, on="date", how="left")
    daily_final_scenario["factor_effect"] = pd.to_numeric(daily_final_scenario["factor_effect_scenario"], errors="coerce").fillna(1.0)
    daily_final_scenario["shock_effect"] = pd.to_numeric(daily_final_scenario["shock_effect"], errors="coerce").fillna(1.0)
    daily_final_scenario = daily_final_scenario.drop(columns=["factor_effect_scenario"])
    daily_final_scenario["price"] = sf["price"]
    daily_final_scenario["discount"] = sf["discount"]
    daily_final_scenario["cost"] = sf["cost"]
    daily_final_scenario["freight_value"] = sf["freight_value"]
    scenario_eco, _ = compute_daily_unit_economics(
        daily_final_scenario,
        quantity_col="actual_sales",
        unit_price_input_type=str(trained_bundle.get("unit_price_input_type", "net")),
        economics_mode=str(trained_bundle.get("economics_mode", "net_price")),
    )

    conf_state = trained_bundle.get("confidence", {}) or {}
    conf_label = str(conf_state.get("overall_confidence", "low"))
    conf_score = _confidence_map(conf_label)
    as_is_demand_total = float(as_is_forecast["actual_sales"].sum())
    scenario_demand_total = float(sf["actual_sales"].sum())
    as_is_revenue_total = float(as_is_eco["total_revenue"].sum())
    scenario_revenue_total = float(scenario_eco["total_revenue"].sum())
    as_is_profit_total = float(as_is_eco["profit"].sum())
    scenario_profit_total = float(scenario_eco["profit"].sum())
    return {
        "demand_total": float(sf["scenario_demand_raw"].sum()),  # backward-compat
        "actual_sales_total": scenario_demand_total,  # backward-compat
        "lost_sales_total": float(sf["lost_sales"].sum()),
        "revenue_total": scenario_revenue_total,  # backward-compat
        "profit_total": scenario_profit_total,  # backward-compat
        "margin": float(scenario_eco["profit"].sum() / scenario_eco["total_revenue"].sum()) if float(scenario_eco["total_revenue"].sum()) > 0 else 0.0,
        "as_is_demand_total": as_is_demand_total,
        "scenario_demand_total": scenario_demand_total,
        "demand_delta_abs": scenario_demand_total - as_is_demand_total,
        "demand_delta_pct": 0.0 if abs(as_is_demand_total) < 1e-9 else (scenario_demand_total - as_is_demand_total) / as_is_demand_total,
        "as_is_revenue_total": as_is_revenue_total,
        "scenario_revenue_total": scenario_revenue_total,
        "revenue_delta_abs": scenario_revenue_total - as_is_revenue_total,
        "as_is_profit_total": as_is_profit_total,
        "scenario_profit_total": scenario_profit_total,
        "profit_delta_abs": scenario_profit_total - as_is_profit_total,
        "confidence_label": conf_label,
        "confidence_score": conf_score,
        "uncertainty_score": float(max(0.0, 1.0 - conf_score)),
        "ood_flags": weekly_result.get("ood_flags", []),
        "scenario_mode": "weekly_native",
        "factor_role": str(trained_bundle.get("factor_role", "unavailable")),
        "scenario_effect_source": weekly_result.get("factor_effect_source", "bounded_rules"),
        "factor_effect_source": weekly_result.get("factor_effect_source", "bounded_rules"),
        "shock_applied": bool(weekly_result.get("shock_applied", False)),
        "baseline_component_total": float(pd.to_numeric(weekly_result["weekly_scenario_forecast"].get("baseline_component", 0.0), errors="coerce").fillna(0.0).sum()) if "weekly_scenario_forecast" in weekly_result else 0.0,
        "factor_effect_avg": float(pd.to_numeric(weekly_result["weekly_factor_effect_scenario"].get("factor_effect_scenario", 1.0), errors="coerce").fillna(1.0).mean()) if "weekly_factor_effect_scenario" in weekly_result else 1.0,
        "shock_effect_avg": float(pd.to_numeric(weekly_result["weekly_scenario_forecast"].get("shock_effect", 1.0), errors="coerce").fillna(1.0).mean()) if "weekly_scenario_forecast" in weekly_result else 1.0,
        "economics_mode": str(trained_bundle.get("economics_mode", "net_price")),
        "unit_price_input_type": str(trained_bundle.get("unit_price_input_type", "net")),
    }
