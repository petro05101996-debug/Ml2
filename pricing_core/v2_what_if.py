from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from calc_engine import compute_daily_unit_economics
from pricing_core.scenario_engine import run_scenario_forecast


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

    baseline_override = trained_bundle.get("neutral_baseline_forecast", trained_bundle.get("baseline_forecast"))
    scenario_result = run_scenario_forecast(
        trained_baseline=trained_bundle.get("trained_baseline_final"),
        trained_factor=trained_bundle.get("trained_factor"),
        base_history=target_history,
        future_dates_df=future_dates,
        baseline_feature_spec=trained_bundle.get("baseline_feature_spec_full", trained_bundle.get("baseline_feature_spec_final", {})),
        factor_feature_spec=trained_bundle.get("factor_feature_spec"),
        scenario_overrides=overrides,
        shocks=(scenario or {}).get("shocks"),
        baseline_override_df=baseline_override[["date", "baseline_pred"]] if isinstance(baseline_override, pd.DataFrame) else None,
        demand_multiplier=float(demand_multiplier),
    )
    as_is_forecast = scenario_result["as_is_forecast"].copy()
    sf = scenario_result["scenario_forecast"].copy()
    sf["price"] = max(float(overrides.get("price", manual_price)), 1e-6)
    sf["discount"] = float(overrides.get("discount", 0.0))
    sf["cost"] = float(overrides.get("cost", 0.0))
    sf["freight_value"] = float(overrides.get("freight_value", 0.0))
    as_is_input = as_is_forecast.copy()
    as_is_input["price"] = max(float(current_ctx.get("price", manual_price)), 1e-6)
    as_is_input["discount"] = float(current_ctx.get("discount", 0.0))
    as_is_input["cost"] = float(current_ctx.get("cost", 0.0))
    as_is_input["freight_value"] = float(current_ctx.get("freight_value", 0.0))
    as_is_eco, _ = compute_daily_unit_economics(
        as_is_input,
        quantity_col="actual_sales",
        unit_price_input_type=str(trained_bundle.get("unit_price_input_type", "net")),
        economics_mode=str(trained_bundle.get("economics_mode", "net_price")),
    )
    scenario_eco, _ = compute_daily_unit_economics(
        sf,
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
        "ood_flags": scenario_result.get("ood_flags", []),
        "scenario_mode": scenario_result.get("scenario_mode", scenario_result.get("mode", "fallback_elasticity")),
        "factor_role": str(trained_bundle.get("factor_role", "unavailable")),
        "scenario_effect_source": scenario_result.get("scenario_effect_source", "fallback_elasticity"),
        "economics_mode": str(trained_bundle.get("economics_mode", "net_price")),
        "unit_price_input_type": str(trained_bundle.get("unit_price_input_type", "net")),
    }
