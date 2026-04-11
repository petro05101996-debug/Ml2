from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from calc_engine import compute_daily_unit_economics
from pricing_core.baseline_model import disaggregate_weekly_to_daily
from pricing_core.scenario_engine import apply_user_overrides, build_current_state_context
from pricing_core.weekly_forecast_features import build_future_weekly_frame
from pricing_core.weekly_forecast_model import recursive_weekly_forecast


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
    history = trained_bundle.get("target_history", pd.DataFrame())
    future_dates = trained_bundle.get("future_dates", pd.DataFrame()).head(int(horizon_days)).copy()
    model = trained_bundle.get("trained_weekly_forecast_model")
    weekly_history = trained_bundle.get("weekly_history", pd.DataFrame())
    weekday_profile = trained_bundle.get("weekday_profile")
    if weekday_profile is None:
        weekday_profile = pd.Series([1 / 7.0] * 7, index=range(7), dtype=float)

    factors = (scenario or {}).get("factors", {})
    current_ctx = dict(trained_bundle.get("current_ctx", {})) or build_current_state_context(history, {})
    overrides = {
        "price": float(factors.get("price", manual_price)),
        "discount": float(factors.get("discount", current_ctx.get("discount", 0.0))) * float(discount_multiplier),
        "promotion": float(factors.get("promotion", current_ctx.get("promotion", 0.0))),
        "cost": float(current_ctx.get("cost", 0.0)) * float(cost_multiplier),
        "freight_value": float(current_ctx.get("freight_value", 0.0)) * float(freight_multiplier),
    }
    for k, v in factors.items():
        if str(k).startswith("user_factor_"):
            overrides[k] = v
    scenario_ctx = apply_user_overrides(current_ctx, overrides)

    horizon_weeks = max(1, int((len(future_dates) + 6) // 7))
    fut_as_is = build_future_weekly_frame(weekly_history["week_start"].max(), horizon_weeks, current_ctx)
    fut_scn = build_future_weekly_frame(weekly_history["week_start"].max(), horizon_weeks, scenario_ctx)
    wk_as_is = recursive_weekly_forecast(model, weekly_history, fut_as_is)
    wk_scn = recursive_weekly_forecast(model, weekly_history, fut_scn)
    wk_scn["sales_week"] = wk_scn["sales_week"] * float(demand_multiplier)

    as_is = disaggregate_weekly_to_daily(wk_as_is.rename(columns={"sales_week": "baseline_pred_weekly"}), future_dates[["date"]], weekday_profile).rename(columns={"baseline_pred": "actual_sales"})
    scn = disaggregate_weekly_to_daily(wk_scn.rename(columns={"sales_week": "baseline_pred_weekly"}), future_dates[["date"]], weekday_profile).rename(columns={"baseline_pred": "actual_sales"})

    as_is["lost_sales"] = 0.0
    scn["lost_sales"] = 0.0
    if stock_cap is not None:
        total = max(float(stock_cap), 0.0)
        running = 0.0
        for i in range(len(scn)):
            raw = float(scn.at[i, "actual_sales"])
            keep = max(min(raw, total - running), 0.0)
            scn.at[i, "lost_sales"] = raw - keep
            scn.at[i, "actual_sales"] = keep
            running += keep

    for df, ctx in [(as_is, current_ctx), (scn, scenario_ctx)]:
        df["price"] = float(ctx.get("price", manual_price))
        df["discount"] = float(ctx.get("discount", 0.0))
        df["cost"] = float(ctx.get("cost", df["price"].iloc[0] * 0.65))
        df["freight_value"] = float(ctx.get("freight_value", 0.0))

    as_is_eco, _ = compute_daily_unit_economics(as_is, quantity_col="actual_sales", unit_price_input_type=str(trained_bundle.get("unit_price_input_type", "net")), economics_mode=str(trained_bundle.get("economics_mode", "net_price")))
    scn_eco, _ = compute_daily_unit_economics(scn, quantity_col="actual_sales", unit_price_input_type=str(trained_bundle.get("unit_price_input_type", "net")), economics_mode=str(trained_bundle.get("economics_mode", "net_price")))

    conf = trained_bundle.get("confidence", {}) or {}
    label = str(conf.get("overall_confidence", "low"))
    score = _confidence_map(label)
    as_is_d = float(as_is["actual_sales"].sum())
    scn_d = float(scn["actual_sales"].sum())
    as_is_r = float(as_is_eco["total_revenue"].sum())
    scn_r = float(scn_eco["total_revenue"].sum())
    as_is_p = float(as_is_eco["profit"].sum())
    scn_p = float(scn_eco["profit"].sum())

    return {
        "demand_total": scn_d,
        "actual_sales_total": scn_d,
        "lost_sales_total": float(scn["lost_sales"].sum()),
        "revenue_total": scn_r,
        "profit_total": scn_p,
        "margin": 0.0 if scn_r <= 0 else scn_p / scn_r,
        "as_is_demand_total": as_is_d,
        "scenario_demand_total": scn_d,
        "demand_delta_abs": scn_d - as_is_d,
        "demand_delta_pct": 0.0 if abs(as_is_d) < 1e-9 else (scn_d - as_is_d) / as_is_d,
        "as_is_revenue_total": as_is_r,
        "scenario_revenue_total": scn_r,
        "revenue_delta_abs": scn_r - as_is_r,
        "as_is_profit_total": as_is_p,
        "scenario_profit_total": scn_p,
        "profit_delta_abs": scn_p - as_is_p,
        "confidence_label": label,
        "confidence_score": score,
        "uncertainty_score": max(0.0, 1.0 - score),
        "ood_flags": list((conf.get("baseline_confidence", {}) or {}).get("ood_flags", [])),
        "scenario_mode": "weekly_native",
        "factor_role": "removed",
        "scenario_effect_source": "scenario_direct",
        "factor_effect_source": "scenario_direct",
        "shock_applied": False,
        "baseline_component_total": as_is_d,
        "factor_effect_avg": 1.0,
        "shock_effect_avg": 1.0,
        "economics_mode": str(trained_bundle.get("economics_mode", "net_price")),
        "unit_price_input_type": str(trained_bundle.get("unit_price_input_type", "net")),
    }
