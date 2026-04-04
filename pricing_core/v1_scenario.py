from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from calc_engine import compute_daily_unit_economics, sanitize_discount, sanitize_non_negative
from .v1_optimizer import simulate_v1_horizon_profit

SENSITIVITY_STEPS = {"price": 0.01, "discount": 0.02, "cost": 0.02, "freight_value": 0.05, "review_score": 0.05}


def _run_local_simulation(trained_bundle: Dict[str, Any], base_history: pd.DataFrame, base_ctx: Dict[str, Any], future_dates: pd.DataFrame, demand_multiplier: float, stock_cap: float) -> Dict[str, Any]:
    sim = simulate_v1_horizon_profit(
        trained_bundle["latest_row"],
        float(base_ctx["price"]),
        future_dates,
        trained_bundle["demand_models"],
        base_history,
        base_ctx,
        trained_bundle["feature_spec"],
        risk_lambda=float(trained_bundle.get("risk_lambda", 0.7)),
    )
    daily = sim["daily"].copy()
    daily["pred_demand"] = daily["pred_demand"] * float(demand_multiplier)
    daily["pred_sales"] = daily["pred_sales"] * float(demand_multiplier)
    daily, econ_checks = compute_daily_unit_economics(daily, quantity_col="pred_sales", stock_cap=float(stock_cap))
    daily["pred_sales"] = daily["pred_quantity"]
    return {"daily": daily, "sim": sim, "econ_warnings": econ_checks.get("sanity_warnings", [])}


def run_v1_what_if_projection(trained_bundle: Dict[str, Any], manual_price: float, freight_multiplier: float = 1.0, demand_multiplier: float = 1.0, horizon_days: Optional[int] = None, discount_multiplier: float = 1.0, cost_multiplier: float = 1.0, stock_cap: float = 0.0, scenario: Optional[Dict[str, Any]] = None, include_sensitivity: bool = True) -> Dict[str, Any]:
    base_history = trained_bundle["daily_base"].copy()
    base_ctx = dict(trained_bundle["base_ctx"])
    scenario_factors = dict((scenario or {}).get("factors", {}))
    if "rating" in scenario_factors and "review_score" not in scenario_factors:
        scenario_factors["review_score"] = scenario_factors["rating"]
    allowed = set(trained_bundle["feature_spec"]["scenario_features"])
    scenario_factors = {k: v for k, v in scenario_factors.items() if k in allowed}

    for k in allowed:
        if k in scenario_factors:
            base_ctx[k] = float(scenario_factors[k])
    base_ctx["price"] = sanitize_non_negative(base_ctx.get("price", manual_price), fallback=float(manual_price))
    base_ctx["discount"] = sanitize_discount(float(base_ctx.get("discount", 0.0)) * float(discount_multiplier))
    base_ctx["cost"] = sanitize_non_negative(float(base_ctx.get("cost", base_ctx["price"] * 0.65)) * float(cost_multiplier))
    base_ctx["freight_value"] = sanitize_non_negative(float(base_ctx.get("freight_value", 0.0)) * float(freight_multiplier))

    n_days = int(horizon_days or len(trained_bundle.get("future_dates", [])) or 30)
    future_dates = pd.DataFrame({"date": pd.date_range(pd.Timestamp(base_history["date"].max()) + pd.Timedelta(days=1), periods=n_days, freq="D")})
    local = _run_local_simulation(trained_bundle, base_history, base_ctx, future_dates, float(demand_multiplier), float(stock_cap))
    daily = local["daily"]
    local_sensitivity: Dict[str, Any] = {}
    if include_sensitivity:
        for factor, step in SENSITIVITY_STEPS.items():
            plus_ctx = dict(base_ctx)
            cur = float(base_ctx.get(factor, 0.0))
            plus_ctx[factor] = cur * (1.0 + step) if abs(cur) > 1e-9 else step
            plus = _run_local_simulation(trained_bundle, base_history, plus_ctx, future_dates, float(demand_multiplier), float(stock_cap))
            local_sensitivity[factor] = {
                "delta_profit": float(plus["daily"]["profit"].sum() - daily["profit"].sum()),
                "delta_revenue": float(plus["daily"]["total_revenue"].sum() - daily["total_revenue"].sum()),
            }

    return {
        "daily": daily,
        "demand_total": float(daily["pred_demand"].sum()),
        "actual_sales_total": float(daily["pred_sales"].sum()),
        "lost_sales_total": 0.0,
        "profit_total": float(daily["profit"].sum()),
        "revenue_total": float(daily["total_revenue"].sum()),
        "confidence": float(trained_bundle.get("confidence", 0.6)),
        "uncertainty": 1.0 - float(trained_bundle.get("confidence", 0.6)),
        "sanity_warnings": local["sim"].get("sanity_warnings", []) + local.get("econ_warnings", []),
        "scenario_meta": {"factors": scenario_factors, "horizon_days": n_days, "stock_cap": float(stock_cap)},
        "local_sensitivity": local_sensitivity,
    }
