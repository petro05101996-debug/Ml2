from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from calc_engine import compute_daily_unit_economics
from pricing_core.scenario_engine import run_scenario_forecast


def run_v2_what_if_projection(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: int = 30,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float = 0.0,
    scenario: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    target_history = trained_bundle.get("target_history", pd.DataFrame())
    future_dates = trained_bundle.get("future_dates", pd.DataFrame())
    if horizon_days > 0 and len(future_dates):
        future_dates = future_dates.head(int(horizon_days)).copy()
    factors = (scenario or {}).get("factors", {})
    base_ctx = {**trained_bundle.get("target_history", pd.DataFrame()).tail(1).to_dict("records")[0]} if len(target_history) else {}
    overrides = {
        "price": float(factors.get("price", manual_price)),
        "discount": float(factors.get("discount", base_ctx.get("discount", 0.0))) * float(discount_multiplier),
        "promotion": float(factors.get("promotion", base_ctx.get("promotion", 0.0))),
        "stock": float(stock_cap) if stock_cap > 0 else float(factors.get("stock", base_ctx.get("stock", 0.0))),
        "review_score": float(factors.get("review_score", base_ctx.get("review_score", 0.0))),
        "reviews_count": float(factors.get("reviews_count", base_ctx.get("reviews_count", 0.0))),
        "cost": float(base_ctx.get("cost", 0.0)) * float(cost_multiplier),
        "freight_value": float(base_ctx.get("freight_value", 0.0)) * float(freight_multiplier),
    }
    for k, v in factors.items():
        if str(k).startswith("user_factor_num__") or str(k).startswith("user_factor_cat__"):
            overrides[k] = v

    scenario_result = run_scenario_forecast(
        trained_baseline=trained_bundle.get("trained_baseline_final"),
        trained_factor=trained_bundle.get("trained_factor"),
        base_history=target_history,
        future_dates_df=future_dates,
        baseline_feature_spec=trained_bundle.get("baseline_feature_spec_full", trained_bundle.get("baseline_feature_spec_final", {})),
        factor_feature_spec=trained_bundle.get("factor_feature_spec"),
        scenario_overrides=overrides,
        shocks=(scenario or {}).get("shocks"),
        baseline_override_df=trained_bundle.get("baseline_forecast")[ ["date", "baseline_pred"] ] if isinstance(trained_bundle.get("baseline_forecast"), pd.DataFrame) else None,
    )
    sf = scenario_result["scenario_forecast"].copy()
    sf["final_demand"] = pd.to_numeric(sf["final_demand"], errors="coerce").fillna(0.0) * float(demand_multiplier)
    sf["price"] = max(float(overrides.get("price", manual_price)), 1e-6)
    sf["discount"] = float(overrides.get("discount", 0.0))
    sf["cost"] = float(overrides.get("cost", 0.0))
    sf["freight_value"] = float(overrides.get("freight_value", 0.0))
    eco, _ = compute_daily_unit_economics(sf, quantity_col="final_demand")

    return {
        "demand_total": float(sf["final_demand"].sum()),
        "actual_sales_total": float(sf["final_demand"].sum()),
        "lost_sales_total": 0.0,
        "revenue_total": float(eco["total_revenue"].sum()),
        "profit_total": float(eco["profit"].sum()),
        "margin": float(eco["profit"].sum() / eco["total_revenue"].sum()) if float(eco["total_revenue"].sum()) > 0 else 0.0,
        "confidence": 0.5,
        "uncertainty": 0.5,
        "ood_flags": scenario_result.get("ood_flags", []),
    }
