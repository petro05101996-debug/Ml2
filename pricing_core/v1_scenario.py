from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from calc_engine import sanitize_discount, sanitize_non_negative
from .v1_optimizer import simulate_v1_horizon_profit


def run_v1_what_if_projection(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: Optional[int] = None,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float = 0.0,
    scenario: Optional[Dict[str, Any]] = None,
    include_sensitivity: bool = True,
) -> Dict[str, Any]:
    _ = include_sensitivity
    base_history = trained_bundle["target_history"].copy() if "target_history" in trained_bundle else trained_bundle["daily_base"].copy()
    base_ctx = dict(trained_bundle["base_ctx"])

    scenario_factors = dict((scenario or {}).get("factors", {}))
    allowed = set(trained_bundle["feature_spec"].get("scenario_features", []))
    scenario_factors = {k: v for k, v in scenario_factors.items() if k in allowed}

    for k, v in scenario_factors.items():
        base_ctx[k] = float(v)

    base_ctx["price"] = sanitize_non_negative(base_ctx.get("price", manual_price), fallback=float(manual_price))
    base_ctx["discount"] = sanitize_discount(float(base_ctx.get("discount", 0.0)) * float(discount_multiplier))
    base_ctx["cost"] = sanitize_non_negative(float(base_ctx.get("cost", base_ctx["price"] * 0.65)) * float(cost_multiplier))
    base_ctx["freight_value"] = sanitize_non_negative(float(base_ctx.get("freight_value", 0.0)) * float(freight_multiplier))

    n_days = int(horizon_days or len(trained_bundle.get("future_dates", [])) or 30)
    future_dates = pd.DataFrame({"date": pd.date_range(pd.Timestamp(base_history["date"].max()) + pd.Timedelta(days=1), periods=n_days, freq="D")})
    sim = simulate_v1_horizon_profit(
        trained_bundle["base_ctx"],
        float(base_ctx["price"]),
        future_dates,
        trained_bundle["demand_models"],
        base_history,
        base_ctx,
        trained_bundle["feature_spec"],
        risk_lambda=float(trained_bundle.get("risk_lambda", 0.7)),
        calibration_factor=float(trained_bundle.get("calibration_factor", 1.0)),
    )

    daily = sim["daily"].copy()
    daily["pred_demand"] = daily["pred_demand"] * float(demand_multiplier)
    daily["pred_sales"] = daily["pred_sales"] * float(demand_multiplier)

    weak_factors = set(trained_bundle.get("factor_diagnostics", {}).get("weak_factors", []))
    weak_used = sorted([f for f in scenario_factors if f in weak_factors])
    ood_flags = list(sim.get("ood_flags", []))
    hist = base_history.copy()
    for f, v in scenario_factors.items():
        s = pd.to_numeric(hist.get(f, None), errors="coerce").dropna() if f in hist.columns else pd.Series(dtype=float)
        if len(s) >= 10:
            if float(v) < float(s.quantile(0.01)) or float(v) > float(s.quantile(0.99)):
                ood_flags.append(f"{f}_scenario_ood")
    ood_flags = sorted(set(ood_flags))
    price_signal_ok = bool(trained_bundle.get("factor_diagnostics", {}).get("price_signal_ok", False))

    conf = 0.7
    if weak_used:
        conf -= 0.2
    if ood_flags:
        conf -= 0.2
    if "price" in scenario_factors and not price_signal_ok:
        conf -= 0.3
    scenario_confidence = float(max(0.05, min(0.95, conf)))

    return {
        "daily": daily,
        "demand_total": float(daily["pred_demand"].sum()),
        "actual_sales_total": float(daily["pred_sales"].sum()),
        "lost_sales_total": 0.0,
        "profit_total": float(daily["profit"].sum()),
        "revenue_total": float(daily["total_revenue"].sum()),
        "confidence": scenario_confidence,
        "uncertainty": 1.0 - scenario_confidence,
        "scenario_confidence": scenario_confidence,
        "weak_factor_warnings": weak_used,
        "ood_flags": ood_flags,
        "sanity_warnings": sim.get("sanity_warnings", []),
        "scenario_meta": {"factors": scenario_factors, "horizon_days": n_days, "stock_cap": float(stock_cap)},
        "local_sensitivity": {},
    }
