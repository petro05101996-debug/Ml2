from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from calc_engine import compute_daily_unit_economics
from .v1_forecast import recursive_v1_demand_forecast

JUMP_PENALTY_RATE = 0.01
OOD_PENALTY_RATE = 0.03
VOLUME_GUARD_PENALTY_RATE = 0.05
RISK_LAMBDA_FLOOR = 0.1


def simulate_v1_horizon_profit(
    base_row: Dict[str, Any],
    candidate_price: float,
    future_dates_df: pd.DataFrame,
    demand_models: List[Any],
    base_history: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
    unit_cost: Optional[float] = None,
    allow_extrapolate: bool = False,
    risk_lambda: float = 0.7,
) -> Dict[str, Any]:
    local_ctx = dict(base_ctx)
    local_ctx["price"] = float(candidate_price)
    if unit_cost is not None:
        local_ctx["cost"] = float(unit_cost)

    fc = recursive_v1_demand_forecast(demand_models, base_history, future_dates_df, local_ctx, feature_spec)
    daily = fc.copy()
    daily["pred_demand"] = pd.to_numeric(daily["pred_sales"], errors="coerce").fillna(0.0)
    daily, econ_checks = compute_daily_unit_economics(daily, quantity_col="pred_sales")
    daily["pred_sales"] = daily["pred_quantity"]

    total_profit = float(daily["profit"].sum()) if len(daily) else 0.0
    total_revenue = float(daily["total_revenue"].sum()) if len(daily) else 0.0
    total_volume = float(daily["pred_quantity"].sum()) if len(daily) else 0.0

    current_price = float(base_row.get("price", candidate_price))
    risk_scale = max(RISK_LAMBDA_FLOOR, float(risk_lambda))
    jump_penalty = abs(candidate_price - current_price) / max(current_price, 1e-9) * JUMP_PENALTY_RATE * max(total_profit, 0.0) * risk_scale
    train_min = float(pd.to_numeric(base_history.get("price", current_price), errors="coerce").min()) if len(base_history) else current_price
    train_max = float(pd.to_numeric(base_history.get("price", current_price), errors="coerce").max()) if len(base_history) else current_price
    ood_penalty = 0.0 if allow_extrapolate else (OOD_PENALTY_RATE * max(total_profit, 0.0) * risk_scale if (candidate_price < train_min or candidate_price > train_max) else 0.0)

    baseline_sim = recursive_v1_demand_forecast(demand_models, base_history, future_dates_df, dict(base_ctx), feature_spec)
    baseline_volume = float(pd.to_numeric(baseline_sim.get("pred_sales", 0.0), errors="coerce").fillna(0.0).sum()) if len(baseline_sim) else 0.0
    volume_guard_penalty = VOLUME_GUARD_PENALTY_RATE * max(total_profit, 0.0) * risk_scale if baseline_volume > 0 and total_volume < baseline_volume * 0.9 else 0.0
    adjusted_profit = float(total_profit - jump_penalty - ood_penalty - volume_guard_penalty)

    return {
        "daily": daily,
        "total_profit": total_profit,
        "total_revenue": total_revenue,
        "total_volume": total_volume,
        "baseline_volume": baseline_volume,
        "adjusted_profit": adjusted_profit,
        "jump_penalty": float(jump_penalty),
        "ood_penalty": float(ood_penalty),
        "volume_guard_penalty": float(volume_guard_penalty),
        "sanity_warnings": econ_checks.get("sanity_warnings", []),
    }


def recommend_v1_price_horizon(base_row: Dict[str, Any], demand_models: List[Any], base_history: pd.DataFrame, base_ctx: Dict[str, Any], feature_spec: Dict[str, Any], n_days: int = 30, objective_mode: str = "maximize_profit", risk_lambda: float = 0.7, can_recommend: bool = True) -> Dict[str, Any]:
    current_price = float(base_row.get("price", 0.0))
    spread = (0.95, 1.05) if not can_recommend else (0.85, 1.15)
    grid = np.linspace(current_price * spread[0], current_price * spread[1], 11)
    future_dates_df = pd.DataFrame({"date": pd.date_range(pd.Timestamp(base_history["date"].max()) + pd.Timedelta(days=1), periods=int(n_days), freq="D")})

    results = []
    for p in grid:
        sim = simulate_v1_horizon_profit(base_row, float(p), future_dates_df, demand_models, base_history, base_ctx, feature_spec, risk_lambda=float(risk_lambda))
        margin_ratio = float(sim["total_profit"] / sim["total_revenue"]) if sim["total_revenue"] > 0 else 0.0
        results.append({"price": float(p), "sim": sim, "adjusted_profit": sim["adjusted_profit"], "revenue": sim["total_revenue"], "pred_volume": sim["total_volume"], "baseline_volume": sim["baseline_volume"], "margin_ratio": margin_ratio})

    current = min(results, key=lambda r: abs(r["price"] - current_price))
    allowed = list(results)
    if objective_mode == "protect_volume":
        allowed = [r for r in allowed if r["pred_volume"] >= r["baseline_volume"] * 0.95]
    elif objective_mode == "maximize_revenue":
        allowed = [r for r in allowed if r["pred_volume"] >= r["baseline_volume"] * 0.90]

    best = max(allowed or results, key=(lambda r: r["revenue"]) if objective_mode == "maximize_revenue" else (lambda r: r["adjusted_profit"]))
    return {
        "best_price": float(base_row.get("price", current_price)) if not can_recommend else float(best["price"]),
        "best_idx": results.index(best),
        "results": [{"price": r["price"], "adjusted_profit": r["adjusted_profit"], "revenue": r["revenue"], "pred_volume": r["pred_volume"], "margin_ratio": r["margin_ratio"]} for r in results],
        "best_sim": best["sim"],
        "current_sim": current["sim"],
    }
