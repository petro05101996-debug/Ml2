from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd


def run_scenario_set(
    trained_bundle: Dict[str, Any],
    scenario_rows: List[Dict[str, Any]],
    runner: Callable[..., Dict[str, Any]],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    baseline = None
    for row in scenario_rows:
        result = runner(
            trained_bundle,
            manual_price=float(row["price"]),
            freight_multiplier=float(row["freight_multiplier"]),
            demand_multiplier=float(row["demand_multiplier"]),
            horizon_days=int(row["horizon_days"]),
            discount_multiplier=float(row.get("discount_multiplier", 1.0)),
            cost_multiplier=float(row.get("cost_multiplier", 1.0)),
            stock_cap=float(row.get("stock_cap", 0.0)),
            scenario={
                "name": str(row.get("name", "scenario")),
                "mode": "manual",
                "horizon_days": int(row["horizon_days"]),
                "factors": {
                    "price": float(row["price"]),
                    "promotion": float(row.get("promotion", 0.0)),
                    "rating": float(row.get("rating", 4.5)),
                    "reviews_count": float(row.get("reviews_count", 0.0)),
                },
            },
        )
        if baseline is None and row["name"] == "Baseline":
            baseline = result

        sales = float(result.get("demand_total", 0.0))
        actual_sales = float(result.get("actual_sales_total", sales))
        lost_sales = float(result.get("lost_sales_total", 0.0))
        revenue = float(result.get("revenue_total", 0.0))
        profit = float(result.get("profit_total", 0.0))
        margin = (profit / revenue) if revenue > 0 else 0.0
        confidence = float(result.get("confidence", 0.0))
        uncertainty = float(result.get("uncertainty", 1.0 - confidence))
        records.append(
            {
                "scenario": row["name"],
                "price": float(row["price"]),
                "sales": sales,
                "actual_sales": actual_sales,
                "lost_sales": lost_sales,
                "revenue": revenue,
                "profit": profit,
                "margin": margin,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "score": profit * (0.7 + 0.3 * confidence),
            }
        )

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    base_df = df[df["scenario"] == "Baseline"]
    base = base_df.iloc[0] if len(base_df) else df.iloc[0]
    for c in ["sales", "revenue", "profit", "margin"]:
        df[f"delta_{c}"] = df[c] - float(base[c])
    return df


def build_sensitivity_grid(
    trained_bundle: Dict[str, Any],
    base_price: float,
    runner: Callable[..., Dict[str, Any]],
    price_steps: int = 9,
    demand_steps: int = 9,
) -> pd.DataFrame:
    price_grid = np.linspace(base_price * 0.85, base_price * 1.15, price_steps)
    demand_grid = np.linspace(0.8, 1.2, demand_steps)
    rows: List[Dict[str, Any]] = []
    for p in price_grid:
        for d in demand_grid:
            r = runner(trained_bundle, manual_price=float(p), demand_multiplier=float(d), horizon_days=30)
            rows.append({"price": p, "demand_multiplier": d, "profit": float(r.get("profit_total", 0.0))})
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["risk_zone"] = np.where(out["profit"] < 0, "risk", "stable")
    return out
