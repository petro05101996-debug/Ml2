from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd


def run_scenario_set(trained_bundle: Dict[str, Any], scenario_rows: List[Dict[str, Any]], runner: Callable[..., Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in scenario_rows:
        factors = {
            "price": float(row["price"]),
            "promotion": float(row.get("promotion", 0.0)),
            "review_score": float(row.get("review_score", row.get("rating", 4.5))),
            "reviews_count": float(row.get("reviews_count", 0.0)),
        }
        if "discount" in row:
            factors["discount"] = float(row.get("discount", 0.0))
        for k, v in row.items():
            if str(k).startswith("user_factor__"):
                factors[k] = float(v)

        result = runner(
            trained_bundle,
            manual_price=float(row["price"]),
            freight_multiplier=float(row.get("freight_multiplier", 1.0)),
            demand_multiplier=float(row.get("demand_multiplier", 1.0)),
            horizon_days=int(row["horizon_days"]),
            discount_multiplier=float(row.get("discount_multiplier", 1.0)),
            cost_multiplier=float(row.get("cost_multiplier", 1.0)),
            stock_cap=float(row.get("stock_cap", 0.0)),
            scenario={"name": str(row.get("name", "scenario")), "mode": "manual", "horizon_days": int(row["horizon_days"]), "factors": factors},
        )
        sales = float(result.get("demand_total", 0.0))
        revenue = float(result.get("revenue_total", 0.0))
        profit = float(result.get("profit_total", 0.0))
        records.append({"scenario": row["name"], "price": float(row["price"]), "sales": sales, "actual_sales": float(result.get("actual_sales_total", sales)), "lost_sales": float(result.get("lost_sales_total", 0.0)), "revenue": revenue, "profit": profit, "margin": (profit / revenue) if revenue > 0 else 0.0, "confidence": float(result.get("confidence", 0.0)), "uncertainty": float(result.get("uncertainty", 1.0)), "score": profit * (0.7 + 0.3 * float(result.get("confidence", 0.0)))})

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    base = (df[df["scenario"] == "Baseline"].iloc[0] if len(df[df["scenario"] == "Baseline"]) else df.iloc[0])
    for c in ["sales", "revenue", "profit", "margin"]:
        df[f"delta_{c}"] = df[c] - float(base[c])
    return df


def build_sensitivity_grid(trained_bundle: Dict[str, Any], base_price: float, runner: Callable[..., Dict[str, Any]], price_steps: int = 9, demand_steps: int = 9) -> pd.DataFrame:
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
