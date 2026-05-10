from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _delta_pct(value: float, baseline: float) -> float:
    if abs(baseline) <= 1e-12:
        return 0.0
    return (value - baseline) / abs(baseline)


def run_scenario_set(
    trained_bundle: Dict[str, Any],
    scenario_rows: List[Dict[str, Any]],
    runner: Callable[..., Dict[str, Any]],
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    baseline = None
    for row in scenario_rows:
        overrides = row.get("overrides", {})
        result = runner(
            trained_bundle,
            manual_price=float(row["price"]),
            freight_multiplier=float(row["freight_multiplier"]),
            demand_multiplier=float(row.get("demand_multiplier", 1.0)),
            horizon_days=int(row["horizon_days"]),
            discount_multiplier=float(row.get("discount_multiplier", 1.0)),
            cost_multiplier=float(row.get("cost_multiplier", 1.0)),
            stock_cap=float(row.get("stock_cap", 0.0)),
            overrides=overrides,
            **(runner_kwargs or {}),
        )
        if baseline is None and row["name"] == "Baseline":
            baseline = result

        sales = _safe_float(result.get("demand_total", 0.0))
        revenue_raw = _safe_float(result.get("revenue_total_raw", result.get("revenue_total", 0.0)))
        revenue_adjusted = _safe_float(result.get("revenue_total_adjusted", result.get("revenue_total", revenue_raw)))
        profit_raw = _safe_float(result.get("profit_total_raw", result.get("profit_total", 0.0)))
        profit_adjusted = _safe_float(result.get("profit_total_adjusted", profit_raw))
        margin_raw = (profit_raw / revenue_raw) if revenue_raw > 0 else 0.0
        margin_adjusted = (profit_adjusted / revenue_adjusted) if revenue_adjusted > 0 else 0.0
        confidence = float(np.clip(_safe_float(result.get("confidence", 0.0)), 0.0, 1.0))
        uncertainty = float(np.clip(_safe_float(result.get("uncertainty", 1.0 - confidence)), 0.0, 1.0))
        records.append(
            {
                "scenario": row["name"],
                "metric_basis": "adjusted",
                "sales": sales,
                "revenue_raw": revenue_raw,
                "revenue_adjusted": revenue_adjusted,
                "revenue": revenue_adjusted,
                "profit_raw": profit_raw,
                "profit_adjusted": profit_adjusted,
                "profit": profit_adjusted,
                "margin_raw": margin_raw,
                "margin_adjusted": margin_adjusted,
                "margin": margin_adjusted,
                "confidence": confidence,
                "confidence_is_advisory": True,
                "confidence_note": "Heuristic/advisory confidence; not a guarantee.",
                "uncertainty": uncertainty,
                "score": profit_adjusted * (0.7 + 0.3 * confidence),
            }
        )

    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    base_df = df[df["scenario"] == "Baseline"]
    base = base_df.iloc[0] if len(base_df) else df.iloc[0]
    metric_cols = [
        "sales",
        "revenue_raw",
        "revenue_adjusted",
        "revenue",
        "profit_raw",
        "profit_adjusted",
        "profit",
        "margin_raw",
        "margin_adjusted",
        "margin",
    ]
    for c in metric_cols:
        base_value = float(base[c])
        df[f"baseline_{c}"] = base_value
        df[f"delta_{c}"] = df[c] - base_value
        df[f"delta_{c}_pct"] = df[c].apply(lambda value, b=base_value: _delta_pct(float(value), b))
    return df


def build_sensitivity_grid(
    trained_bundle: Dict[str, Any],
    base_price: float,
    runner: Callable[..., Dict[str, Any]],
    price_steps: int = 9,
    demand_steps: int = 9,
    horizon_days: int = 30,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    price_grid = np.linspace(base_price * 0.85, base_price * 1.15, price_steps)
    demand_grid = np.linspace(0.8, 1.2, demand_steps)
    rows: List[Dict[str, Any]] = []
    for p in price_grid:
        for d in demand_grid:
            r = runner(trained_bundle, manual_price=float(p), horizon_days=int(horizon_days), demand_multiplier=float(d), **(runner_kwargs or {}))
            rows.append({
                "price": p,
                "demand_multiplier": d,
                "profit_raw": float(r.get("profit_total_raw", r.get("profit_total", 0.0))),
                "profit_adjusted": float(r.get("profit_total_adjusted", r.get("profit_total", 0.0))),
                "forecast_total": float(r.get("demand_total", 0.0)),
            })
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["risk_zone"] = np.where(out["profit_adjusted"] < 0, "risk", "stable")
    return out
