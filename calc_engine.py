from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def sanitize_discount(discount: Any) -> float:
    val = float(pd.to_numeric(pd.Series([discount]), errors="coerce").fillna(0.0).iloc[0])
    return float(np.clip(val, 0.0, 0.95))


def sanitize_non_negative(value: Any, fallback: float = 0.0) -> float:
    val = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(fallback).iloc[0])
    if not np.isfinite(val):
        val = fallback
    return float(max(0.0, val))


def compute_daily_unit_economics(
    daily: pd.DataFrame,
    unit_price_col: str = "price",
    unit_cost_col: str = "cost",
    quantity_col: str = "pred_sales",
    discount_col: str = "discount",
    freight_col: str = "freight_value",
    stock_cap: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = daily.copy()
    out["unit_price"] = pd.to_numeric(out.get(unit_price_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["unit_cost"] = pd.to_numeric(out.get(unit_cost_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["pred_quantity"] = pd.to_numeric(out.get(quantity_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["raw_discount_rate"] = pd.to_numeric(out.get(discount_col, 0.0), errors="coerce").fillna(0.0)
    out["discount_rate"] = out["raw_discount_rate"].clip(lower=0.0, upper=0.95)
    freight_raw = out[freight_col] if freight_col in out.columns else pd.Series(0.0, index=out.index)
    out["freight_value"] = pd.to_numeric(freight_raw, errors="coerce").fillna(0.0).clip(lower=0.0)

    if stock_cap and stock_cap > 0:
        out["pred_quantity"] = np.minimum(out["pred_quantity"], float(stock_cap))

    out["effective_unit_price"] = out["unit_price"] * (1.0 - out["discount_rate"])
    out["total_revenue"] = out["effective_unit_price"] * out["pred_quantity"]
    out["unit_variable_cost"] = out["unit_cost"] + out["freight_value"]
    out["total_cost"] = out["unit_variable_cost"] * out["pred_quantity"]
    out["profit"] = out["total_revenue"] - out["total_cost"]
    out["margin"] = np.where(out["total_revenue"] > 0, out["profit"] / out["total_revenue"], 0.0)

    checks: List[str] = []
    if (out["total_revenue"] < -1e-9).any():
        checks.append("negative_revenue")
    if (out["pred_quantity"] < -1e-9).any():
        checks.append("negative_quantity")
    if ((out["raw_discount_rate"] < 0) | (out["raw_discount_rate"] > 0.95)).any():
        checks.append("discount_raw_out_of_range")
    if ((out["margin"] > 0) & (out["profit"] < 0)).any():
        checks.append("positive_margin_negative_profit")
    return out, {"sanity_warnings": checks}
