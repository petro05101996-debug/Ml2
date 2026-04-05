from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from df_utils import get_numeric_series


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
    unit_price_input_type: str = "net",
    economics_mode: str = "net_price",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = daily.copy()
    out["unit_price"] = get_numeric_series(out, unit_price_col, 0.0).fillna(0.0).clip(lower=0.0)
    out["unit_cost"] = get_numeric_series(out, unit_cost_col, 0.0).fillna(0.0).clip(lower=0.0)
    out["actual_quantity"] = get_numeric_series(out, quantity_col, 0.0).fillna(0.0).clip(lower=0.0)
    out["pred_quantity"] = out["actual_quantity"]
    out["raw_discount_rate"] = get_numeric_series(out, discount_col, 0.0).fillna(0.0)
    out["discount_rate"] = out["raw_discount_rate"].clip(lower=0.0, upper=0.95)
    out["freight_value"] = get_numeric_series(out, freight_col, 0.0).fillna(0.0).clip(lower=0.0)

    normalized_price_type = str(unit_price_input_type).strip().lower()
    normalized_mode = str(economics_mode).strip().lower()
    out["unit_price_input_type"] = normalized_price_type if normalized_price_type in {"net", "list"} else "net"
    out["economics_mode"] = normalized_mode if normalized_mode in {"net_price", "list_less_discount"} else "net_price"
    use_list_less_discount = (out["economics_mode"] == "list_less_discount") | (out["unit_price_input_type"] == "list")
    out["effective_unit_price"] = np.where(
        use_list_less_discount,
        out["unit_price"] * (1.0 - out["discount_rate"].clip(lower=0.0, upper=0.95)),
        out["unit_price"],
    )
    out["total_revenue"] = out["effective_unit_price"] * out["actual_quantity"]
    out["unit_variable_cost"] = out["unit_cost"] + out["freight_value"]
    out["total_cost"] = out["unit_variable_cost"] * out["actual_quantity"]
    out["profit"] = out["total_revenue"] - out["total_cost"]
    out["margin"] = np.where(out["total_revenue"] > 0, out["profit"] / out["total_revenue"], 0.0)

    checks: List[str] = []
    if (out["total_revenue"] < -1e-9).any():
        checks.append("negative_revenue")
    if (out["actual_quantity"] < -1e-9).any():
        checks.append("negative_quantity")
    if ((out["raw_discount_rate"] < 0) | (out["raw_discount_rate"] > 0.95)).any():
        checks.append("discount_raw_out_of_range")
    if ((out["margin"] > 0) & (out["profit"] < 0)).any():
        checks.append("positive_margin_negative_profit")
    return out, {"sanity_warnings": checks}
