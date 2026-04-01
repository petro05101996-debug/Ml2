from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from pricing_core import run_full_pricing_analysis, run_full_pricing_analysis_universal_v1


UNIVERSAL_LOAD_MODES = {
    "Universal CSV",
    "Универсальный CSV",
}


def _read_csv_input(source: Any) -> pd.DataFrame:
    if source is None:
        raise ValueError("CSV source is required.")
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source)


def run_analysis_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    load_mode = ctx.get("load_mode", "Legacy Olist (3 CSV)")
    target_category = ctx["target_category"]
    target_sku = ctx["target_sku"]
    horizon_days = int(ctx.get("forecast_horizon_days", 30))

    caution_to_risk_lambda = {"Низкий": 0.45, "Средний": 0.7, "Высокий": 1.0}
    selected_caution = ctx.get("caution_level", "Средний")
    risk_lambda = float(caution_to_risk_lambda.get(selected_caution, 0.7))

    if load_mode in UNIVERSAL_LOAD_MODES:
        return run_full_pricing_analysis_universal_v1(
            ctx["universal_txn"],
            target_category,
            target_sku,
            objective_mode=ctx.get("objective_mode", "maximize_profit"),
            horizon_days=horizon_days,
            risk_lambda=risk_lambda,
        )

    orders_file = ctx.get("orders_file")
    items_file = ctx.get("items_file")
    products_file = ctx.get("products_file")
    reviews_file = ctx.get("reviews_file")

    orders = _read_csv_input(orders_file)
    items = _read_csv_input(items_file)
    products = _read_csv_input(products_file)
    reviews = _read_csv_input(reviews_file) if reviews_file is not None else pd.DataFrame()

    if ctx.get("orders_col_map"):
        orders = orders.rename(columns=ctx["orders_col_map"])
    if ctx.get("items_col_map"):
        items = items.rename(columns=ctx["items_col_map"])
    if ctx.get("products_col_map"):
        products = products.rename(columns=ctx["products_col_map"])
    if ctx.get("reviews_col_map") and len(reviews):
        reviews = reviews.rename(columns=ctx["reviews_col_map"])

    if "order_item_id" not in items.columns:
        items["order_item_id"] = np.arange(1, len(items) + 1)
    if "freight_value" not in items.columns:
        items["freight_value"] = 0.0

    return run_full_pricing_analysis(
        orders,
        items,
        products,
        reviews,
        target_category,
        target_sku,
        horizon_days=horizon_days,
        risk_lambda=risk_lambda,
    )
