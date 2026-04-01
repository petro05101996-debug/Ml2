from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .core import CONFIG, compute_monthly_group_elasticities, estimate_pooled_elasticity


def evaluate_price_signal(daily_df: pd.DataFrame) -> Dict[str, Any]:
    df = daily_df.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    history_days = int((dates.max() - dates.min()).days + 1) if len(df) else 0
    total_sales = float(pd.to_numeric(df.get("sales", 0.0), errors="coerce").fillna(0.0).sum())
    price = pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0)
    unique_prices = int(price.nunique())
    price_changes = int((price.diff().fillna(0.0) != 0.0).sum())
    median_price = float(price.median()) if len(price) else 0.0
    rel_span = float((price.max() - price.min()) / median_price) if median_price > 0 else 0.0

    issues: List[str] = []
    if history_days < CONFIG["MIN_COVERAGE_DAYS"]:
        issues.append("history_days_below_min")
    if total_sales < CONFIG["MIN_TOTAL_SALES"]:
        issues.append("total_sales_below_min")
    if unique_prices < CONFIG["MIN_UNIQUE_PRICES"]:
        issues.append("unique_prices_below_min")
    if price_changes < CONFIG["MIN_PRICE_CHANGES"]:
        issues.append("price_changes_below_min")
    if rel_span < CONFIG["MIN_REL_PRICE_SPAN"]:
        issues.append("rel_price_span_below_min")

    return {
        "history_days": history_days,
        "total_sales": total_sales,
        "unique_prices": unique_prices,
        "price_changes": price_changes,
        "rel_price_span": rel_span,
        "can_recommend_price": len(issues) == 0,
        "issues": issues,
    }


def estimate_v1_elasticity(train_df: pd.DataFrame) -> Dict[str, Any]:
    pooled = float(estimate_pooled_elasticity(train_df, small_mode=False))
    monthly, _ = compute_monthly_group_elasticities(train_df, pooled, small_mode=False)
    return {
        "pooled_elasticity": pooled,
        "elasticity_by_month": monthly,
        "price_signal": evaluate_price_signal(train_df),
        "estimation_method": "pooled_plus_monthly_shrinkage",
    }


def compute_price_multiplier(
    candidate_price: float,
    reference_price: float,
    elasticity: float,
    floor: float = 0.25,
    cap: float = 4.0,
) -> float:
    ratio = max(float(candidate_price), 1e-9) / max(float(reference_price), 1e-9)
    mult = ratio ** float(elasticity)
    return float(np.clip(mult, floor, cap))
