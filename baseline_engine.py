from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


NEUTRAL_BASELINE_FEATURES: List[str] = [
    "sales_lag1w",
    "sales_lag2w",
    "sales_lag4w",
    "sales_lag8w",
    "sales_ma4w",
    "sales_ma8w",
    "sales_ma12w",
    "sales_std4w",
    "sales_std8w",
    "week_sin",
    "week_cos",
    "trend_idx",
    "stock_mean",
    "stockout_share",
]


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return neutral baseline feature subset (no scenario-driven exogenous factors)."""
    out = df.copy()
    keep = [c for c in NEUTRAL_BASELINE_FEATURES if c in out.columns]
    return out[keep].copy()


def train_baseline_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    v1 neutral baseline placeholder: expose feature contract for training pipeline.
    Existing training logic remains in app.py; this function documents the active contract.
    """
    return {
        "mode": "neutral_baseline_v1",
        "features": [c for c in NEUTRAL_BASELINE_FEATURES if c in df.columns],
        "rows": int(len(df)),
    }


def predict_baseline(df_future: pd.DataFrame) -> pd.Series:
    """Read already computed baseline units from scenario preparation output."""
    if "baseline_units" in df_future.columns:
        return pd.to_numeric(df_future["baseline_units"], errors="coerce").fillna(0.0)
    if "base_pred_sales" in df_future.columns:
        return pd.to_numeric(df_future["base_pred_sales"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df_future), index=df_future.index, dtype=float)
