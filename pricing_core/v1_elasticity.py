from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_price_multiplier(price_candidate: float, reference_price: float, elasticity: float) -> float:
    ratio = max(float(price_candidate), 1e-9) / max(float(reference_price), 1e-9)
    return float(np.clip(np.exp(float(elasticity) * np.log(np.clip(ratio, 0.2, 5.0))), 0.15, 3.0))


def estimate_v1_elasticity(df: pd.DataFrame) -> Dict[str, Any]:
    return {"elasticity_by_month": {}, "pooled_elasticity": -1.0}


def evaluate_price_signal(df: pd.DataFrame) -> Dict[str, Any]:
    p = pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0)
    return {
        "history_days": int((pd.to_datetime(df["date"]).max() - pd.to_datetime(df["date"]).min()).days + 1) if len(df) else 0,
        "total_sales": float(pd.to_numeric(df.get("sales", 0.0), errors="coerce").fillna(0.0).sum()),
        "unique_prices": int(p.nunique()),
        "price_changes": int((p.diff().fillna(0) != 0).sum()),
        "rel_price_span": float((p.max() - p.min()) / max(p.median(), 1e-9)) if len(p) else 0.0,
    }
