from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

ALLOWED_SHAPES = {"flat", "pulse", "ramp_up", "ramp_down", "decay", "spike_then_fade"}
ALLOWED_DIRECTIONS = {"positive", "negative"}


def validate_shocks(shocks) -> List[str]:
    warns: List[str] = []
    if not shocks:
        return warns
    for i, s in enumerate(shocks):
        shape = s.get("shape", "flat")
        direction = s.get("direction", "positive")
        intensity = float(s.get("intensity", 0.0))
        if shape not in ALLOWED_SHAPES:
            raise ValueError(f"unsupported shape: {shape}")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"unsupported direction: {direction}")
        if intensity <= 0:
            raise ValueError("intensity must be > 0")
        sd = pd.Timestamp(s.get("start_date"))
        ed = pd.Timestamp(s.get("end_date"))
        if sd > ed:
            raise ValueError("start_date must be <= end_date")
        if "scope" not in s:
            warns.append(f"shock_{i}_scope_defaulted")
        if "confidence" not in s:
            warns.append(f"shock_{i}_confidence_defaulted")
    return warns


def build_default_no_shock_profile(future_dates_df) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(future_dates_df["date"]), "shock_multiplier": 1.0})


def _effect_curve(shape: str, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    if shape == "flat":
        return np.ones(n)
    if shape == "pulse":
        return np.array([1.0] + [0.0] * (n - 1))
    if shape == "ramp_up":
        return np.linspace(0.0, 1.0, n)
    if shape == "ramp_down":
        return np.linspace(1.0, 0.0, n)
    if shape == "decay":
        x = np.linspace(0, 2.5, n)
        return np.exp(-x)
    if shape == "spike_then_fade":
        return np.array([1.0] + list(np.linspace(0.7, 0.0, max(0, n - 1))))
    return np.ones(n)


def build_shock_profile(shocks: List[Dict[str, Any]], future_dates_df: pd.DataFrame) -> pd.DataFrame:
    validate_shocks(shocks)
    out = build_default_no_shock_profile(future_dates_df)
    out = out.set_index("date")
    for s in shocks or []:
        sd, ed = pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"])
        idx = out.index[(out.index >= sd) & (out.index <= ed)]
        curve = _effect_curve(s.get("shape", "flat"), len(idx))
        sign = 1.0 if s.get("direction", "positive") == "positive" else -1.0
        intensity = float(s.get("intensity", 0.0))
        mult = 1.0 + sign * intensity * curve
        if len(idx):
            out.loc[idx, "shock_multiplier"] = out.loc[idx, "shock_multiplier"].values * mult
    out["shock_multiplier"] = pd.to_numeric(out["shock_multiplier"], errors="coerce").fillna(1.0).clip(0.2, 5.0)
    return out.reset_index()
