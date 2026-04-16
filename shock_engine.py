from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


VALID_TYPES = {"percent", "units"}
MIN_PERCENT_SHOCK = -0.8
MAX_PERCENT_SHOCK = 2.0


def validate_shocks(shocks: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for raw in shocks or []:
        row = dict(raw)
        st = str(row.get("shock_type", "")).strip().lower()
        if st not in VALID_TYPES:
            warnings.append(f"Skipped shock with unknown type: {row.get('shock_name', 'unnamed')}")
            continue
        row["shock_type"] = st
        raw_value = float(row.get("shock_value", 0.0))
        if st == "percent":
            clipped = float(np.clip(raw_value, MIN_PERCENT_SHOCK, MAX_PERCENT_SHOCK))
            if abs(clipped - raw_value) > 1e-12:
                warnings.append(
                    f"Shock {row.get('shock_name', 'unnamed')} clipped to bounds "
                    f"[{MIN_PERCENT_SHOCK}, {MAX_PERCENT_SHOCK}]"
                )
            row["shock_value"] = clipped
        else:
            row["shock_value"] = raw_value
        row["units_mode"] = str(row.get("units_mode", "per_day")).strip().lower()
        row["start_date"] = pd.to_datetime(row.get("start_date"), errors="coerce")
        row["end_date"] = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.isna(row["start_date"]) or pd.isna(row["end_date"]):
            warnings.append(f"Skipped shock with invalid dates: {row.get('shock_name', 'unnamed')}")
            continue
        out.append(row)
    return out, warnings


def compute_shock_multiplier(dates: pd.Series, shocks: List[Dict[str, Any]]) -> np.ndarray:
    mult = np.ones(len(dates), dtype=float)
    dts = pd.to_datetime(dates)
    for s in shocks:
        if s["shock_type"] != "percent":
            continue
        mask = (dts >= s["start_date"]) & (dts <= s["end_date"])
        mult[mask.values if hasattr(mask, 'values') else mask] *= max(0.0, 1.0 + float(s["shock_value"]))
    return mult


def compute_shock_units(dates: pd.Series, shocks: List[Dict[str, Any]]) -> np.ndarray:
    add = np.zeros(len(dates), dtype=float)
    dts = pd.to_datetime(dates)
    for s in shocks:
        if s["shock_type"] != "units":
            continue
        mask = (dts >= s["start_date"]) & (dts <= s["end_date"])
        idx = mask.values if hasattr(mask, "values") else mask
        if str(s.get("units_mode", "per_day")) == "total_over_window":
            denom = max(int(np.sum(idx)), 1)
            add[idx] += float(s["shock_value"]) / float(denom)
        else:
            add[idx] += float(s["shock_value"])
    return add
