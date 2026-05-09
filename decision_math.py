from __future__ import annotations

import math
from typing import Any, Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def finite_or_none(value: Any):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def safe_float(value: Any, default: float = 0.0) -> float:
    val = finite_or_none(value)
    return float(default) if val is None else float(val)


def clamp(value: Any, lo: float, hi: float) -> float:
    val = safe_float(value, lo)
    return float(max(float(lo), min(float(hi), val)))


def score_0_100(value: Any) -> float:
    return clamp(value, 0.0, 100.0)


def safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    n = finite_or_none(num)
    d = finite_or_none(den)
    if n is None or d is None or abs(d) < 1e-9:
        return float(default)
    out = n / d
    return float(out) if math.isfinite(out) else float(default)


def safe_pct_delta(new_value: Any, base_value: Any) -> float:
    base = finite_or_none(base_value)
    new = finite_or_none(new_value)
    if base is None or new is None or abs(base) < 1e-9:
        return 0.0
    out = (new - base) / abs(base) * 100.0
    return float(out) if math.isfinite(out) else 0.0


def _get_path(obj: Any, path: Any):
    cur = obj
    parts = path if isinstance(path, (list, tuple)) else str(path).split(".")
    for part in parts:
        if cur is None:
            return None
        if pd is not None and isinstance(cur, pd.DataFrame):
            if len(cur) == 0:
                return None
            if part in cur.columns:
                cur = cur.iloc[0][part]
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)) and str(part).isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
    return cur


def extract_first_number(paths: Iterable[Any], obj: Any, default=None):
    for path in paths:
        val = finite_or_none(_get_path(obj, path))
        if val is not None:
            return val
    return default


def extract_metric(results: Any, *paths: Any, default=None):
    if not paths:
        return default
    for path in paths:
        raw = _get_path(results, path)
        val = finite_or_none(raw)
        if val is not None:
            return val
    return default
