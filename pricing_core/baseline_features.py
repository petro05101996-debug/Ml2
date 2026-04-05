from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_BASELINE_NUMERIC = [
    "sales_lag1",
    "sales_lag7",
    "sales_lag14",
    "sales_lag28",
    "sales_ma7",
    "sales_ma14",
    "sales_ma28",
    "sales_std28",
    "dow",
    "week_of_year",
    "month",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "month_sin",
    "month_cos",
]
DEFAULT_BASELINE_CATEGORICAL = ["product_id", "category"]


def _usable_categorical(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col]
    if float(s.notna().mean()) < 0.5:
        return False
    vc = s.astype(str).value_counts(dropna=True)
    if vc.empty or int(vc.size) <= 1:
        return False
    if float(vc.iloc[0] / max(1, len(s))) > 0.99:
        return False
    return True


def build_baseline_feature_matrix(panel_daily: pd.DataFrame) -> pd.DataFrame:
    out = panel_daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["product_id", "date"]).reset_index(drop=True)

    grp = out.groupby("product_id", dropna=False, group_keys=False)
    sales = pd.to_numeric(out.get("sales", 0.0), errors="coerce").fillna(0.0)
    out["sales"] = sales

    for lag in (1, 7, 14, 28):
        col = f"sales_lag{lag}"
        lagged = grp["sales"].shift(lag)
        out[col] = pd.to_numeric(lagged, errors="coerce").fillna(0.0)

    shifted = grp["sales"].shift(1)
    out["sales_ma7"] = shifted.groupby(out["product_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_ma14"] = shifted.groupby(out["product_id"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_ma28"] = shifted.groupby(out["product_id"]).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_std28"] = shifted.groupby(out["product_id"]).rolling(28, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)

    d = out["date"].dt
    out["dow"] = d.dayofweek.fillna(0).astype(int)
    out["week_of_year"] = d.isocalendar().week.astype("int64").fillna(1)
    out["month"] = d.month.fillna(1).astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    doy = d.dayofyear.fillna(1)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def derive_baseline_feature_spec(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = [c for c in DEFAULT_BASELINE_NUMERIC if c in df.columns]
    categorical = [c for c in DEFAULT_BASELINE_CATEGORICAL if _usable_categorical(df, c)]
    features = numeric + categorical
    return {
        "baseline_numeric_features": numeric,
        "baseline_categorical_features": categorical,
        "baseline_features": features,
        "baseline_context_features": [c for c in DEFAULT_BASELINE_CATEGORICAL if c in df.columns],
        "cat_features": categorical,
    }


def build_baseline_one_step_features(
    history_df: pd.DataFrame,
    current_date: pd.Timestamp,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> pd.DataFrame:
    h = history_df.copy().sort_values("date")
    sales = pd.to_numeric(h.get("sales", 0.0), errors="coerce").fillna(0.0)
    row: Dict[str, Any] = {"date": pd.Timestamp(current_date)}
    for c in feature_spec.get("baseline_context_features", []):
        row[c] = base_ctx.get(c, "unknown")

    for lag in (1, 7, 14, 28):
        row[f"sales_lag{lag}"] = float(sales.iloc[-lag]) if len(sales) >= lag else 0.0

    shifted = sales
    row["sales_ma7"] = float(shifted.tail(7).mean()) if len(shifted) else 0.0
    row["sales_ma14"] = float(shifted.tail(14).mean()) if len(shifted) else 0.0
    row["sales_ma28"] = float(shifted.tail(28).mean()) if len(shifted) else 0.0
    row["sales_std28"] = float(shifted.tail(28).std(ddof=0)) if len(shifted) else 0.0

    dt = pd.Timestamp(current_date)
    row["dow"] = int(dt.dayofweek)
    row["week_of_year"] = int(dt.isocalendar().week)
    row["month"] = int(dt.month)
    row["is_weekend"] = int(dt.dayofweek >= 5)
    row["sin_doy"] = float(np.sin(2 * np.pi * dt.dayofyear / 365.25))
    row["cos_doy"] = float(np.cos(2 * np.pi * dt.dayofyear / 365.25))
    row["month_sin"] = float(np.sin(2 * np.pi * dt.month / 12.0))
    row["month_cos"] = float(np.cos(2 * np.pi * dt.month / 12.0))

    return pd.DataFrame([row])
