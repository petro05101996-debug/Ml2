from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from df_utils import get_numeric_series


def _usable_cat(df: pd.DataFrame, c: str) -> bool:
    if c not in df.columns:
        return False
    s = df[c]
    if float(s.notna().mean()) < 0.5:
        return False
    vc = s.astype(str).value_counts(dropna=True)
    return (not vc.empty) and vc.size > 1 and (vc.iloc[0] / max(1, len(s)) <= 0.99)


def _usable_numeric(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    if float(s.notna().mean()) < 0.6:
        return False
    if int(s.nunique(dropna=True)) <= 1:
        return False
    vc = s.dropna().value_counts(normalize=True)
    if not vc.empty and float(vc.iloc[0]) > 0.98:
        return False
    non_zero = s.fillna(0.0).abs() > 1e-12
    if float(non_zero.mean()) < 0.02:
        return False
    return True


def derive_factor_feature_spec(df: pd.DataFrame) -> Dict[str, Any]:
    base_num = ["price", "discount", "promotion", "price_rel_to_recent_median_28", "discount_rate", "promo_flag"]
    user_num = [c for c in df.columns if c.startswith("user_factor_num__") and _usable_numeric(df, c)]
    factor_numeric = base_num + user_num

    base_cat = ["series_id", "product_id", "category", "region", "channel", "segment"]
    user_cat = [c for c in df.columns if c.startswith("user_factor_cat__") and _usable_cat(df, c)]
    factor_cat = [c for c in base_cat if _usable_cat(df, c)] + user_cat

    interactions = ["price_rel_to_recent_median_28_x_promo_flag", "price_rel_to_recent_median_28_x_is_weekend"]
    features = factor_numeric + interactions + factor_cat
    controllable = [c for c in ["price", "discount", "promotion"] + user_num + user_cat if c in df.columns or c in ["price", "discount", "promotion"]]
    return {
        "factor_numeric_features": factor_numeric,
        "factor_categorical_features": factor_cat,
        "factor_features": features,
        "controllable_features": controllable,
        "context_features": [c for c in base_cat if c in df.columns],
        "interaction_features": interactions,
    }


def build_factor_feature_matrix(panel_with_baseline: pd.DataFrame, feature_spec: Dict[str, Any]) -> pd.DataFrame:
    out = panel_with_baseline.copy()
    if "series_id" not in out.columns:
        out["series_id"] = out.get("product_id", "unknown").astype(str)
    out = out.sort_values(["series_id", "date"]).reset_index(drop=True)
    out["price"] = get_numeric_series(out, "price", 0.0)
    rolling_median = out.groupby("series_id", dropna=False)["price"].shift(1).groupby(out["series_id"]).rolling(28, min_periods=1).median().reset_index(level=0, drop=True)
    price_hist_med = float(out["price"].replace(0, pd.NA).dropna().median()) if out["price"].replace(0, pd.NA).dropna().notna().any() else 1.0
    denom = rolling_median.replace(0, pd.NA).fillna(price_hist_med if pd.notna(price_hist_med) and price_hist_med > 0 else 1.0)
    out["price_rel_to_recent_median_28"] = (out["price"].fillna(0.0) / denom) - 1.0
    out["discount_rate"] = get_numeric_series(out, "discount", 0.0).fillna(0.0)
    out["promo_flag"] = (get_numeric_series(out, "promotion", 0.0).fillna(0.0) > 0).astype(float)
    out["is_weekend"] = pd.to_datetime(out["date"]).dt.dayofweek.ge(5).astype(float)
    out["price_rel_to_recent_median_28_x_promo_flag"] = out["price_rel_to_recent_median_28"] * out["promo_flag"]
    out["price_rel_to_recent_median_28_x_is_weekend"] = out["price_rel_to_recent_median_28"] * out["is_weekend"]
    for c in feature_spec.get("factor_features", []):
        if c not in out.columns:
            out[c] = 0.0 if c in feature_spec.get("factor_numeric_features", []) else "unknown"
    return out


def build_factor_target(df: pd.DataFrame) -> pd.Series:
    sales = pd.to_numeric(df.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    baseline_oof = pd.to_numeric(df.get("baseline_oof", pd.NA), errors="coerce").clip(lower=0.0)
    y = np.log((sales + 1.0) / (baseline_oof + 1.0))
    if "stockout_flag" in df.columns:
        mask = pd.to_numeric(df["stockout_flag"], errors="coerce").fillna(0.0) > 0.0
        y = pd.Series(y, index=df.index)
        y.loc[mask] = np.nan
        return y
    return y
