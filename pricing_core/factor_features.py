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
    base_num = [
        "price",
        "discount",
        "promotion",
        "price_rel_to_recent_median_28",
        "discount_rate",
        "promo_flag",
        "dow",
        "week_of_month",
        "month",
        "is_month_start",
        "is_month_end",
        "recent_sales_level_7",
        "recent_sales_level_28",
        "sales_level_ratio_7_to_28",
        "weekday_profile_share",
        "days_since_last_promo",
        "price_rank_vs_last_8_weeks",
    ]
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
    dts = pd.to_datetime(out["date"], errors="coerce")
    out["dow"] = dts.dt.dayofweek.fillna(0).astype(int)
    out["week_of_month"] = ((dts.dt.day.fillna(1).astype(int) - 1) // 7 + 1).astype(int)
    out["month"] = dts.dt.month.fillna(1).astype(int)
    out["is_month_start"] = dts.dt.is_month_start.fillna(False).astype(float)
    out["is_month_end"] = dts.dt.is_month_end.fillna(False).astype(float)
    grp = out.groupby("series_id", dropna=False)
    out["recent_sales_level_7"] = grp["sales"].shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0)
    out["recent_sales_level_28"] = grp["sales"].shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0)
    out["sales_level_ratio_7_to_28"] = out["recent_sales_level_7"] / out["recent_sales_level_28"].replace(0.0, np.nan)
    out["sales_level_ratio_7_to_28"] = out["sales_level_ratio_7_to_28"].fillna(1.0)
    out["weekday_profile_share"] = 1.0 / 7.0
    for sid, g in out.groupby("series_id", dropna=False):
        sales_by_dow = pd.to_numeric(g["sales"], errors="coerce").fillna(0.0).groupby(g["dow"]).sum().reindex(range(7), fill_value=0.0)
        if float(sales_by_dow.sum()) > 1e-9:
            sales_by_dow = sales_by_dow / float(sales_by_dow.sum())
        else:
            sales_by_dow = pd.Series([1.0 / 7.0] * 7, index=range(7), dtype=float)
        out.loc[g.index, "weekday_profile_share"] = g["dow"].map(sales_by_dow).fillna(1.0 / 7.0).values
    promo_date = dts.where(pd.to_numeric(out.get("promotion", 0.0), errors="coerce").fillna(0.0) > 0)
    out["_last_promo_date"] = promo_date.groupby(out["series_id"]).ffill()
    out["days_since_last_promo"] = (dts - out["_last_promo_date"]).dt.days.fillna(999.0)
    out.drop(columns=["_last_promo_date"], inplace=True)
    out["days_since_last_promo"] = pd.to_numeric(out["days_since_last_promo"], errors="coerce").fillna(999.0)
    out["price_rank_vs_last_8_weeks"] = grp["price"].transform(lambda s: s.rolling(56, min_periods=1).apply(lambda w: float((w <= w.iloc[-1]).mean()), raw=False)).fillna(0.5)
    for c in feature_spec.get("factor_features", []):
        if c not in out.columns:
            out[c] = 0.0 if c in feature_spec.get("factor_numeric_features", []) else "unknown"
    return out


def build_factor_target(df: pd.DataFrame) -> pd.Series:
    frame = build_factor_target_frame(df)
    y = frame["factor_target"].copy()
    y[~frame["factor_target_valid"].astype(bool)] = np.nan
    return y


def build_factor_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    sales = pd.to_numeric(df.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    baseline_oof = pd.to_numeric(df.get("baseline_oof", pd.NA), errors="coerce").clip(lower=0.0)
    baseline_abs_error = (sales - baseline_oof).abs()
    baseline_rel_error = baseline_abs_error / (sales.abs() + 1.0)
    baseline_ratio = (baseline_oof + 1.0) / (sales + 1.0)
    y = np.log((sales + 1.0) / (baseline_oof + 1.0)).clip(-1.2, 1.2)

    valid = (
        baseline_oof.notna()
        & (baseline_rel_error <= 0.60)
        & (baseline_ratio >= 0.5)
        & (baseline_ratio <= 1.8)
    )
    if "stockout_flag" in df.columns:
        mask = pd.to_numeric(df["stockout_flag"], errors="coerce").fillna(0.0) > 0.0
        valid = valid & (~mask)
    return pd.DataFrame(
        {
            "factor_target": pd.Series(y, index=df.index),
            "factor_target_valid": pd.Series(valid, index=df.index).astype(bool),
            "baseline_rel_error": pd.Series(baseline_rel_error, index=df.index),
            "baseline_ratio": pd.Series(baseline_ratio, index=df.index),
        },
        index=df.index,
    )
