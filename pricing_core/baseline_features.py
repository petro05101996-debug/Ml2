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
    "week_of_month",
    "is_month_start",
    "is_month_end",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "month_sin",
    "month_cos",
    "lag7",
    "lag14",
    "lag28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_28",
    "sales_level_ratio_7_to_28",
    "weekday_profile_share",
]
DEFAULT_BASELINE_CATEGORICAL = ["series_id", "product_id", "category", "region", "channel", "segment"]


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
    if "series_id" not in out.columns:
        out["series_id"] = out.get("product_id", "unknown").astype(str)
    out = out.sort_values(["series_id", "date"]).reset_index(drop=True)

    grp = out.groupby("series_id", dropna=False, group_keys=False)
    sales = pd.to_numeric(out.get("sales", 0.0), errors="coerce").fillna(0.0)
    out["sales"] = sales

    for lag in (1, 7, 14, 28):
        col = f"sales_lag{lag}"
        lagged = grp["sales"].shift(lag)
        out[col] = pd.to_numeric(lagged, errors="coerce").fillna(0.0)

    shifted = grp["sales"].shift(1)
    out["sales_ma7"] = shifted.groupby(out["series_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_ma14"] = shifted.groupby(out["series_id"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_ma28"] = shifted.groupby(out["series_id"]).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_std28"] = shifted.groupby(out["series_id"]).rolling(28, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    out["lag7"] = out["sales_lag7"]
    out["lag14"] = out["sales_lag14"]
    out["lag28"] = out["sales_lag28"]
    out["rolling_mean_7"] = out["sales_ma7"]
    out["rolling_mean_14"] = out["sales_ma14"]
    out["rolling_mean_28"] = out["sales_ma28"]
    out["rolling_std_28"] = out["sales_std28"]
    out["sales_level_ratio_7_to_28"] = (
        pd.to_numeric(out["rolling_mean_7"], errors="coerce").fillna(0.0)
        / pd.to_numeric(out["rolling_mean_28"], errors="coerce").fillna(0.0).clip(lower=1e-6)
    )

    d = out["date"].dt
    out["dow"] = d.dayofweek.fillna(0).astype(int)
    out["week_of_year"] = d.isocalendar().week.astype("int64").fillna(1)
    out["week_of_month"] = ((d.day.fillna(1).astype(int) - 1) // 7 + 1).astype(int)
    out["month"] = d.month.fillna(1).astype(int)
    day = d.day.fillna(1).astype(int)
    dim = d.days_in_month.fillna(31).astype(int)
    out["is_month_start"] = (day <= 3).astype(int)
    out["is_month_end"] = ((dim - day) < 3).astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    doy = d.dayofyear.fillna(1)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["weekday_profile_share"] = 1.0 / 7.0
    for sid, idx in out.groupby("series_id", dropna=False).groups.items():
        g = out.loc[idx].copy().sort_values("date")
        shares = []
        for i in range(len(g)):
            hist = g.iloc[max(0, i - 56):i]
            if hist.empty:
                shares.append(1.0 / 7.0)
                continue
            total_hist = float(pd.to_numeric(hist["sales"], errors="coerce").fillna(0.0).sum())
            dow_hist = hist[hist["dow"] == int(g.iloc[i]["dow"])]
            dow_total = float(pd.to_numeric(dow_hist["sales"], errors="coerce").fillna(0.0).sum())
            shares.append(float(dow_total / max(total_hist, 1e-6)))
        out.loc[g.index, "weekday_profile_share"] = pd.Series(shares, index=g.index).fillna(1.0 / 7.0).clip(lower=0.0)
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
    row["lag7"] = row["sales_lag7"]
    row["lag14"] = row["sales_lag14"]
    row["lag28"] = row["sales_lag28"]
    row["rolling_mean_7"] = row["sales_ma7"]
    row["rolling_mean_14"] = row["sales_ma14"]
    row["rolling_mean_28"] = row["sales_ma28"]
    row["rolling_std_28"] = row["sales_std28"]
    row["sales_level_ratio_7_to_28"] = float(row["rolling_mean_7"]) / max(float(row["rolling_mean_28"]), 1e-6)

    dt = pd.Timestamp(current_date)
    row["dow"] = int(dt.dayofweek)
    row["week_of_year"] = int(dt.isocalendar().week)
    row["week_of_month"] = int((dt.day - 1) // 7 + 1)
    row["month"] = int(dt.month)
    row["is_month_start"] = int(dt.day <= 3)
    row["is_month_end"] = int((dt.days_in_month - dt.day) < 3)
    row["is_weekend"] = int(dt.dayofweek >= 5)
    row["sin_doy"] = float(np.sin(2 * np.pi * dt.dayofyear / 365.25))
    row["cos_doy"] = float(np.cos(2 * np.pi * dt.dayofyear / 365.25))
    row["month_sin"] = float(np.sin(2 * np.pi * dt.month / 12.0))
    row["month_cos"] = float(np.cos(2 * np.pi * dt.month / 12.0))
    last_56 = sales.tail(56).copy()
    if len(last_56):
        recent = history_df.copy().tail(56)
        recent["date"] = pd.to_datetime(recent["date"], errors="coerce")
        recent["sales"] = pd.to_numeric(recent.get("sales", 0.0), errors="coerce").fillna(0.0)
        total_recent = float(recent["sales"].sum())
        dow_recent = float(recent.loc[recent["date"].dt.dayofweek == int(dt.dayofweek), "sales"].sum())
        row["weekday_profile_share"] = float(dow_recent / max(total_recent, 1e-6))
    else:
        row["weekday_profile_share"] = 1.0 / 7.0

    return pd.DataFrame([row])
