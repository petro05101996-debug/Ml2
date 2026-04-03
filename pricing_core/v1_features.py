from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

V1_BASELINE_FEATURES = [
    "sales_lag1",
    "sales_lag7",
    "sales_lag28",
    "sales_ma7",
    "sales_ma14",
    "sales_ma28",
    "sales_trend_gap_7_28",
    "sales_std28",
    "sales_same_dow_ma8",
    "freight_value",
    "review_score",
    "promotion",
    "dow",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "month_sin",
    "month_cos",
    "time_index_norm",
]


def derive_v1_feature_spec(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "baseline_features": [c for c in V1_BASELINE_FEATURES if c in df.columns],
        "cat_features_baseline": [],
    }


def build_v1_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    out["sales"] = pd.to_numeric(out.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["price"] = pd.to_numeric(out.get("price", 0.0), errors="coerce").fillna(0.0).clip(lower=0.01)
    out["freight_value"] = pd.to_numeric(out.get("freight_value", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["review_score"] = pd.to_numeric(out.get("review_score", 4.5), errors="coerce").fillna(4.5)
    out["promotion"] = pd.to_numeric(out.get("promotion", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["discount"] = pd.to_numeric(out.get("discount", out.get("discount_rate", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0, upper=0.95)
    out["cost"] = pd.to_numeric(out.get("cost", out["price"] * 0.65), errors="coerce").fillna(out["price"] * 0.65).clip(lower=0.0)

    out["log_sales"] = np.log1p(out["sales"])
    out["log_price"] = np.log(out["price"].clip(lower=1e-9))
    out["sales_lag1"] = out["sales"].shift(1)
    out["sales_lag7"] = out["sales"].shift(7)
    out["sales_lag28"] = out["sales"].shift(28)
    out["sales_ma7"] = out["sales"].shift(1).rolling(7, min_periods=3).mean()
    out["sales_ma14"] = out["sales"].shift(1).rolling(14, min_periods=5).mean()
    out["sales_ma28"] = out["sales"].shift(1).rolling(28, min_periods=7).mean()
    out["sales_trend_gap_7_28"] = out["sales_ma7"] - out["sales_ma28"]
    out["sales_std28"] = out["sales"].shift(1).rolling(28, min_periods=7).std()

    out["dow"] = out["date"].dt.dayofweek
    out["sales_same_dow_ma8"] = (
        out.groupby("dow")["sales"]
        .transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    )
    out["is_weekend"] = (out["dow"] >= 5).astype(float)
    day_of_year = out["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["time_index_norm"] = np.linspace(0.0, 1.0, num=len(out)) if len(out) > 1 else 0.0

    prev_mean = out["sales"].shift(1).expanding(min_periods=1).mean()

    out["sales_lag1"] = out["sales_lag1"].fillna(0.0)
    out["sales_lag7"] = out["sales_lag7"].fillna(out["sales_lag1"]).fillna(0.0)
    out["sales_lag28"] = out["sales_lag28"].fillna(out["sales_lag7"]).fillna(out["sales_lag1"]).fillna(0.0)

    out["sales_ma7"] = out["sales_ma7"].fillna(prev_mean).fillna(0.0)
    out["sales_ma14"] = out["sales_ma14"].fillna(prev_mean).fillna(0.0)
    out["sales_ma28"] = out["sales_ma28"].fillna(prev_mean).fillna(0.0)
    out["sales_same_dow_ma8"] = out["sales_same_dow_ma8"].fillna(out["sales_ma7"]).fillna(prev_mean).fillna(0.0)
    out["sales_trend_gap_7_28"] = out["sales_trend_gap_7_28"].fillna(0.0)
    out["sales_std28"] = out["sales_std28"].fillna(0.0)

    for c in V1_BASELINE_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _get_sales(history_df: pd.DataFrame, lag: int) -> float:
    if len(history_df) < lag:
        return float(history_df["sales"].iloc[0]) if len(history_df) else 0.0
    return float(history_df["sales"].iloc[-lag])


def build_v1_one_step_features(
    history_df: pd.DataFrame,
    current_date: pd.Timestamp,
    base_ctx: Dict[str, Any],
    history_span_days: int,
) -> pd.DataFrame:
    hist = history_df.copy()
    sales = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0)
    ma7 = float(sales.tail(7).mean()) if len(sales) else 0.0
    ma14 = float(sales.tail(14).mean()) if len(sales) else ma7
    ma28 = float(sales.tail(28).mean()) if len(sales) else ma7
    trend_gap_7_28 = ma7 - ma28
    std28 = float(sales.tail(28).std(ddof=0)) if len(sales) > 1 else 0.0
    hist_dates = pd.to_datetime(hist["date"], errors="coerce")
    current_dow = int(current_date.dayofweek)
    same_dow_sales = sales[hist_dates.dt.dayofweek == current_dow]
    sales_same_dow_ma8 = float(same_dow_sales.tail(8).mean()) if len(same_dow_sales) else ma7

    day_of_year = int(current_date.dayofyear)
    month = int(current_date.month)
    denom = max(float(history_span_days), 1.0)

    row = {
        "sales_lag1": _get_sales(hist, 1),
        "sales_lag7": _get_sales(hist, 7),
        "sales_lag28": _get_sales(hist, 28),
        "sales_ma7": ma7,
        "sales_ma14": ma14,
        "sales_ma28": ma28,
        "sales_trend_gap_7_28": trend_gap_7_28,
        "sales_std28": std28,
        "sales_same_dow_ma8": sales_same_dow_ma8,
        "freight_value": float(base_ctx.get("freight_value", 0.0)),
        "review_score": float(base_ctx.get("review_score", 4.5)),
        "promotion": float(base_ctx.get("promotion", 0.0)),
        "dow": int(current_date.dayofweek),
        "is_weekend": float(1 if current_date.dayofweek >= 5 else 0),
        "sin_doy": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * day_of_year / 365.25)),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "time_index_norm": float(len(hist) / denom),
    }
    return pd.DataFrame([row])
