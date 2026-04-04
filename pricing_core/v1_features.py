from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

USER_FACTOR_PREFIX = "user_factor__"
LAG_FEATURES = ["sales_lag1", "sales_lag7", "sales_lag28", "sales_ma7", "sales_ma14", "sales_ma28", "sales_trend_gap_7_28", "sales_std28"]
CAL_FEATURES = ["dow", "is_weekend", "sin_doy", "cos_doy", "month_sin", "month_cos", "time_index_norm"]
BUILTIN_DEMAND = ["price", "discount", "promotion", "stock", "freight_value", "review_score", "reviews_count"]
SCENARIO_BASE = ["price", "discount", "cost", "promotion", "stock", "freight_value", "review_score", "reviews_count"]


def get_projection_safe_user_factors(df: pd.DataFrame) -> list[str]:
    blocked_exact = {
        "user_factor__sales", "user_factor__quantity", "user_factor__qty", "user_factor__revenue", "user_factor__profit",
        "user_factor__margin", "user_factor__log_sales", "user_factor__unit_price", "user_factor__price_with_tax",
    }
    blocked_contains = ["sales", "qty", "quantity", "revenue", "profit", "margin"]
    out: List[str] = []
    for c in [x for x in df.columns if str(x).startswith(USER_FACTOR_PREFIX)]:
        name = str(c).lower()
        if name in blocked_exact or any(t in name for t in blocked_contains):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if float(s.notna().mean()) < 0.60:
            continue
        nz = s[s.notna() & (s != 0)]
        if int(nz.nunique()) < 2:
            continue
        if int(s.nunique(dropna=True)) <= 1:
            continue
        out.append(c)
    return sorted(out)


def get_projection_safe_factor_columns(df: pd.DataFrame) -> list[str]:
    return get_projection_safe_user_factors(df)


def get_v1_demand_features(df: pd.DataFrame) -> list[str]:
    return LAG_FEATURES + CAL_FEATURES + BUILTIN_DEMAND + get_projection_safe_factor_columns(df)


def derive_v1_feature_spec(df: pd.DataFrame) -> Dict[str, Any]:
    user_factors = get_projection_safe_factor_columns(df)
    return {
        "demand_features": get_v1_demand_features(df),
        "scenario_features": SCENARIO_BASE + user_factors,
        "user_factor_features": user_factors,
        "context_passthrough_features": ["cost", "category", "product_id"],
        "cat_features_demand": [],
    }


def _get_sales(history_df: pd.DataFrame, lag: int) -> float:
    if len(history_df) < lag:
        return float(history_df["sales"].iloc[0]) if len(history_df) else 0.0
    return float(history_df["sales"].iloc[-lag])


def build_v1_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)

    out["sales"] = pd.to_numeric(out.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["price"] = pd.to_numeric(out.get("price", 0.0), errors="coerce").fillna(0.0).clip(lower=0.01)
    out["discount"] = pd.to_numeric(out.get("discount", out.get("discount_rate", 0.0)), errors="coerce").fillna(0.0).clip(0.0, 0.95)
    out["cost"] = pd.to_numeric(out.get("cost", out["price"] * 0.65), errors="coerce").fillna(out["price"] * 0.65).clip(lower=0.0)
    out["promotion"] = pd.to_numeric(out.get("promotion", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["stock"] = pd.to_numeric(out.get("stock", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["freight_value"] = pd.to_numeric(out.get("freight_value", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["review_score"] = pd.to_numeric(out.get("review_score", out.get("rating", 4.5)), errors="coerce").fillna(4.5)
    out["reviews_count"] = pd.to_numeric(out.get("reviews_count", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    user_cols = [c for c in out.columns if str(c).startswith(USER_FACTOR_PREFIX)]
    for c in user_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        med = s.median()
        out[c] = s.fillna(0.0 if pd.isna(med) else float(med))

    out["sales_lag1"] = out["sales"].shift(1)
    out["sales_lag7"] = out["sales"].shift(7)
    out["sales_lag28"] = out["sales"].shift(28)
    out["sales_ma7"] = out["sales"].shift(1).rolling(7, min_periods=1).mean()
    out["sales_ma14"] = out["sales"].shift(1).rolling(14, min_periods=1).mean()
    out["sales_ma28"] = out["sales"].shift(1).rolling(28, min_periods=1).mean()
    out["sales_trend_gap_7_28"] = out["sales_ma7"] - out["sales_ma28"]
    out["sales_std28"] = out["sales"].shift(1).rolling(28, min_periods=1).std().fillna(0.0)

    out["dow"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(float)
    day_of_year = out["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["time_index_norm"] = np.linspace(0.0, 1.0, len(out)) if len(out) > 1 else 0.0

    for c in LAG_FEATURES + CAL_FEATURES + BUILTIN_DEMAND + user_cols:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce").fillna(0.0)
    return out


def build_v1_one_step_features(history_df: pd.DataFrame, current_date: pd.Timestamp, base_ctx: Dict[str, Any], history_span_days: int, feature_spec: Dict[str, Any]) -> pd.DataFrame:
    hist = history_df.copy()
    sales = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0)
    ma7 = float(sales.tail(7).mean()) if len(sales) else 0.0
    ma14 = float(sales.tail(14).mean()) if len(sales) else ma7
    ma28 = float(sales.tail(28).mean()) if len(sales) else ma7
    row: Dict[str, float] = {
        "sales_lag1": _get_sales(hist, 1),
        "sales_lag7": _get_sales(hist, 7),
        "sales_lag28": _get_sales(hist, 28),
        "sales_ma7": ma7,
        "sales_ma14": ma14,
        "sales_ma28": ma28,
        "sales_trend_gap_7_28": ma7 - ma28,
        "sales_std28": float(sales.tail(28).std(ddof=0)) if len(sales) > 1 else 0.0,
        "dow": int(current_date.dayofweek),
        "is_weekend": float(1 if current_date.dayofweek >= 5 else 0),
        "sin_doy": float(np.sin(2 * np.pi * int(current_date.dayofyear) / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * int(current_date.dayofyear) / 365.25)),
        "month_sin": float(np.sin(2 * np.pi * int(current_date.month) / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * int(current_date.month) / 12.0)),
        "time_index_norm": float(len(hist) / max(float(history_span_days), 1.0)),
    }
    for c in feature_spec.get("scenario_features", []):
        row[c] = float(base_ctx.get(c, 0.0))
    out = {c: float(row.get(c, 0.0)) for c in feature_spec.get("demand_features", [])}
    return pd.DataFrame([out])
