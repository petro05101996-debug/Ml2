from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

USER_FACTOR_NUM_PREFIX = "user_factor_num__"
USER_FACTOR_CAT_PREFIX = "user_factor_cat__"
LAG_FEATURES = ["sales_lag1", "sales_lag7", "sales_lag28", "sales_ma7", "sales_ma14", "sales_ma28", "sales_trend_gap_7_28", "sales_std28"]
CAL_FEATURES = ["dow", "is_weekend", "sin_doy", "cos_doy", "month_sin", "month_cos", "time_index_norm"]
BUILTIN_NUMERIC = ["price", "discount", "promotion", "stock", "freight_value", "review_score", "reviews_count"]
BUILTIN_CATEGORICAL = ["product_id", "category", "region", "channel", "segment"]
SCENARIO_BASE = ["price", "discount", "promotion", "stock", "freight_value", "review_score", "reviews_count"]


def _freq_dominance(s: pd.Series) -> float:
    vv = s.dropna()
    if vv.empty:
        return 1.0
    return float(vv.value_counts(normalize=True).iloc[0])


def get_projection_safe_user_numeric_factors(df: pd.DataFrame) -> list[str]:
    blocked_contains = ["sales", "qty", "quantity", "revenue", "profit", "margin", "log_sales", "unit_price", "price_with_tax"]
    out: List[str] = []
    for c in [x for x in df.columns if str(x).startswith(USER_FACTOR_NUM_PREFIX)]:
        n = str(c).lower()
        if any(b in n for b in blocked_contains):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if float(s.notna().mean()) < 0.60:
            continue
        if int(s.nunique(dropna=True)) <= 1:
            continue
        if _freq_dominance(s) > 0.98:
            continue
        out.append(c)
    return sorted(out)


def get_projection_safe_user_categorical_factors(df: pd.DataFrame) -> list[str]:
    out: List[str] = []
    for c in [x for x in df.columns if str(x).startswith(USER_FACTOR_CAT_PREFIX)]:
        s = df[c].astype(str).replace("nan", np.nan)
        if s.dropna().empty:
            continue
        if int(s.nunique(dropna=True)) <= 1:
            continue
        if _freq_dominance(s) > 0.98:
            continue
        out.append(c)
    return sorted(out)


def build_v1_panel_feature_matrix(panel_daily: pd.DataFrame) -> pd.DataFrame:
    out = panel_daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["product_id", "date"]).reset_index(drop=True)
    out["sales"] = pd.to_numeric(out.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    for c in BUILTIN_NUMERIC + ["cost"]:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce")
    out["discount"] = pd.to_numeric(out.get("discount", out.get("discount_rate", 0.0)), errors="coerce").fillna(0.0).clip(0.0, 0.95)

    for c in BUILTIN_CATEGORICAL:
        out[c] = out.get(c, "unknown").fillna("unknown").astype(str)

    for c in [x for x in out.columns if str(x).startswith(USER_FACTOR_NUM_PREFIX)]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in [x for x in out.columns if str(x).startswith(USER_FACTOR_CAT_PREFIX)]:
        out[c] = out[c].fillna("unknown").astype(str)

    g = out.groupby("product_id", group_keys=False)
    out["sales_lag1"] = g["sales"].shift(1)
    out["sales_lag7"] = g["sales"].shift(7)
    out["sales_lag28"] = g["sales"].shift(28)
    out["sales_ma7"] = g["sales"].shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_ma14"] = g["sales"].shift(1).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_ma28"] = g["sales"].shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sales_trend_gap_7_28"] = out["sales_ma7"] - out["sales_ma28"]
    out["sales_std28"] = g["sales"].shift(1).rolling(28, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)

    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(float)
    day_of_year = out["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["time_index_norm"] = g.cumcount() / g["date"].transform("size").clip(lower=1)

    for c in LAG_FEATURES + CAL_FEATURES + BUILTIN_NUMERIC + [x for x in out.columns if x.startswith(USER_FACTOR_NUM_PREFIX)]:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce")
    return out


def build_v1_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    return build_v1_panel_feature_matrix(daily)


def derive_v1_feature_spec(df: pd.DataFrame) -> Dict[str, Any]:
    user_num = get_projection_safe_user_numeric_factors(df)
    user_cat = get_projection_safe_user_categorical_factors(df)
    numeric_demand = LAG_FEATURES + CAL_FEATURES + BUILTIN_NUMERIC + user_num
    categorical_demand = [c for c in BUILTIN_CATEGORICAL + user_cat if c in df.columns]
    scenario_features = [c for c in SCENARIO_BASE if c in df.columns] + [c for c in user_num if c in df.columns]
    demand_features = numeric_demand + categorical_demand
    return {
        "demand_features": demand_features,
        "numeric_demand_features": numeric_demand,
        "categorical_demand_features": categorical_demand,
        "scenario_features": scenario_features,
        "user_numeric_features": user_num,
        "user_categorical_features": user_cat,
        "context_passthrough_features": ["cost", "product_id", "category", "region", "channel", "segment"],
        "cat_features_demand": categorical_demand,
    }


def _get_sales(history_df: pd.DataFrame, lag: int) -> float:
    if len(history_df) < lag:
        return float(history_df["sales"].iloc[0]) if len(history_df) else 0.0
    return float(history_df["sales"].iloc[-lag])


def build_v1_one_step_features(
    history_df: pd.DataFrame,
    current_date: pd.Timestamp,
    base_ctx: Dict[str, Any],
    history_span_days: int,
    feature_spec: Dict[str, Any],
    scenario_overrides: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    hist = history_df.copy()
    sales = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0)
    ma7 = float(sales.tail(7).mean()) if len(sales) else 0.0
    ma14 = float(sales.tail(14).mean()) if len(sales) else ma7
    ma28 = float(sales.tail(28).mean()) if len(sales) else ma7

    row: Dict[str, Any] = {
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
    merged_ctx = dict(base_ctx)
    merged_ctx.update(scenario_overrides or {})

    for c in feature_spec.get("scenario_features", []):
        row[c] = float(pd.to_numeric(pd.Series([merged_ctx.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0])

    for c in feature_spec.get("categorical_demand_features", []):
        row[c] = str(merged_ctx.get(c, "unknown"))

    out = {c: row.get(c, "unknown" if c in feature_spec.get("cat_features_demand", []) else 0.0) for c in feature_spec.get("demand_features", [])}
    return pd.DataFrame([out])
