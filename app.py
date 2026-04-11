from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_adapter import (
    build_auto_mapping,
    build_daily_from_transactions,
    normalize_transactions,
)
from data_schema import CANONICAL_FIELDS
from what_if import build_sensitivity_grid, run_scenario_set

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

np.random.seed(42)

try:
    import statsmodels.api as sm
    from statsmodels.tools.tools import add_constant
    USE_STATSMODELS = True
except Exception:
    sm = None
    add_constant = None
    USE_STATSMODELS = False

try:
    import catboost
    CatBoostRegressor = catboost.CatBoostRegressor
    USE_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    USE_CATBOOST = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_HGB = True
except Exception:
    HistGradientBoostingRegressor = None
    USE_HGB = False


CONFIG = {
    "MIN_COVERAGE_DAYS": 80,
    "MIN_TOTAL_SALES": 100,
    "MIN_CATEGORY_ROWS": 120,
    "MIN_UNIQUE_PRICES": 6,
    "MIN_PRICE_CHANGES": 5,
    "MIN_REL_PRICE_SPAN": 0.08,
    "TRAIN_FRACTION": 0.70,
    "VAL_FRACTION": 0.15,
    "ENSEMBLE_SIZE": 5,
    "CAT_ITER": 700,
    "CAT_LR": 0.04,
    "CAT_DEPTH": 6,
    "HGB_MAX_ITER": 450,
    "HGB_LR": 0.05,
    "HGB_DEPTH": 8,
    "HGB_MIN_LEAF": 20,
    "RF_TREES": 250,
    "RF_DEPTH": 12,
    "HORIZON_DAYS_DEFAULT": 30,
    "MIN_REL_STEP": -0.20,
    "MAX_REL_STEP": 0.12,
    "SEARCH_MARGIN": 0.25,
    "PRICE_CHANGE_PENALTY_SCALE": 0.015,
    "QUAD_PENALTY_COEF": 0.006,
    "UNCERTAINTY_MULTIPLIER": 0.75,
    "DISAGREEMENT_PENALTY_SCALE": 0.02,
    "MIN_PROFIT_REL_IMPROV": 0.015,
    "ABSOLUTE_MIN_PROFIT_IMPROV": 1500.0,
    "PRIOR_ELASTICITY": -1.10,
    "ELASTICITY_FLOOR": -5.0,
    "ELASTICITY_CEILING": -0.08,
    "TAU2_AUTO_TUNE": True,
    "COST_PROXY_RATIO": 0.65,
    "PRICING_DIRECT_WEIGHT_CAP": 0.85,
    "PRICING_DIRECT_WEIGHT_FLOOR": 0.65,
    "MIN_ML_SALES": 120,
    "FORCE_ENHANCED_MODE": True,
    "SMALL_CAT_L2": 12.0,
    "SMALL_CAT_DEPTH": 4,
    "SMALL_CAT_ITER": 350,
    "SMALL_HGB_L2": 15.0,
    "SMALL_RF_MAX_FEATURES": 0.55,
    "AUGMENT_N": 8,
    "AUGMENT_PRICE_NOISE": 0.05,
}

FEATURE_STATS: Dict[str, float] = {}


def robust_clean_dirty_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sales", "price", "freight_value"]:
        if col in df.columns and len(df) > 0:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    if "sales" in df.columns and len(df) > 0 and (df["sales"] == 0).mean() > 0.7:
        df = df[df["sales"] > 0].copy().reset_index(drop=True)
    return df


def _recompute_augmented_derived_cols(df_aug: pd.DataFrame) -> pd.DataFrame:
    out = df_aug.copy()
    if "sales" in out.columns:
        out["log_sales"] = np.log1p(out["sales"].clip(lower=0.0))
    if "price" in out.columns:
        out["price"] = out["price"].clip(lower=0.01)
        out["log_price"] = np.log(out["price"])
    if "freight_value" in out.columns:
        out["log_freight"] = np.log1p(out["freight_value"].clip(lower=0.0))

    if "price" in out.columns:
        if "price_change_1d" in out.columns:
            out["price_change_1d"] = out["price"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if "price_ma7" in out.columns:
            out["price_vs_ma7"] = out["price"] / out["price_ma7"].replace(0, np.nan)
        if "price_ma28" in out.columns:
            out["price_vs_ma28"] = out["price"] / out["price_ma28"].replace(0, np.nan)
        if "price_ma90" in out.columns:
            out["price_vs_ma90"] = out["price"] / out["price_ma90"].replace(0, np.nan)
        if "price_median" in out.columns:
            out["price_vs_cat_median"] = out["price"] / out["price_median"].replace(0, np.nan)
        if "freight_value" in out.columns:
            out["freight_to_price"] = out["freight_value"] / out["price"].replace(0, np.nan)

    for c in [
        "price_vs_ma7",
        "price_vs_ma28",
        "price_vs_ma90",
        "price_vs_cat_median",
        "freight_to_price",
        "sales_momentum_7_28",
        "sales_momentum_28_90",
    ]:
        if c in out.columns:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    can_rebuild_lags = all(col in out.columns for col in ["date", "sales", "price", "freight_value"])
    if can_rebuild_lags:
        lag_seed = out[["date", "sales", "price", "freight_value"]].copy()
        lag_seed["date"] = pd.to_datetime(lag_seed["date"], errors="coerce")
        if "review_score" in out.columns:
            lag_seed["review_score"] = pd.to_numeric(out["review_score"], errors="coerce")
        else:
            lag_seed["review_score"] = 4.5
        lag_seed["review_score"] = lag_seed["review_score"].fillna(safe_median(lag_seed["review_score"], 4.5))
        lag_seed = lag_seed.sort_values("date").reset_index(drop=True)
        lag_df = add_leak_free_lag_features(lag_seed)
        lag_cols = [c for c in lag_df.columns if c in out.columns]
        if lag_cols:
            out.loc[lag_df.index, lag_cols] = lag_df[lag_cols].values

    return out


def augment_price_variations(df: pd.DataFrame, n_aug: int = CONFIG["AUGMENT_N"]) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    aug_frames = [df.copy()]
    orig_price = df["price"].copy()
    for _ in range(n_aug):
        df_aug = df.copy()
        noise = np.random.normal(0, CONFIG["AUGMENT_PRICE_NOISE"], len(df_aug))
        df_aug["price"] = orig_price * (1 + noise)
        df_aug["price"] = df_aug["price"].clip(lower=0.01)
        price_ratio = df_aug["price"] / orig_price.clip(lower=0.01)
        if "sales" in df_aug.columns:
            df_aug["sales"] = (df_aug["sales"] * (price_ratio ** CONFIG["PRIOR_ELASTICITY"])).clip(lower=0.1)
        df_aug = _recompute_augmented_derived_cols(df_aug)
        aug_frames.append(df_aug)
    return pd.concat(aug_frames, ignore_index=True)


def safe_median(series: pd.Series, default: float = 0.0) -> float:
    try:
        x = float(pd.Series(series).median())
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def calculate_mape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > eps)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def calculate_smape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return float("nan")
    denom = np.maximum(np.abs(yt[mask]) + np.abs(yp[mask]), eps)
    return float(np.mean(2.0 * np.abs(yp[mask] - yt[mask]) / denom) * 100.0)


def calculate_wape(y_true, y_pred, eps: float = 1e-9) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return float("nan")
    denom = np.sum(np.abs(yt[mask]))
    return float(np.sum(np.abs(yt[mask] - yp[mask])) / max(denom, eps) * 100.0)


def is_price_plausible(price: float, train_min: float, train_max: float, margin: float = 0.25) -> bool:
    if not np.isfinite(price):
        return False
    if train_min >= train_max:
        return True
    span = train_max - train_min
    allowed_min = max(0.01, train_min - span * margin)
    allowed_max = train_max + span * margin
    return allowed_min <= price <= allowed_max


def validate_schema(df: pd.DataFrame, required: Iterable[str], name: str = "dataframe") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def _suggest_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    if not columns:
        return None
    norm_to_orig = {_norm_col(c): c for c in columns}
    for a in aliases:
        if a in norm_to_orig:
            return norm_to_orig[a]
    for c in columns:
        cn = _norm_col(c)
        if any(a in cn for a in aliases):
            return c
    return None


def _rename_with_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    rename_map = {src: dst for dst, src in mapping.items() if src is not None and src in out.columns}
    out = out.rename(columns=rename_map)
    return out


def fit_feature_stats(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for c in features:
        stats[c] = safe_median(df[c], 0.0) if c in df.columns and len(df) > 0 else 0.0
    return stats


def _safe_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _build_dataset_passport(txn: pd.DataFrame) -> Dict[str, Any]:
    fields = ["date", "product_id", "category", "quantity", "price", "revenue", "cost", "discount", "promotion", "freight_value", "stock", "rating", "reviews_count", "region", "channel", "segment"]
    date_min = str(pd.to_datetime(txn["date"], errors="coerce").min()) if "date" in txn.columns else None
    date_max = str(pd.to_datetime(txn["date"], errors="coerce").max()) if "date" in txn.columns else None
    history_by_sku = []
    if {"product_id", "date"}.issubset(txn.columns):
        history_df = txn.groupby("product_id")["date"].agg(["min", "max"]).reset_index()
        history_df["history_days"] = (pd.to_datetime(history_df["max"]) - pd.to_datetime(history_df["min"])).dt.days + 1
        history_df["min"] = pd.to_datetime(history_df["min"], errors="coerce").dt.strftime("%Y-%m-%d")
        history_df["max"] = pd.to_datetime(history_df["max"], errors="coerce").dt.strftime("%Y-%m-%d")
        history_by_sku = history_df.to_dict("records")
    field_stats = {}
    for f in fields:
        if f not in txn.columns:
            field_stats[f] = {"found": False}
            continue
        s = pd.to_numeric(txn[f], errors="coerce") if f not in {"date", "product_id", "category", "region", "channel", "segment"} else txn[f]
        numeric = pd.to_numeric(txn[f], errors="coerce")
        field_stats[f] = {
            "found": True,
            "missing_share": float(txn[f].isna().mean()),
            "n_unique": int(txn[f].nunique(dropna=True)),
            "std": float(numeric.std()) if numeric.notna().any() else None,
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "median": float(numeric.median()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
            "zero_share": float((numeric.fillna(0.0) == 0).mean()) if numeric.notna().any() else None,
        }
    return {
        "rows": int(len(txn)),
        "unique_sku": int(txn["product_id"].nunique()) if "product_id" in txn.columns else 0,
        "date_min": date_min,
        "date_max": date_max,
        "history_by_sku": history_by_sku,
        "fields": field_stats,
    }


def _tail_mean(arr: np.ndarray, k: int, default: float = 0.0) -> float:
    if len(arr) == 0:
        return default
    return float(np.mean(arr[-min(k, len(arr)) :]))


def _tail_std(arr: np.ndarray, k: int, default: float = 0.0) -> float:
    if len(arr) == 0:
        return default
    return float(np.std(arr[-min(k, len(arr)) :], ddof=0))


def build_raw_frame(orders: pd.DataFrame, order_items: pd.DataFrame, products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    order_items = order_items.copy()
    products = products.copy()
    reviews = reviews.copy()

    validate_schema(orders, ["order_id", "order_purchase_timestamp"], "orders")
    validate_schema(order_items, ["order_id", "product_id"], "order_items")
    validate_schema(products, ["product_id", "product_category_name"], "products")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    if not reviews.empty and "review_creation_date" in reviews.columns:
        reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")

    raw_df = order_items.merge(orders, on="order_id", how="inner")
    raw_df = raw_df.merge(products, on="product_id", how="left")

    if not reviews.empty and "order_id" in reviews.columns and "review_score" in reviews.columns:
        raw_df = raw_df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
    else:
        raw_df["review_score"] = np.nan

    needed = ["order_purchase_timestamp", "product_category_name", "price"]
    raw_df = raw_df.dropna(subset=[c for c in needed if c in raw_df.columns]).copy()
    raw_df["date"] = pd.to_datetime(raw_df["order_purchase_timestamp"], errors="coerce").dt.normalize()
    raw_df = raw_df.dropna(subset=["date"]).sort_values(["product_category_name", "date"]).reset_index(drop=True)
    return raw_df


def build_daily_sku_frame(sku_df: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    validate_schema(sku_df, ["date", "order_item_id", "order_id", "product_id", "price"], "sku_df")
    if len(sku_df) == 0:
        raise ValueError("Нет данных по выбранному SKU.")
    daily_agg = {
        "sales": ("order_item_id", "count"),
        "revenue": ("price", "sum"),
        "price": ("price", "mean"),
        "price_median": ("price", "median"),
        "orders": ("order_id", "nunique"),
        "products_sold": ("product_id", "nunique"),
    }
    if "freight_value" in sku_df.columns:
        daily_agg["freight_value"] = ("freight_value", "mean")
    else:
        daily_agg["freight_value"] = ("price", "mean")
    if "review_score" in sku_df.columns:
        daily_agg["review_score"] = ("review_score", "mean")

    daily = sku_df.groupby("date").agg(**daily_agg).reset_index()
    full_dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full_dates.merge(daily, on="date", how="left").sort_values("date").reset_index(drop=True)

    for col in ["sales", "revenue", "orders", "products_sold"]:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0).astype(float)

    for col in ["price", "price_median", "freight_value", "review_score"]:
        if col not in daily.columns:
            daily[col] = np.nan
        daily[col] = daily[col].ffill().bfill()

    daily["price"] = daily["price"].fillna(safe_median(daily["price"], 1.0)).clip(lower=0.01)
    daily["price_median"] = daily["price_median"].fillna(daily["price"]).clip(lower=0.01)
    daily["freight_value"] = daily["freight_value"].fillna(safe_median(daily["freight_value"], 0.0)).clip(lower=0.0)
    daily["review_score"] = 4.5 if daily["review_score"].isna().all() else daily["review_score"].fillna(safe_median(daily["review_score"], 4.5))
    daily["sku_id"] = str(sku_id)
    daily["category"] = str(sku_df["product_category_name"].mode().iloc[0]) if "product_category_name" in sku_df.columns else "unknown"
    return daily


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["dayofyear"] = out["date"].dt.dayofyear.astype(int)
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["time_index"] = np.arange(len(out), dtype=float)
    out["time_index_norm"] = out["time_index"] / max(1.0, float(len(out) - 1))
    return out


def add_leak_free_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["sales"] = out["sales"].astype(float).fillna(0.0)
    out["price"] = out["price"].astype(float).clip(lower=0.01)
    if "price_median" not in out.columns:
        out["price_median"] = out["price"]
    out["price_median"] = out["price_median"].astype(float).fillna(out["price"]).clip(lower=0.01)
    out["freight_value"] = out["freight_value"].astype(float).clip(lower=0.0)
    out["review_score"] = out["review_score"].astype(float).fillna(safe_median(out["review_score"], 4.5))
    if "discount" not in out.columns:
        out["discount"] = 0.0
    if "promotion" not in out.columns:
        out["promotion"] = 0.0
    if "reviews_count" not in out.columns:
        out["reviews_count"] = 0.0
    if "net_unit_price" not in out.columns:
        out["net_unit_price"] = out["price"] * (1.0 - out["discount"].astype(float).fillna(0.0))
    out["discount"] = out["discount"].astype(float).fillna(0.0).clip(lower=0.0, upper=0.95)
    out["promotion"] = out["promotion"].astype(float).fillna(0.0).clip(lower=0.0, upper=1.0)
    out["reviews_count"] = out["reviews_count"].astype(float).fillna(0.0).clip(lower=0.0)
    out["net_unit_price"] = out["net_unit_price"].astype(float).fillna(out["price"] * (1.0 - out["discount"])).clip(lower=0.01)

    out["log_sales"] = np.log1p(out["sales"])
    out["log_price"] = np.log(out["net_unit_price"].clip(lower=0.01))
    out["log_freight"] = np.log1p(out["freight_value"].clip(lower=0.0))

    effective_price = out["net_unit_price"].astype(float).clip(lower=0.01)
    out["price_lag1"] = effective_price.shift(1)
    out["price_lag7"] = effective_price.shift(7)
    out["price_lag28"] = effective_price.shift(28)

    out["sales_lag1"] = out["sales"].shift(1)
    out["sales_lag7"] = out["sales"].shift(7)
    out["sales_lag14"] = out["sales"].shift(14)
    out["sales_lag28"] = out["sales"].shift(28)

    out["freight_lag1"] = out["freight_value"].shift(1)
    out["freight_lag7"] = out["freight_value"].shift(7)
    out["freight_lag28"] = out["freight_value"].shift(28)

    for win in [7, 28, 90]:
        minp = max(3, min(7, win))
        out[f"sales_ma{win}"] = out["sales"].shift(1).rolling(win, min_periods=minp).mean()
        out[f"price_ma{win}"] = effective_price.shift(1).rolling(win, min_periods=minp).mean()
        out[f"freight_ma{win}"] = out["freight_value"].shift(1).rolling(win, min_periods=minp).mean()

    out["sales_std28"] = out["sales"].shift(1).rolling(28, min_periods=7).std()
    out["price_change_1d"] = effective_price.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["price_vs_ma7"] = effective_price / out["price_ma7"].replace(0, np.nan)
    out["price_vs_ma28"] = effective_price / out["price_ma28"].replace(0, np.nan)
    out["price_vs_ma90"] = effective_price / out["price_ma90"].replace(0, np.nan)
    out["price_vs_cat_median"] = effective_price / effective_price.replace(0, np.nan).median()
    out["freight_to_price"] = out["freight_value"] / effective_price.replace(0, np.nan)
    out["sales_momentum_7_28"] = out["sales_ma7"] / out["sales_ma28"].replace(0, np.nan)
    out["sales_momentum_28_90"] = out["sales_ma28"] / out["sales_ma90"].replace(0, np.nan)

    for c in ["price_vs_ma7", "price_vs_ma28", "price_vs_ma90", "price_vs_cat_median", "freight_to_price", "sales_momentum_7_28", "sales_momentum_28_90"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        med = float(out[c].median()) if not out[c].isna().all() else 0.0
        out[c] = out[c].fillna(med)

    return out


def build_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    return add_leak_free_lag_features(add_time_features(daily))


DIRECT_FEATURES: List[str] = [
    "log_price", "price", "net_unit_price", "price_change_1d", "price_lag1", "price_lag7", "price_lag28",
    "price_ma7", "price_ma28", "price_ma90", "price_vs_ma7", "price_vs_ma28",
    "price_vs_ma90", "price_vs_cat_median", "sales_lag1", "sales_lag7", "sales_lag14",
    "sales_lag28", "sales_ma7", "sales_ma28", "sales_ma90", "sales_std28",
    "sales_momentum_7_28", "sales_momentum_28_90", "freight_value", "discount", "promotion",
    "reviews_count", "log_freight",
    "freight_lag1", "freight_lag7", "freight_lag28", "freight_ma7", "freight_ma28",
    "freight_ma90", "freight_to_price", "review_score", "dow", "month", "weekofyear",
    "is_weekend", "sin_doy", "cos_doy", "month_sin", "month_cos", "time_index", "time_index_norm"
]

BASELINE_FEATURES: List[str] = [
    "sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28", "sales_ma7", "sales_ma28",
    "sales_ma90", "sales_std28", "sales_momentum_7_28", "sales_momentum_28_90",
    "freight_value", "log_freight", "freight_lag1", "freight_lag7", "freight_lag28",
    "freight_ma7", "freight_ma28", "freight_ma90", "review_score", "dow", "month",
    "weekofyear", "is_weekend", "sin_doy", "cos_doy", "month_sin", "month_cos",
    "time_index", "time_index_norm"
]


def clean_feature_frame(df: pd.DataFrame, features: List[str], feature_stats: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    out = df.copy()
    stats = FEATURE_STATS if feature_stats is None else feature_stats
    for c in features:
        if c not in out.columns:
            out[c] = stats.get(c, 0.0) if isinstance(stats, dict) else 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")
        fallback = float(stats.get(c, 0.0)) if isinstance(stats, dict) else 0.0
        med = float(out[c].median()) if not out[c].isna().all() and np.isfinite(out[c].median()) else fallback
        if isinstance(stats, dict) and c in stats and np.isfinite(stats[c]):
            med = float(stats[c])
        out[c] = out[c].fillna(med)
        out[c] = out[c].replace([np.inf, -np.inf], med)
    return out


def fit_slope_with_controls(df_fit: pd.DataFrame, target_col: str, slope_col: str, control_cols: List[str]) -> Tuple[float, float]:
    cols = [slope_col] + [c for c in control_cols if c in df_fit.columns]
    data = df_fit.dropna(subset=cols + [target_col]).copy()
    if len(data) < 12:
        return CONFIG["PRIOR_ELASTICITY"], 1.0
    X = data[cols].astype(float)
    y = data[target_col].astype(float).values
    if USE_STATSMODELS and sm is not None and add_constant is not None:
        try:
            Xc = add_constant(X, has_constant="add")
            res = sm.OLS(y, Xc).fit(cov_type="HC1")
            slope = float(res.params[slope_col])
            s2 = float(res.bse[slope_col] ** 2) if slope_col in res.bse.index else float(np.var(res.resid, ddof=max(1, X.shape[1] + 1)))
            return slope, max(1e-6, s2)
        except Exception:
            pass
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X.values, y)
        pred = ridge.predict(X.values)
        resid = y - pred
        dof = max(1, len(y) - X.shape[1] - 1)
        s2 = float(np.sum(resid**2) / dof)
        return float(ridge.coef_[0]), max(1e-6, s2)
    except Exception:
        return CONFIG["PRIOR_ELASTICITY"], 1.0


def estimate_pooled_elasticity(df_fit: pd.DataFrame, small_mode: bool = False) -> float:
    data = df_fit.dropna(subset=["log_price", "log_sales"]).copy()
    if len(data) < 50:
        return CONFIG["PRIOR_ELASTICITY"]
    data["month_bucket"] = data["date"].dt.to_period("M").astype(str)
    month_dummies = pd.get_dummies(data["month_bucket"], drop_first=True)
    control_cols = [c for c in ["sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28", "sales_ma7", "sales_ma28", "sales_ma90", "price_change_1d", "price_ma7", "price_ma28", "freight_ma7", "freight_ma28", "time_index_norm", "dow", "is_weekend", "sin_doy", "cos_doy"] if c in data.columns]
    X = pd.concat([data[["log_price"] + control_cols], month_dummies], axis=1).astype(float)
    y = data["log_sales"].astype(float)
    try:
        if USE_STATSMODELS and sm is not None and add_constant is not None:
            X2 = add_constant(X, has_constant="add")
            model = sm.OLS(y, X2).fit(cov_type="HC1")
            coef = float(model.params["log_price"])
        else:
            ridge = Ridge(alpha=1e-4)
            ridge.fit(X.values, y.values)
            coef = float(ridge.coef_[0])
    except Exception:
        try:
            hub = HuberRegressor(alpha=1e-6, max_iter=500)
            hub.fit(data[["log_price"]].values, y.values)
            coef = float(hub.coef_[0])
        except Exception:
            coef = CONFIG["PRIOR_ELASTICITY"]
    if abs(coef) < 0.15:
        weight_data = min(0.5 if small_mode else 0.8, len(data) / 500.0)
        coef = weight_data * coef + (1 - weight_data) * CONFIG["PRIOR_ELASTICITY"]
    return float(np.clip(coef, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"]))


def compute_monthly_group_elasticities(df_all: pd.DataFrame, pooled_prior: float, small_mode: bool = False) -> Tuple[Dict[str, float], pd.DataFrame]:
    df_all = df_all.copy()
    if "elasticity_bucket" not in df_all.columns:
        df_all["elasticity_bucket"] = df_all["date"].dt.to_period("M").astype(str)
    controls = [c for c in ["sales_lag1", "sales_lag7", "sales_ma7", "sales_ma28", "price_change_1d", "price_ma7", "price_ma28", "freight_ma7", "review_score", "dow", "is_weekend", "time_index_norm"] if c in df_all.columns]

    group_stats = []
    raw_slopes = {}
    raw_s2 = {}
    for bucket, sub in df_all.groupby("elasticity_bucket", sort=True):
        n = len(sub)
        price_std = float(sub["price"].std(ddof=0)) if n > 1 else 0.0
        price_unique = int(sub["price"].nunique())
        if n < CONFIG["MIN_CATEGORY_ROWS"] / 10 or price_std <= 1e-9 or price_unique < 4:
            slope, s2, note = pooled_prior, 1.0, "fallback"
        else:
            slope, s2 = fit_slope_with_controls(sub, target_col="log_sales", slope_col="log_price", control_cols=controls)
            note = "ols_or_ridge"
        slope = float(np.clip(slope, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"]))
        raw_slopes[str(bucket)] = slope
        raw_s2[str(bucket)] = float(s2)
        group_stats.append({"bucket": str(bucket), "n": int(n), "price_std": float(price_std), "price_unique": int(price_unique), "raw_slope": float(slope), "s2": float(s2), "note": note})

    slopes_arr = np.array(list(raw_slopes.values()), dtype=float)
    s2_arr = np.array(list(raw_s2.values()), dtype=float)
    n_arr = np.array([g["n"] for g in group_stats], dtype=float)
    if len(slopes_arr) <= 1:
        tau2 = 1e-6
    else:
        w = n_arr / max(n_arr.sum(), 1.0)
        mean_slope = float(np.sum(w * slopes_arr))
        between = float(np.sum(w * (slopes_arr - mean_slope) ** 2))
        within = float(np.sum(w * s2_arr))
        tau2 = max(1e-6, between - within)
        if CONFIG["TAU2_AUTO_TUNE"]:
            tau2 = max(tau2, (0.25 if small_mode else 0.05) * max(between, 1e-6))

    shrunk = {}
    lam_map = {}
    for bucket, slope in raw_slopes.items():
        s2 = float(raw_s2[bucket])
        lam = float(np.clip(tau2 / (tau2 + s2 + 1e-12), 1e-3, 0.999))
        val = lam * slope + (1.0 - lam) * pooled_prior
        val = float(np.clip(val, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"]))
        shrunk[bucket] = val
        lam_map[bucket] = lam

    diag_rows = []
    for row in group_stats:
        b = row["bucket"]
        diag_rows.append({"bucket": b, "n": int(row["n"]), "price_std": float(row["price_std"]), "price_unique": int(row["price_unique"]), "raw_slope": float(raw_slopes[b]), "s2": float(raw_s2[b]), "lambda": float(lam_map[b]), "shrunk": float(shrunk[b]), "note": row["note"]})

    diag_df = pd.DataFrame(diag_rows).sort_values("bucket").reset_index(drop=True)
    return shrunk, diag_df


def _make_direct_monotone_constraints(feature_names: List[str]) -> List[int]:
    monotone = [0] * len(feature_names)
    negative = {"log_price", "price", "price_change_1d", "price_vs_ma7", "price_vs_ma28", "price_vs_ma90", "price_vs_cat_median"}
    positive = {"sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28", "sales_ma7", "sales_ma28", "sales_ma90"}
    for idx, fname in enumerate(feature_names):
        if fname in negative:
            monotone[idx] = -1
        elif fname in positive:
            monotone[idx] = 1
    return monotone


def build_models(X: pd.DataFrame, y: pd.Series, feature_names: List[str], n_models: int = CONFIG["ENSEMBLE_SIZE"], kind: str = "direct", small_mode: bool = False) -> List[Any]:
    ensemble: List[Any] = []
    if len(X) == 0:
        raise ValueError("Пустая обучающая выборка.")
    monotone = _make_direct_monotone_constraints(feature_names) if kind == "direct" else [0] * len(feature_names)
    cat_depth = CONFIG["SMALL_CAT_DEPTH"] if small_mode else CONFIG["CAT_DEPTH"]
    cat_iter = CONFIG["SMALL_CAT_ITER"] if small_mode else CONFIG["CAT_ITER"]
    cat_l2 = CONFIG["SMALL_CAT_L2"] if small_mode else 3.0

    if USE_CATBOOST and CatBoostRegressor is not None:
        for i in range(n_models):
            model = CatBoostRegressor(
                iterations=cat_iter, learning_rate=CONFIG["CAT_LR"], depth=cat_depth,
                loss_function="RMSE", verbose=0, random_seed=42 + i,
                monotone_constraints=monotone, allow_writing_files=False,
                od_type="Iter", od_wait=50, l2_leaf_reg=cat_l2, thread_count=1
            )
            idx = np.random.choice(len(X), size=len(X), replace=True)
            model.fit(X.iloc[idx], y.iloc[idx])
            ensemble.append(model)
            gc.collect()
        return ensemble

    if USE_HGB and HistGradientBoostingRegressor is not None:
        for i in range(n_models):
            try:
                kwargs = dict(
                    loss="squared_error", learning_rate=CONFIG["HGB_LR"], max_iter=CONFIG["HGB_MAX_ITER"],
                    max_depth=CONFIG["HGB_DEPTH"], min_samples_leaf=CONFIG["HGB_MIN_LEAF"],
                    l2_regularization=CONFIG["SMALL_HGB_L2"] if small_mode else 0.1, random_state=42 + i
                )
                model = HistGradientBoostingRegressor(monotonic_cst=monotone, **kwargs)
                idx = np.random.choice(len(X), size=len(X), replace=True)
                model.fit(X.iloc[idx], y.iloc[idx])
                ensemble.append(model)
                gc.collect()
            except Exception:
                pass
        if len(ensemble) > 0:
            return ensemble

    for i in range(n_models):
        max_feat = CONFIG["SMALL_RF_MAX_FEATURES"] if small_mode else "sqrt"
        model = RandomForestRegressor(
            n_estimators=CONFIG["RF_TREES"], max_depth=CONFIG["RF_DEPTH"], max_features=max_feat,
            random_state=42 + i, n_jobs=1
        )
        idx = np.random.choice(len(X), size=len(X), replace=True)
        model.fit(X.iloc[idx], y.iloc[idx])
        ensemble.append(model)
        gc.collect()
    return ensemble


def ensemble_predict(models_local: List[Any], X_local: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if len(models_local) == 0:
        raise ValueError("No models in ensemble")
    preds = np.vstack([m.predict(X_local) for m in models_local])
    return preds.mean(axis=0), preds.std(axis=0, ddof=0)


def predict_direct_log(frame: pd.DataFrame, models_local: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    return ensemble_predict(models_local, frame[DIRECT_FEATURES].astype(float))


def predict_baseline_log(frame: pd.DataFrame, models_local: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    return ensemble_predict(models_local, frame[BASELINE_FEATURES].astype(float))


def structural_predict_log(frame: pd.DataFrame, baseline_models: List[Any], elasticity_map: Dict[str, float], pooled_prior: float, price_ref_col: str = "price_ma28") -> Tuple[np.ndarray, np.ndarray]:
    base_log, base_std = predict_baseline_log(frame, baseline_models)
    price_col = "net_unit_price" if "net_unit_price" in frame.columns else "price"
    price = frame[price_col].astype(float).values
    ref = frame[price_ref_col].astype(float).values if price_ref_col in frame.columns else frame[price_col].astype(float).values
    ref = np.where(np.isfinite(ref) & (ref > 0), ref, np.nanmedian(price) if np.isfinite(np.nanmedian(price)) else 1.0)
    ref = np.where(ref <= 0, 1.0, ref)

    if "elasticity_group" in frame.columns:
        elasticity = frame["elasticity_group"].astype(float).values
    elif "date" in frame.columns:
        elasticity = np.array([elasticity_map.get(str(pd.Timestamp(d).to_period("M")), pooled_prior) for d in frame["date"]], dtype=float)
    else:
        elasticity = np.full(len(frame), pooled_prior, dtype=float)

    elasticity = np.clip(elasticity, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
    price_ratio = np.maximum(price, 1e-9) / np.maximum(ref, 1e-9)
    price_ratio = np.clip(price_ratio, 0.20, 5.0)
    struct_log = base_log + elasticity * np.log(price_ratio)
    return struct_log, base_std


def blended_predict_log(frame: pd.DataFrame, direct_models: List[Any], baseline_models: List[Any], elasticity_map: Dict[str, float], pooled_elasticity: float, w_direct: float) -> np.ndarray:
    direct_log, _ = predict_direct_log(frame, direct_models)
    struct_log, _ = structural_predict_log(frame, baseline_models, elasticity_map, pooled_elasticity)
    return w_direct * direct_log + (1.0 - w_direct) * struct_log


def forecast_future_dates(last_date: pd.Timestamp, n_days: int = CONFIG["HORIZON_DAYS_DEFAULT"]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=n_days, freq="D")})


def current_price_context(base_df: pd.DataFrame) -> Dict[str, Any]:
    return base_df.iloc[-1].to_dict()


def build_one_step_baseline_features(history_df: pd.DataFrame, current_date: pd.Timestamp, base_ctx: Dict[str, Any], history_span_days: int) -> pd.DataFrame:
    h = history_df.sort_values("date").reset_index(drop=True)
    sales_series = h["sales"].astype(float).values
    freight_series = h["freight_value"].astype(float).values if "freight_value" in h.columns else np.zeros(len(h), dtype=float)
    review_series = h["review_score"].astype(float).values if "review_score" in h.columns else np.full(len(h), 4.5)
    dt = pd.Timestamp(current_date)
    dow = int(dt.dayofweek)
    month = int(dt.month)
    doy = int(dt.dayofyear)
    weekofyear = int(dt.isocalendar().week)
    is_weekend = int(dow >= 5)
    last_sales = float(sales_series[-1]) if len(sales_series) else 0.0
    last_freight = float(freight_series[-1]) if len(freight_series) else float(base_ctx.get("freight_value", 0.0))
    last_review = float(review_series[-1]) if len(review_series) else float(base_ctx.get("review_score", 4.5))
    history_norm_den = max(1.0, float(history_span_days - 1))
    time_index = float(len(history_df))
    time_index_norm = min(1.5, time_index / history_norm_den)

    row = {
        "date": dt,
        "sales_lag1": float(sales_series[-1]) if len(sales_series) >= 1 else 0.0,
        "sales_lag7": float(sales_series[-7]) if len(sales_series) >= 7 else last_sales,
        "sales_lag14": float(sales_series[-14]) if len(sales_series) >= 14 else last_sales,
        "sales_lag28": float(sales_series[-28]) if len(sales_series) >= 28 else last_sales,
        "sales_ma7": _tail_mean(sales_series, 7, 0.0),
        "sales_ma28": _tail_mean(sales_series, 28, 0.0),
        "sales_ma90": _tail_mean(sales_series, 90, 0.0),
        "sales_std28": _tail_std(sales_series, 28, 0.0),
        "sales_momentum_7_28": _tail_mean(sales_series, 7, 0.0) / max(_tail_mean(sales_series, 28, 0.0), 1e-9),
        "sales_momentum_28_90": _tail_mean(sales_series, 28, 0.0) / max(_tail_mean(sales_series, 90, 0.0), 1e-9),
        "freight_value": last_freight,
        "log_freight": float(np.log1p(max(last_freight, 0.0))),
        "freight_lag1": float(freight_series[-1]) if len(freight_series) >= 1 else last_freight,
        "freight_lag7": float(freight_series[-7]) if len(freight_series) >= 7 else last_freight,
        "freight_lag28": float(freight_series[-28]) if len(freight_series) >= 28 else last_freight,
        "freight_ma7": _tail_mean(freight_series, 7, last_freight),
        "freight_ma28": _tail_mean(freight_series, 28, last_freight),
        "freight_ma90": _tail_mean(freight_series, 90, last_freight),
        "review_score": last_review,
        "dow": dow, "month": month, "weekofyear": weekofyear, "is_weekend": is_weekend,
        "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "time_index": time_index, "time_index_norm": time_index_norm,
        "sales": float(last_sales), "category": base_ctx.get("category", "unknown"),
        "elasticity_bucket": str(dt.to_period("M")),
    }
    return pd.DataFrame([row])


def build_one_step_direct_features(history_df: pd.DataFrame, current_date: pd.Timestamp, base_ctx: Dict[str, Any], history_span_days: int, price_value: float) -> pd.DataFrame:
    h = history_df.sort_values("date").reset_index(drop=True)
    sales_series = h["sales"].astype(float).values if "sales" in h.columns else np.zeros(len(h), dtype=float)
    price_series = h["price"].astype(float).values if "price" in h.columns else np.full(len(h), float(price_value), dtype=float)
    net_price_series = h["net_unit_price"].astype(float).values if "net_unit_price" in h.columns else price_series.copy()
    freight_series = h["freight_value"].astype(float).values if "freight_value" in h.columns else np.zeros(len(h), dtype=float)
    discount_series = h["discount"].astype(float).values if "discount" in h.columns else np.zeros(len(h), dtype=float)
    promo_series = h["promotion"].astype(float).values if "promotion" in h.columns else np.zeros(len(h), dtype=float)
    reviews_series = h["reviews_count"].astype(float).values if "reviews_count" in h.columns else np.zeros(len(h), dtype=float)
    review_series = h["review_score"].astype(float).values if "review_score" in h.columns else np.full(len(h), 4.5)

    dt = pd.Timestamp(current_date)
    dow = int(dt.dayofweek)
    month = int(dt.month)
    doy = int(dt.dayofyear)
    weekofyear = int(dt.isocalendar().week)
    is_weekend = int(dow >= 5)

    current_price = float(price_value)
    if not np.isfinite(current_price) or current_price <= 0:
        current_price = float(price_series[-1]) if len(price_series) else float(base_ctx.get("price", 1.0))
    current_price = max(current_price, 0.01)
    current_discount = float(discount_series[-1]) if len(discount_series) else float(base_ctx.get("discount", 0.0))
    current_net_price = max(0.01, current_price * (1.0 - current_discount))
    last_price = float(net_price_series[-1]) if len(net_price_series) else current_net_price
    last_sales = float(sales_series[-1]) if len(sales_series) else 0.0
    last_freight = float(freight_series[-1]) if len(freight_series) else float(base_ctx.get("freight_value", 0.0))
    last_review = float(review_series[-1]) if len(review_series) else float(base_ctx.get("review_score", 4.5))
    history_norm_den = max(1.0, float(history_span_days - 1))
    time_index = float(len(history_df))
    time_index_norm = min(1.5, time_index / history_norm_den)

    row = {
        "date": dt,
        "log_price": float(np.log(current_net_price)),
        "price": float(current_price),
        "net_unit_price": float(current_net_price),
        "price_change_1d": float(current_net_price / max(last_price, 1e-9) - 1.0) if len(net_price_series) >= 1 else 0.0,
        "price_lag1": float(net_price_series[-1]) if len(net_price_series) >= 1 else current_net_price,
        "price_lag7": float(net_price_series[-7]) if len(net_price_series) >= 7 else current_net_price,
        "price_lag28": float(net_price_series[-28]) if len(net_price_series) >= 28 else current_net_price,
        "sales_lag1": float(sales_series[-1]) if len(sales_series) >= 1 else 0.0,
        "sales_lag7": float(sales_series[-7]) if len(sales_series) >= 7 else last_sales,
        "sales_lag14": float(sales_series[-14]) if len(sales_series) >= 14 else last_sales,
        "sales_lag28": float(sales_series[-28]) if len(sales_series) >= 28 else last_sales,
        "sales_ma7": _tail_mean(sales_series, 7, 0.0),
        "sales_ma28": _tail_mean(sales_series, 28, 0.0),
        "sales_ma90": _tail_mean(sales_series, 90, 0.0),
        "sales_std28": _tail_std(sales_series, 28, 0.0),
        "sales_momentum_7_28": _tail_mean(sales_series, 7, 0.0) / max(_tail_mean(sales_series, 28, 0.0), 1e-9),
        "sales_momentum_28_90": _tail_mean(sales_series, 28, 0.0) / max(_tail_mean(sales_series, 90, 0.0), 1e-9),
        "freight_value": float(last_freight),
        "log_freight": float(np.log1p(max(last_freight, 0.0))),
        "freight_lag1": float(freight_series[-1]) if len(freight_series) >= 1 else last_freight,
        "freight_lag7": float(freight_series[-7]) if len(freight_series) >= 7 else last_freight,
        "freight_lag28": float(freight_series[-28]) if len(freight_series) >= 28 else last_freight,
        "freight_ma7": _tail_mean(freight_series, 7, last_freight),
        "freight_ma28": _tail_mean(freight_series, 28, last_freight),
        "freight_ma90": _tail_mean(freight_series, 90, last_freight),
        "discount": float(current_discount),
        "promotion": float(promo_series[-1]) if len(promo_series) else float(base_ctx.get("promotion", 0.0)),
        "reviews_count": float(reviews_series[-1]) if len(reviews_series) else float(base_ctx.get("reviews_count", 0.0)),
        "freight_to_price": float(last_freight / max(current_net_price, 1e-9)),
        "price_ma7": _tail_mean(net_price_series, 7, current_net_price),
        "price_ma28": _tail_mean(net_price_series, 28, current_net_price),
        "price_ma90": _tail_mean(net_price_series, 90, current_net_price),
        "price_vs_ma7": float(current_net_price / max(_tail_mean(net_price_series, 7, current_net_price), 1e-9)),
        "price_vs_ma28": float(current_net_price / max(_tail_mean(net_price_series, 28, current_net_price), 1e-9)),
        "price_vs_ma90": float(current_net_price / max(_tail_mean(net_price_series, 90, current_net_price), 1e-9)),
        "price_vs_cat_median": float(current_net_price / max(safe_median(pd.Series(net_price_series), current_net_price), 1e-9)),
        "review_score": float(last_review),
        "dow": dow, "month": month, "weekofyear": weekofyear, "is_weekend": is_weekend,
        "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "time_index": time_index, "time_index_norm": time_index_norm,
        "category": base_ctx.get("category", "unknown"),
        "elasticity_bucket": str(dt.to_period("M")),
    }
    return pd.DataFrame([row])


def _safe_split_sizes(n: int) -> Tuple[int, int]:
    if n < 5:
        raise ValueError("Слишком мало дневных наблюдений для обучения и проверки модели.")
    train_end = max(3, int(n * CONFIG["TRAIN_FRACTION"]))
    val_end = max(train_end + 1, int(n * (CONFIG["TRAIN_FRACTION"] + CONFIG["VAL_FRACTION"])))
    if val_end >= n:
        val_end = n - 1
    if train_end >= val_end:
        train_end = max(1, n - 2)
        val_end = n - 1
    return train_end, val_end


def recursive_baseline_forecast(base_history: pd.DataFrame, horizon_df: pd.DataFrame, baseline_models: List[Any], base_ctx: Dict[str, Any]) -> pd.DataFrame:
    keep_cols = [c for c in ["date", "sales", "freight_value", "review_score"] if c in base_history.columns]
    history = base_history[keep_cols].copy()
    if "freight_value" not in history.columns:
        history["freight_value"] = float(base_ctx.get("freight_value", 0.0))
    if "review_score" not in history.columns:
        history["review_score"] = float(base_ctx.get("review_score", 4.5))
    history_span_days = max(len(base_history), 2)
    outputs = []
    for _, fr in horizon_df.iterrows():
        current_date = pd.Timestamp(fr["date"])
        feat = build_one_step_baseline_features(history, current_date, base_ctx, history_span_days)
        feat = clean_feature_frame(feat, BASELINE_FEATURES)
        X = feat[BASELINE_FEATURES].astype(float)
        pred_log_mean, pred_log_std = ensemble_predict(baseline_models, X)
        pred_log_mean = float(pred_log_mean[0]); pred_log_std = float(pred_log_std[0])
        pred_sales = max(0.0, float(np.expm1(pred_log_mean)))
        outputs.append(pd.DataFrame({"date": [current_date], "base_pred_log_sales": [pred_log_mean], "base_pred_sales": [pred_sales], "base_pred_std_log": [pred_log_std], "year_month": [str(current_date.to_period("M"))]}))
        history = pd.concat([history, pd.DataFrame({"date": [current_date], "sales": [pred_sales], "freight_value": [float(base_ctx.get("freight_value", 0.0))], "review_score": [float(base_ctx.get("review_score", 4.5))]})], ignore_index=True)
    return pd.concat(outputs, ignore_index=True)


def recursive_direct_forecast(base_history: pd.DataFrame, horizon_df: pd.DataFrame, direct_models: List[Any], base_ctx: Dict[str, Any], price_value: float) -> pd.DataFrame:
    keep_cols = [c for c in ["date", "sales", "price", "freight_value", "review_score", "discount", "promotion", "reviews_count", "net_unit_price"] if c in base_history.columns]
    history = base_history[keep_cols].copy()
    if "price" not in history.columns:
        history["price"] = float(price_value)
    if "freight_value" not in history.columns:
        history["freight_value"] = float(base_ctx.get("freight_value", 0.0))
    if "review_score" not in history.columns:
        history["review_score"] = float(base_ctx.get("review_score", 4.5))
    history_span_days = max(len(base_history), 2)
    outputs = []
    for _, fr in horizon_df.iterrows():
        current_date = pd.Timestamp(fr["date"])
        feat = build_one_step_direct_features(history, current_date, base_ctx, history_span_days, price_value)
        feat = clean_feature_frame(feat, DIRECT_FEATURES)
        X = feat[DIRECT_FEATURES].astype(float)
        pred_log_mean, pred_log_std = ensemble_predict(direct_models, X)
        pred_log_mean = float(pred_log_mean[0]); pred_log_std = float(pred_log_std[0])
        pred_sales = max(0.0, float(np.expm1(pred_log_mean)))
        outputs.append(pd.DataFrame({"date": [current_date], "direct_pred_log_sales": [pred_log_mean], "direct_pred_sales": [pred_sales], "direct_pred_std_log": [pred_log_std], "year_month": [str(current_date.to_period("M"))], "price": [price_value]}))
        history = pd.concat([history, pd.DataFrame({"date": [current_date], "sales": [pred_sales], "price": [price_value], "net_unit_price": [float(price_value) * (1.0 - float(base_ctx.get("discount", 0.0)))], "freight_value": [float(base_ctx.get("freight_value", 0.0))], "review_score": [float(base_ctx.get("review_score", 4.5))], "discount": [float(base_ctx.get("discount", 0.0))], "promotion": [float(base_ctx.get("promotion", 0.0))], "reviews_count": [float(base_ctx.get("reviews_count", 0.0))]})], ignore_index=True)
    return pd.concat(outputs, ignore_index=True)


def eval_prediction_frame(frame: pd.DataFrame, y_log_pred: np.ndarray, label: str = "test") -> Dict[str, float]:
    if len(frame) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "smape": float("nan"), "wape": float("nan"), "sigma_log": float("nan")}
    pred_sales = np.expm1(y_log_pred).clip(min=0.0)
    actual_sales = frame["sales"].astype(float).values
    y_log_true = frame["log_sales"].astype(float).values
    rmse = float(np.sqrt(mean_squared_error(actual_sales, pred_sales)))
    mae = float(mean_absolute_error(actual_sales, pred_sales))
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": calculate_mape(actual_sales, pred_sales),
        "smape": calculate_smape(actual_sales, pred_sales),
        "wape": calculate_wape(actual_sales, pred_sales),
        "sigma_log": float(np.std(y_log_true - y_log_pred, ddof=1)) if len(y_log_true) > 1 else 1.0,
    }


def choose_blend_weight(val_df: pd.DataFrame, direct_models: List[Any], baseline_models: List[Any], elasticity_map: Dict[str, float], pooled_elasticity: float) -> Tuple[float, pd.DataFrame]:
    if len(val_df) == 0:
        return 0.5, pd.DataFrame(columns=["w_direct", "rmse", "smape", "wape"])
    direct_log, _ = predict_direct_log(val_df, direct_models)
    struct_log, _ = structural_predict_log(val_df, baseline_models, elasticity_map, pooled_elasticity)
    rows = []
    best_w = 1.0
    best_wape = float("inf")
    for w in np.linspace(0.0, 1.0, 41):
        blend_log = w * direct_log + (1.0 - w) * struct_log
        pred_sales = np.expm1(blend_log).clip(min=0.0)
        y_true = val_df["sales"].astype(float).values
        wape = calculate_wape(y_true, pred_sales)
        rows.append({"w_direct": float(w), "rmse": float(np.sqrt(mean_squared_error(y_true, pred_sales))), "smape": calculate_smape(y_true, pred_sales), "wape": wape})
        if np.isfinite(wape) and wape < best_wape:
            best_wape = wape
            best_w = float(w)
    return best_w, pd.DataFrame(rows)


def _monotone_price_multiplier(price_candidate: float, current_price: float, elasticity: float) -> float:
    elasticity = float(np.clip(elasticity, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"]))
    ratio = max(float(price_candidate), 1e-9) / max(float(current_price), 1e-9)
    ratio = float(np.clip(ratio, 0.20, 5.0))
    mult = np.exp(elasticity * np.log(ratio))
    return float(np.clip(mult, 0.15, 3.0))


def simulate_horizon_profit(base_row: Dict[str, Any], price_candidate: float, future_dates_df: pd.DataFrame, direct_models: List[Any], baseline_models: List[Any], base_history: pd.DataFrame, base_ctx: Dict[str, Any], elasticity_map: Dict[str, float], pooled_elasticity: float, w_direct: float, allow_extrapolate: bool = False, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = overrides or {}
    train_min = float(base_history["price"].min())
    train_max = float(base_history["price"].max())
    requested_price = float(price_candidate)
    current_price_raw = float(base_row.get("price", base_ctx.get("price", requested_price)))
    if not np.isfinite(current_price_raw) or current_price_raw <= 0:
        current_price_raw = float(base_ctx.get("price", max(train_min, 1.0)))
    current_price_model = current_price_raw if allow_extrapolate else float(np.clip(current_price_raw, train_min, train_max))
    price_for_model = requested_price if allow_extrapolate else float(np.clip(requested_price, train_min, train_max))
    ood = bool(not is_price_plausible(requested_price, train_min, train_max, margin=0.25))
    current_discount = float(base_ctx.get("discount", safe_median(base_history.get("discount", pd.Series([0.0])), 0.0)))
    current_discount = float(np.clip(current_discount, 0.0, 0.95))
    discount_base = float(overrides.get("discount", current_discount))
    discount_base *= float(overrides.get("discount_multiplier", 1.0))
    discount_base = float(np.clip(discount_base, 0.0, 0.95))
    freight_val = float(overrides.get("freight_value", base_ctx.get("freight_value", safe_median(base_history.get("freight_value", pd.Series([0.0])), 0.0))))
    freight_val *= float(overrides.get("freight_multiplier", 1.0))
    promo_val = float(overrides.get("promotion", base_ctx.get("promotion", 0.0)))
    unit_cost = float(overrides.get("cost", base_ctx.get("cost", safe_median(base_history.get("cost", pd.Series([current_price_raw * CONFIG["COST_PROXY_RATIO"]])), current_price_raw * CONFIG["COST_PROXY_RATIO"]))))
    unit_cost *= float(overrides.get("cost_multiplier", 1.0))
    unit_cost = max(0.0, unit_cost)
    stock_cap = float(overrides.get("stock_cap", 0.0))
    manual_shock = float(overrides.get("manual_shock_multiplier", 1.0))
    current_net_price_model = max(0.01, current_price_model * (1.0 - current_discount))
    scenario_net_price_model = max(0.01, price_for_model * (1.0 - discount_base))

    baseline_daily = recursive_baseline_forecast(base_history, future_dates_df, baseline_models, base_ctx)
    direct_current_daily = recursive_direct_forecast(base_history, future_dates_df, direct_models, base_ctx, current_price_model)

    direct_level_ratio = np.array(direct_current_daily["direct_pred_sales"].values / np.maximum(baseline_daily["base_pred_sales"].values, 1e-9), dtype=float)
    direct_level_ratio = np.clip(direct_level_ratio, 0.50, 1.50)
    direct_level_factor = np.power(direct_level_ratio, min(max(w_direct, CONFIG["PRICING_DIRECT_WEIGHT_FLOOR"]), CONFIG["PRICING_DIRECT_WEIGHT_CAP"]))

    future_months = [str(pd.Timestamp(d).to_period("M")) for d in future_dates_df["date"]]
    elasticities = np.array([elasticity_map.get(m, pooled_elasticity) for m in future_months], dtype=float)
    elasticities = np.clip(elasticities, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
    price_multiplier = np.array([_monotone_price_multiplier(scenario_net_price_model, current_net_price_model, e) for e in elasticities], dtype=float)

    pred_sales = np.maximum(0.0, baseline_daily["base_pred_sales"].values * direct_level_factor * price_multiplier)
    pred_sales = pred_sales * manual_shock
    pred_log_sales = np.log1p(pred_sales)
    pred_std_log = np.maximum(direct_current_daily["direct_pred_std_log"].values, baseline_daily["base_pred_std_log"].values)

    daily = pd.DataFrame({
        "date": future_dates_df["date"].values,
        "price": float(price_for_model),
        "base_pred_sales": baseline_daily["base_pred_sales"].values,
        "direct_current_sales": direct_current_daily["direct_pred_sales"].values,
        "direct_level_factor": direct_level_factor,
        "pred_sales": pred_sales,
        "pred_log_sales": pred_log_sales,
        "pred_std_log": pred_std_log,
        "elasticity": elasticities,
        "price_multiplier": price_multiplier,
        "discount": discount_base,
        "promotion": promo_val,
        "freight_value": freight_val,
        "cost": float(unit_cost),
    })
    daily["net_unit_price"] = (daily["price"] * (1.0 - daily["discount"])).clip(lower=0.01)
    daily["unconstrained_demand"] = daily["pred_sales"].clip(lower=0.0)
    if stock_cap > 0:
        daily["actual_sales"] = np.minimum(daily["unconstrained_demand"], stock_cap)
    else:
        daily["actual_sales"] = daily["unconstrained_demand"]
    daily["lost_sales"] = (daily["unconstrained_demand"] - daily["actual_sales"]).clip(lower=0.0)
    daily["revenue"] = daily["net_unit_price"] * daily["actual_sales"]
    daily["profit"] = (daily["net_unit_price"] - daily["cost"] - daily["freight_value"]) * daily["actual_sales"]

    std_mean = float(np.nanmean(daily["pred_std_log"].values)) if len(daily) else 0.0
    total_demand = float(np.nansum(daily["actual_sales"].values)) if len(daily) else 0.0
    uncertainty_penalty = CONFIG["UNCERTAINTY_MULTIPLIER"] * std_mean * max((price_for_model - unit_cost), 0.0) * max(total_demand, 1.0)
    mean_abs_disagreement = float(np.nanmean(np.abs(direct_current_daily["direct_pred_sales"].values - baseline_daily["base_pred_sales"].values))) if len(daily) else 0.0
    disagreement_penalty = CONFIG["DISAGREEMENT_PENALTY_SCALE"] * mean_abs_disagreement * max((price_for_model - unit_cost), 0.0)
    raw_profit = float(np.nansum(daily["profit"].values))
    adjusted_profit = float(raw_profit - uncertainty_penalty - disagreement_penalty)

    return {
        "requested_price": float(requested_price),
        "price": float(requested_price),
        "price_for_model": float(price_for_model),
        "current_price_raw": float(current_price_raw),
        "current_price_for_model": float(current_price_model),
        "daily": daily,
        "daily_prices": daily["price"].values if len(daily) else np.array([]),
        "daily_demands": daily["pred_sales"].values if len(daily) else np.array([]),
        "daily_actual_sales": daily["actual_sales"].values if len(daily) else np.array([]),
        "daily_profits": daily["profit"].values if len(daily) else np.array([]),
        "total_profit": raw_profit,
        "adjusted_profit": adjusted_profit,
        "mean_log": daily["pred_log_sales"].values if len(daily) else np.array([]),
        "std_log": daily["pred_std_log"].values if len(daily) else np.array([]),
        "ood_flag": ood,
        "uncertainty_penalty": float(uncertainty_penalty),
        "disagreement_penalty": float(disagreement_penalty),
        "price_multiplier": daily["price_multiplier"].values if len(daily) else np.array([]),
        "elasticity_used": elasticities,
        "direct_current_daily": direct_current_daily,
        "baseline_daily": baseline_daily,
    }


def recommend_price_horizon(base_row: Dict[str, Any], direct_models: List[Any], baseline_models: List[Any], base_history: pd.DataFrame, base_ctx: Dict[str, Any], elasticity_map: Dict[str, float], pooled_elasticity: float, w_direct: float, min_rel: float = CONFIG["MIN_REL_STEP"], max_rel: float = CONFIG["MAX_REL_STEP"], n_grid: int = 120, n_days: int = None, risk_lambda: float = 0.7, price_change_penalty: float = None, search_margin: float = None, allow_extrapolate: bool = False) -> Dict[str, Any]:
    if n_days is None:
        n_days = CONFIG["HORIZON_DAYS_DEFAULT"]
    if price_change_penalty is None:
        price_change_penalty = CONFIG["PRICE_CHANGE_PENALTY_SCALE"]
    if search_margin is None:
        search_margin = CONFIG["SEARCH_MARGIN"]
    future_dates_df = forecast_future_dates(pd.Timestamp(base_history["date"].max()), n_days=n_days)
    train_min = float(base_history["price"].min())
    train_max = float(base_history["price"].max())
    train_span = max(1e-9, train_max - train_min)
    current_price_raw = float(base_row.get("price", base_history["price"].median()))
    if not np.isfinite(current_price_raw) or current_price_raw <= 0:
        current_price_raw = float(base_history["price"].median())
    current_price_model = current_price_raw if allow_extrapolate else float(np.clip(current_price_raw, train_min, train_max))
    cost = max(0.01, float(current_price_raw) * CONFIG["COST_PROXY_RATIO"])

    window_min = max(cost * 1.01, current_price_model * (1.0 + min_rel), train_min - search_margin * train_span)
    window_max = min(current_price_model * (1.0 + max_rel), train_max + search_margin * train_span)
    if not allow_extrapolate:
        window_min = max(window_min, train_min)
        window_max = min(window_max, train_max)
    if not np.isfinite(window_min) or not np.isfinite(window_max) or window_min >= window_max:
        window_min = max(cost * 1.01, train_min)
        window_max = max(window_min + 1e-6, train_max)

    prices = np.linspace(window_min, window_max, max(2, int(n_grid)))
    results = []

    current_sim = simulate_horizon_profit(base_row, current_price_raw, future_dates_df, direct_models, baseline_models, base_history, base_ctx, elasticity_map, pooled_elasticity, w_direct, allow_extrapolate=allow_extrapolate)
    current_profit = float(np.nansum(current_sim["daily_profits"]))
    current_adjusted = float(current_sim["adjusted_profit"])
    current_demand = float(np.nansum(current_sim["daily_demands"]))

    for p in prices:
        sim = simulate_horizon_profit(base_row, float(p), future_dates_df, direct_models, baseline_models, base_history, base_ctx, elasticity_map, pooled_elasticity, w_direct, allow_extrapolate=allow_extrapolate)
        demand_sum = float(np.nansum(sim["daily_demands"]))
        demand_drop_rel = (demand_sum - current_demand) / max(current_demand, 1e-9)
        mean_std = float(np.nanmean(sim["std_log"])) if len(sim["std_log"]) else 0.0
        margin = max(float(sim["price_for_model"]) - cost, 0.0)
        scale = max(margin * max(current_demand, 1.0), 1.0)
        demand_penalty = risk_lambda * max(0.0, -demand_drop_rel) * scale
        jump_penalty = price_change_penalty * abs((float(p) - current_price_model) / max(current_price_model, 1e-9)) * scale
        quad_penalty = CONFIG["QUAD_PENALTY_COEF"] * ((float(p) - current_price_model) ** 2) * scale
        adjusted = float(sim["adjusted_profit"] - demand_penalty - jump_penalty - quad_penalty)
        results.append({"price": float(p), "raw_profit": float(sim["total_profit"]), "adjusted_profit": float(adjusted), "demand_sum": demand_sum, "mean_std": mean_std, "daily": sim["daily"], "ood_flag": bool(sim["ood_flag"]), "uncertainty_penalty": float(sim["uncertainty_penalty"]), "disagreement_penalty": float(sim["disagreement_penalty"]), "demand_penalty": float(demand_penalty), "jump_penalty": float(jump_penalty), "quad_penalty": float(quad_penalty)})

    if len(results) == 0:
        return {"prices": np.array([]), "results": [], "best_idx": None, "best_price": None, "best_profit_adjusted": None, "best_profit_raw": None, "best_daily": None, "ml_best_price": None, "ml_best_profit_raw": None, "ml_best_daily": None, "current_profit_raw": current_profit, "current_profit_adjusted": current_adjusted, "explain": "No valid candidate prices found within search window."}

    best_idx = int(np.nanargmax([r["adjusted_profit"] for r in results]))
    ml_best_idx = int(np.nanargmax([r["raw_profit"] for r in results]))
    best = results[best_idx]
    ml_best = results[ml_best_idx]
    return {
        "prices": np.array([r["price"] for r in results]),
        "results": results,
        "best_idx": best_idx,
        "best_price": float(best["price"]),
        "best_profit_adjusted": float(best["adjusted_profit"]),
        "best_profit_raw": float(best["raw_profit"]),
        "best_daily": best["daily"],
        "ml_best_price": float(ml_best["price"]),
        "ml_best_profit_raw": float(ml_best["raw_profit"]),
        "ml_best_daily": ml_best["daily"],
        "current_profit_raw": float(current_profit),
        "current_profit_adjusted": float(current_adjusted),
        "explain": {"search_min": float(window_min), "search_max": float(window_max), "n_candidates": len(results), "train_range": (float(train_min), float(train_max)), "current_price": float(current_price_raw), "pricing_w_direct": float(min(max(w_direct, CONFIG["PRICING_DIRECT_WEIGHT_FLOOR"]), CONFIG["PRICING_DIRECT_WEIGHT_CAP"]))},
    }


def decision_flag(base_row: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    current_price = float(base_row.get("price", 1.0))
    best_price = float(rec["best_price"]) if rec.get("best_price") is not None else current_price
    current_adjusted = float(rec.get("current_profit_adjusted", rec.get("current_profit_raw", 0.0)))
    best_adjusted = float(rec.get("best_profit_adjusted", 0.0))
    improvement = best_adjusted - current_adjusted
    rel_imp = improvement / max(1e-3, abs(current_adjusted))
    ood = bool(base_row.get("_ood_flag", False))
    best_step = abs(best_price - current_price) / max(current_price, 1e-9)
    big_step = bool(best_step > CONFIG["MAX_REL_STEP"])
    profit_ok = bool((rel_imp >= CONFIG["MIN_PROFIT_REL_IMPROV"]) or (improvement >= CONFIG["ABSOLUTE_MIN_PROFIT_IMPROV"]))
    auto_apply = (not ood) and (not big_step) and profit_ok
    return {"auto_apply": bool(auto_apply), "reasons": {"ood": bool(ood), "big_step": bool(big_step), "profit_ok": bool(profit_ok), "rel_imp": float(rel_imp), "improvement": float(improvement), "current_adjusted": float(current_adjusted), "best_adjusted": float(best_adjusted)}}


@st.cache_resource
def run_full_pricing_analysis(orders: pd.DataFrame, order_items: pd.DataFrame, products: pd.DataFrame, reviews: pd.DataFrame, target_category: str, target_sku: str):
    raw_df = build_raw_frame(orders, order_items, products, reviews)
    if len(raw_df) == 0:
        raise ValueError("После объединения данных не осталось строк.")
    sku_df = raw_df[(raw_df["product_category_name"] == target_category) & (raw_df["product_id"].astype(str) == str(target_sku))].copy()
    if len(sku_df) == 0:
        raise ValueError("Для выбранной категории и SKU нет данных.")

    total_sales = len(sku_df)
    small_data_mode = total_sales < CONFIG["MIN_ML_SALES"] or CONFIG.get("FORCE_ENHANCED_MODE", False)

    daily_base = build_daily_sku_frame(sku_df, target_sku)
    if small_data_mode:
        daily_base = robust_clean_dirty_data(daily_base)
    daily_base = build_feature_matrix(daily_base).dropna(subset=["sales", "price", "log_sales", "log_price"]).reset_index(drop=True)
    if len(daily_base) < 5:
        raise ValueError("Слишком мало дневных наблюдений после агрегации для обучения модели.")

    all_feats = list(dict.fromkeys(DIRECT_FEATURES + BASELINE_FEATURES))
    n = len(daily_base)
    train_end, val_end = _safe_split_sizes(n)
    train_raw = daily_base.iloc[:train_end].copy().reset_index(drop=True)
    val_raw = daily_base.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_raw = daily_base.iloc[val_end:].copy().reset_index(drop=True)

    global FEATURE_STATS
    FEATURE_STATS = fit_feature_stats(train_raw, all_feats)

    train_df = clean_feature_frame(train_raw, all_feats, FEATURE_STATS)
    val_df = clean_feature_frame(val_raw, all_feats, FEATURE_STATS)
    test_df = clean_feature_frame(test_raw, all_feats, FEATURE_STATS)

    if small_data_mode and len(train_df) < 300:
        train_df = clean_feature_frame(augment_price_variations(train_df), all_feats, FEATURE_STATS)

    X_train_direct = train_df[DIRECT_FEATURES].astype(float).copy()
    y_train = train_df["log_sales"].astype(float).copy()
    X_train_base = train_df[BASELINE_FEATURES].astype(float).copy()

    base_ctx = current_price_context(daily_base)
    base_ctx["category"] = target_category
    base_ctx["product_id"] = target_sku

    fixed_log_price_coef = estimate_pooled_elasticity(train_df, small_mode=small_data_mode)
    shrunk_random_effects, _ = compute_monthly_group_elasticities(train_df, fixed_log_price_coef, small_mode=small_data_mode)

    daily_base["elasticity_group"] = daily_base["date"].dt.to_period("M").astype(str).map(shrunk_random_effects).fillna(fixed_log_price_coef)
    train_df = clean_feature_frame(train_df, all_feats, FEATURE_STATS)
    val_df = clean_feature_frame(val_df, all_feats, FEATURE_STATS)
    test_df = clean_feature_frame(test_df, all_feats, FEATURE_STATS)

    direct_models = build_models(X_train_direct, y_train, DIRECT_FEATURES, n_models=CONFIG["ENSEMBLE_SIZE"], kind="direct", small_mode=small_data_mode)
    baseline_models = build_models(X_train_base, y_train, BASELINE_FEATURES, n_models=CONFIG["ENSEMBLE_SIZE"], kind="baseline", small_mode=small_data_mode)

    w_direct, _ = choose_blend_weight(val_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef)

    holdout_metrics = eval_prediction_frame(test_df, blended_predict_log(test_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, w_direct), label="holdout") if len(test_df) > 0 else {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "smape": float("nan"), "wape": float("nan"), "sigma_log": float("nan")}

    latest_row = dict(base_ctx)
    latest_row["requested_price"] = float(base_ctx.get("price", train_df["price"].median()))
    latest_row["_ood_flag"] = bool(not is_price_plausible(float(latest_row["requested_price"]), float(train_df["price"].min()), float(train_df["price"].max()), margin=0.25))

    rec = recommend_price_horizon(latest_row, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct)
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()))

    current_sim = simulate_horizon_profit(latest_row, float(base_ctx.get("price")), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct)
    optimal_sim = simulate_horizon_profit(latest_row, float(rec["best_price"]), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct)

    profit_curve_df = pd.DataFrame([{"price": p, "adjusted_profit": simulate_horizon_profit(latest_row, p, future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct)["adjusted_profit"]} for p in np.linspace(float(base_ctx.get("price")) * 0.8, float(base_ctx.get("price")) * 1.2, 50)])

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        daily_base.to_excel(writer, sheet_name="История", index=False)
        current_sim["daily"].to_excel(writer, sheet_name="Прогноз_текущая", index=False)
        optimal_sim["daily"].to_excel(writer, sheet_name="Прогноз_оптимальная", index=False)
        profit_curve_df.to_excel(writer, sheet_name="Кривая_прибыли", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="Метрики", index=False)
        pd.DataFrame(list(shrunk_random_effects.items()), columns=["Месяц", "Эластичность"]).to_excel(writer, sheet_name="Эластичность", index=False)
    excel_buffer.seek(0)

    profit_lift_pct = ((optimal_sim["adjusted_profit"] - current_sim["adjusted_profit"]) / max(current_sim["adjusted_profit"], 1) * 100) if current_sim["adjusted_profit"] > 0 else 0

    return {
        "daily": daily_base,
        "recommendation": rec,
        "forecast_current": current_sim["daily"],
        "forecast_optimal": optimal_sim["daily"],
        "profit_curve": profit_curve_df,
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "elasticity_map": shrunk_random_effects,
        "current_price": float(base_ctx.get("price")),
        "best_price": float(rec.get("best_price")),
        "current_profit": float(current_sim.get("adjusted_profit", current_sim.get("total_profit", 0.0))),
        "best_profit": float(optimal_sim.get("adjusted_profit", optimal_sim.get("total_profit", 0.0))),
        "profit_lift_pct": profit_lift_pct,
        "excel_buffer": excel_buffer,
        "flag": decision_flag(latest_row, rec),
        "_trained_bundle": {
            "direct_models": direct_models,
            "baseline_models": baseline_models,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "elasticity_map": shrunk_random_effects,
            "pooled_elasticity": fixed_log_price_coef,
            "w_direct": w_direct,
        },
    }


@st.cache_resource
def run_full_pricing_analysis_universal(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    region: Optional[str] = None,
    channel: Optional[str] = None,
    segment: Optional[str] = None,
):
    txn = normalized_txn.copy()
    if len(txn) == 0:
        raise ValueError("Пустой датасет после нормализации.")
    sku_df = txn[(txn["category"].astype(str) == str(target_category)) & (txn["product_id"].astype(str) == str(target_sku))].copy()
    if len(sku_df) == 0:
        raise ValueError("Для выбранной категории и SKU нет данных.")

    daily_base = build_daily_from_transactions(txn, target_sku, region=region, channel=channel, segment=segment)
    daily_base = robust_clean_dirty_data(daily_base)
    daily_base = build_feature_matrix(daily_base).dropna(subset=["sales", "price", "log_sales", "log_price"]).reset_index(drop=True)
    if len(daily_base) < 5:
        raise ValueError("Слишком мало дневных наблюдений после агрегации.")

    all_feats = list(dict.fromkeys(DIRECT_FEATURES + BASELINE_FEATURES))
    n = len(daily_base)
    train_end, val_end = _safe_split_sizes(n)
    train_raw = daily_base.iloc[:train_end].copy().reset_index(drop=True)
    val_raw = daily_base.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_raw = daily_base.iloc[val_end:].copy().reset_index(drop=True)

    global FEATURE_STATS
    FEATURE_STATS = fit_feature_stats(train_raw, all_feats)
    train_df = clean_feature_frame(train_raw, all_feats, FEATURE_STATS)
    val_df = clean_feature_frame(val_raw, all_feats, FEATURE_STATS)
    test_df = clean_feature_frame(test_raw, all_feats, FEATURE_STATS)

    X_train_direct = train_df[DIRECT_FEATURES].astype(float).copy()
    y_train = train_df["log_sales"].astype(float).copy()
    X_train_base = train_df[BASELINE_FEATURES].astype(float).copy()

    base_ctx = current_price_context(daily_base)
    base_ctx["category"] = target_category
    base_ctx["product_id"] = target_sku
    fixed_log_price_coef = estimate_pooled_elasticity(train_df, small_mode=True)
    shrunk_random_effects, _ = compute_monthly_group_elasticities(train_df, fixed_log_price_coef, small_mode=True)
    direct_models = build_models(X_train_direct, y_train, DIRECT_FEATURES, kind="direct", small_mode=True)
    baseline_models = build_models(X_train_base, y_train, BASELINE_FEATURES, kind="baseline", small_mode=True)
    w_direct, _ = choose_blend_weight(val_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef)
    holdout_metrics = eval_prediction_frame(test_df, blended_predict_log(test_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, w_direct), label="holdout") if len(test_df) > 0 else {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "smape": float("nan"), "wape": float("nan"), "sigma_log": float("nan")}
    if len(test_df) > 0:
        direct_log, _ = predict_direct_log(test_df, direct_models)
        base_log, _ = predict_baseline_log(test_df, baseline_models)
        final_log = blended_predict_log(test_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, w_direct)
        holdout_predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d"),
                "series_id": str(target_sku),
                "actual_sales": test_df["sales"].astype(float).values,
                "pred_baseline": np.expm1(base_log).clip(min=0.0),
                "pred_direct": np.expm1(direct_log).clip(min=0.0),
                "pred_final": np.expm1(final_log).clip(min=0.0),
                "actual_price": test_df["price"].astype(float).values if "price" in test_df.columns else np.nan,
                "actual_discount": test_df["discount"].astype(float).values if "discount" in test_df.columns else np.nan,
                "actual_promotion": test_df["promotion"].astype(float).values if "promotion" in test_df.columns else np.nan,
                "actual_stock": test_df["stock"].astype(float).values if "stock" in test_df.columns else np.nan,
                "actual_revenue": test_df["revenue"].astype(float).values if "revenue" in test_df.columns else np.nan,
            }
        )
        holdout_predictions["abs_error"] = np.abs(holdout_predictions["actual_sales"] - holdout_predictions["pred_final"])
        holdout_predictions["pct_error"] = holdout_predictions["abs_error"] / np.maximum(holdout_predictions["actual_sales"], 1e-9)
        holdout_predictions["signed_error"] = holdout_predictions["pred_final"] - holdout_predictions["actual_sales"]
    else:
        holdout_predictions = pd.DataFrame(columns=["date", "series_id", "actual_sales", "pred_baseline", "pred_direct", "pred_final", "actual_price", "actual_discount", "actual_promotion", "actual_stock", "actual_revenue", "abs_error", "pct_error", "signed_error"])

    latest_row = dict(base_ctx)
    latest_row["requested_price"] = float(base_ctx.get("price", train_df["price"].median()))
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()))
    baseline_price = float(safe_median(daily_base["price"], float(base_ctx.get("price", 1.0))))
    as_is_sim = simulate_horizon_profit(latest_row, float(base_ctx.get("price")), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct)
    neutral_overrides = {
        "discount": 0.0,
        "promotion": 0.0,
        "freight_value": float(safe_median(daily_base.get("freight_value", pd.Series([0.0])), 0.0)),
        "cost": float(safe_median(daily_base.get("cost", pd.Series([float(base_ctx.get("price", 1.0)) * CONFIG["COST_PROXY_RATIO"]])), float(base_ctx.get("price", 1.0) * CONFIG["COST_PROXY_RATIO"]))),
    }
    baseline_sim = simulate_horizon_profit(latest_row, baseline_price, future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct, overrides=neutral_overrides)
    confidence = float(1.0 / (1.0 + max(0.0, holdout_metrics.get("wape", 100.0) / 100.0)))

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        daily_base.to_excel(writer, sheet_name="history", index=False)
        baseline_sim["daily"].to_excel(writer, sheet_name="neutral_baseline", index=False)
        as_is_sim["daily"].to_excel(writer, sheet_name="as_is", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="metrics", index=False)
    excel_buffer.seek(0)
    feature_usage_rows = []
    for f in all_feats:
        source_found = f in daily_base.columns
        source_missing = float(daily_base[f].isna().mean()) if source_found else 1.0
        source_unique = int(daily_base[f].nunique(dropna=True)) if source_found else 0
        feature_usage_rows.append(
            {
                "feature": f,
                "source_found": bool(source_found),
                "source_missing_share": source_missing,
                "source_unique_count": source_unique,
                "eligible_for_model": bool(source_found and source_unique > 1),
                "listed_in_feature_set": bool((f in DIRECT_FEATURES or f in BASELINE_FEATURES)),
                "fallback_only": bool((not source_found) and (f in train_df.columns)),
            }
        )
    feature_usage_report = pd.DataFrame(feature_usage_rows)
    scenario_forecast = None
    delta_vs_as_is = None
    delta_vs_baseline = {
        "demand_total": float(as_is_sim["daily"]["actual_sales"].sum() - baseline_sim["daily"]["actual_sales"].sum()),
        "revenue_total": float(as_is_sim["daily"]["revenue"].sum() - baseline_sim["daily"]["revenue"].sum()),
        "profit_total": float(as_is_sim["daily"]["profit"].sum() - baseline_sim["daily"]["profit"].sum()),
    }
    scenario_daily_output = pd.DataFrame(
        {
            "date": pd.to_datetime(as_is_sim["daily"]["date"]).dt.strftime("%Y-%m-%d"),
            "series_id": str(target_sku),
            "baseline_demand": baseline_sim["daily"]["actual_sales"].values,
            "as_is_demand": as_is_sim["daily"]["actual_sales"].values,
            "scenario_demand": as_is_sim["daily"]["actual_sales"].values,
            "actual_sales_after_cap": as_is_sim["daily"]["actual_sales"].values,
            "lost_sales": as_is_sim["daily"]["lost_sales"].values,
            "scenario_revenue": as_is_sim["daily"]["revenue"].values,
            "scenario_profit": as_is_sim["daily"]["profit"].values,
            "scenario_price": as_is_sim["daily"]["price"].values,
            "scenario_discount": as_is_sim["daily"]["discount"].values,
            "scenario_promotion": as_is_sim["daily"]["promotion"].values,
            "scenario_stock": as_is_sim["daily"].get("stock_cap", pd.Series([np.nan] * len(as_is_sim["daily"]))).values if isinstance(as_is_sim["daily"], pd.DataFrame) else np.nan,
        }
    )
    feature_report = feature_usage_report.rename(
        columns={
            "feature": "factor_name",
            "source_found": "found_raw",
            "eligible_for_model": "used_in_model",
        }
    ).copy()
    feature_report["aggregated_daily"] = feature_report["found_raw"]
    feature_report["used_in_scenario"] = feature_report["factor_name"].isin(["price", "discount", "promotion", "freight_value", "cost", "stock"])
    feature_report["role"] = feature_report["factor_name"].map({f.name: f.role for f in CANONICAL_FIELDS}).fillna("unknown")
    feature_report["dtype"] = feature_report["factor_name"].map({f.name: f.dtype for f in CANONICAL_FIELDS}).fillna("unknown")
    feature_report["missing_share"] = feature_report["source_missing_share"]
    feature_report["n_unique"] = feature_report["source_unique_count"]
    feature_report["variability_flag"] = feature_report["n_unique"] > 1
    feature_report["reason_excluded"] = np.where(~feature_report["used_in_model"], "not_eligible_or_missing", "")
    feature_report = feature_report[["factor_name", "found_raw", "aggregated_daily", "used_in_model", "used_in_scenario", "role", "dtype", "missing_share", "n_unique", "variability_flag", "reason_excluded"]]

    run_summary = {
        "config": {
            "commit": _safe_git_commit(),
            "date_utc": str(pd.Timestamp.utcnow()),
            "series_id": str(target_sku),
            "category": str(target_category),
            "train_period": [str(train_df["date"].min()) if "date" in train_df.columns else None, str(train_df["date"].max()) if "date" in train_df.columns else None],
            "validation_period": [str(val_df["date"].min()) if "date" in val_df.columns else None, str(val_df["date"].max()) if "date" in val_df.columns else None],
            "holdout_period": [str(test_df["date"].min()) if "date" in test_df.columns else None, str(test_df["date"].max()) if "date" in test_df.columns else None],
            "horizon_days": int(len(future_dates)),
            "w_direct": float(w_direct),
            "direct_features": DIRECT_FEATURES,
            "baseline_features": BASELINE_FEATURES,
        },
        "dataset_passport": _build_dataset_passport(txn),
        "metrics_summary": {"holdout": holdout_metrics},
        "warnings": [],
        "feature_usage_report": feature_report.to_dict("records"),
        "scenario_inputs": {"as_is": {"price": float(base_ctx.get("price")), "discount": float(base_ctx.get("discount", 0.0))}, "neutral_baseline": neutral_overrides},
        "scenario_output_summary": {
            "baseline_demand_total": float(baseline_sim["daily"]["actual_sales"].sum()),
            "as_is_demand_total": float(as_is_sim["daily"]["actual_sales"].sum()),
            "scenario_demand_total": float(as_is_sim["daily"]["actual_sales"].sum()),
            "baseline_revenue_total": float(baseline_sim["daily"]["revenue"].sum()),
            "as_is_revenue_total": float(as_is_sim["daily"]["revenue"].sum()),
            "scenario_revenue_total": float(as_is_sim["daily"]["revenue"].sum()),
            "baseline_profit_total": float(baseline_sim["daily"]["profit"].sum()),
            "as_is_profit_total": float(as_is_sim["daily"]["profit"].sum()),
            "scenario_profit_total": float(as_is_sim["daily"]["profit"].sum()),
        },
    }
    current_price = float(base_ctx.get("price"))
    curve_prices = np.linspace(max(0.01, current_price * 0.9), current_price * 1.1, 9)
    profit_curve = pd.DataFrame(
        [
            {
                "price": p,
                "adjusted_profit": simulate_horizon_profit(
                    latest_row,
                    float(p),
                    future_dates,
                    direct_models,
                    baseline_models,
                    daily_base,
                    base_ctx,
                    shrunk_random_effects,
                    fixed_log_price_coef,
                    w_direct,
                )["adjusted_profit"],
            }
            for p in curve_prices
        ]
    )
    return {
        "history_daily": daily_base,
        "quality_report": {"holdout_metrics": holdout_metrics},
        "feature_usage_report": feature_usage_report,
        "feature_report": feature_report,
        "neutral_baseline_forecast": baseline_sim["daily"],
        "as_is_forecast": as_is_sim["daily"],
        "scenario_forecast": scenario_forecast,
        "delta_vs_as_is": delta_vs_as_is,
        "delta_vs_baseline": delta_vs_baseline,
        "warnings": [],
        "profit_curve": profit_curve,
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "elasticity_map": shrunk_random_effects,
        "current_price": float(base_ctx.get("price")),
        "scenario_price": float(base_ctx.get("price")),
        "current_profit": float(as_is_sim.get("adjusted_profit", as_is_sim.get("total_profit", 0.0))),
        "best_profit": float(as_is_sim.get("adjusted_profit", as_is_sim.get("total_profit", 0.0))),
        "profit_lift_pct": 0.0,
        "excel_buffer": excel_buffer,
        "run_summary_json": json.dumps(run_summary, ensure_ascii=False, indent=2).encode("utf-8"),
        "holdout_predictions_csv": holdout_predictions.to_csv(index=False).encode("utf-8"),
        "scenario_daily_output_csv": scenario_daily_output.to_csv(index=False).encode("utf-8"),
        "feature_report_csv": feature_report.to_csv(index=False).encode("utf-8"),
        "flag": {"auto_apply": False, "reasons": {"mode": "single-core-what-if"}},
        "_trained_bundle": {
            "direct_models": direct_models,
            "baseline_models": baseline_models,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "elasticity_map": shrunk_random_effects,
            "pooled_elasticity": fixed_log_price_coef,
            "w_direct": w_direct,
            "confidence": confidence,
        },
    }


def run_what_if_projection(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: Optional[int] = None,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float = 0.0,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_history = trained_bundle["daily_base"].copy()
    base_ctx = dict(trained_bundle["base_ctx"])
    latest_row = dict(trained_bundle["latest_row"])

    scenario_overrides = dict(overrides or {})
    scenario_overrides.setdefault("freight_multiplier", float(freight_multiplier))
    scenario_overrides.setdefault("discount_multiplier", float(discount_multiplier))
    scenario_overrides.setdefault("cost_multiplier", float(cost_multiplier))
    if stock_cap:
        scenario_overrides["stock_cap"] = float(stock_cap)
    if float(demand_multiplier) != 1.0:
        scenario_overrides["manual_shock_multiplier"] = float(demand_multiplier)

    base_ctx["price"] = float(manual_price)
    if "freight_value" in base_ctx:
        base_ctx["freight_value"] = float(base_ctx["freight_value"]) * float(freight_multiplier)
    if "discount" in base_ctx:
        base_ctx["discount"] = float(base_ctx["discount"]) * float(scenario_overrides.get("discount_multiplier", 1.0))
    if "cost" in base_ctx:
        base_ctx["cost"] = float(base_ctx["cost"]) * float(scenario_overrides.get("cost_multiplier", 1.0))

    latest_row.update(base_ctx)
    latest_row["requested_price"] = float(manual_price)

    future_dates = trained_bundle["future_dates"]
    if horizon_days is not None:
        future_dates = forecast_future_dates(pd.Timestamp(base_history["date"].max()), n_days=int(horizon_days))

    sim = simulate_horizon_profit(
        latest_row,
        float(manual_price),
        future_dates,
        trained_bundle["direct_models"],
        trained_bundle["baseline_models"],
        base_history,
        base_ctx,
        trained_bundle["elasticity_map"],
        trained_bundle["pooled_elasticity"],
        trained_bundle["w_direct"],
        overrides=scenario_overrides,
    )
    daily = sim["daily"].copy()
    demand_total = float(daily["actual_sales"].sum()) if "actual_sales" in daily.columns else 0.0
    profit_total = float(daily["profit"].sum()) if "profit" in daily.columns else 0.0
    revenue_total = float(daily["revenue"].sum()) if "revenue" in daily.columns else 0.0
    lost_sales_total = float(daily["lost_sales"].sum()) if "lost_sales" in daily.columns else 0.0
    confidence = float(trained_bundle.get("confidence", 0.6))
    return {"daily": daily, "demand_total": demand_total, "profit_total": profit_total, "revenue_total": revenue_total, "lost_sales_total": lost_sales_total, "confidence": confidence, "uncertainty": 1.0 - confidence}


st.set_page_config(page_title="💰 AI What-if Engine", layout="wide", page_icon="💰")

if "results" not in st.session_state:
    st.session_state.results = None
if "what_if_result" not in st.session_state:
    st.session_state.what_if_result = None
    st.session_state.scenario_table = None
    st.session_state.sensitivity_df = None
if "app_stage" not in st.session_state:
    st.session_state.app_stage = "landing"
if "selected_category_for_results" not in st.session_state:
    st.session_state.selected_category_for_results = None
if "selected_sku_for_results" not in st.session_state:
    st.session_state.selected_sku_for_results = None
if "scenario_table" not in st.session_state:
    st.session_state.scenario_table = None
if "sensitivity_df" not in st.session_state:
    st.session_state.sensitivity_df = None

PLOTLY_WORKSPACE_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=Space+Grotesk:wght@600;700&display=swap');

:root {
  --bg: #0A0A12;
  --text: #F0F4FF;
  --muted: #A0A8C0;
  --card: rgba(20,22,38,.92);
  --border: rgba(255,255,255,.12);
  --orange: #FF8A00;
  --magenta: #E400FF;
  --cyan: #00D4FF;
}

html, body, [class*="css"], .stApp { font-family: Inter, sans-serif; color: var(--text); }
.stApp {
  background:
    radial-gradient(circle at 15% -10%, rgba(255,138,0,.22), transparent 35%),
    radial-gradient(circle at 80% -10%, rgba(228,0,255,.20), transparent 35%),
    radial-gradient(circle at 50% 0%, rgba(0,212,255,.10), transparent 42%),
    var(--bg);
}
.block-container { max-width: 1450px; padding-top: 1.2rem; padding-bottom: 2rem; }
#MainMenu, footer, header { visibility: hidden; }

.glass-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  backdrop-filter: blur(12px);
  box-shadow: 0 12px 30px rgba(0,0,0,.35), 0 0 24px rgba(228,0,255,.08);
  animation: fadeScale .45s ease;
}
.metric-card:hover, .feature-card:hover {
  transform: translateY(-3px);
  transition: .25s ease;
  border-color: rgba(255,138,0,.5);
  box-shadow: 0 12px 32px rgba(255,138,0,.16), 0 0 20px rgba(0,212,255,.10);
}
.hero-title {
  font-family: "Space Grotesk", Inter, sans-serif;
  font-size: clamp(2rem, 5vw, 3.5rem);
  font-weight: 700;
  line-height: 1.05;
  background: linear-gradient(90deg, var(--orange), #ffbf63, var(--magenta));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: .5rem;
}
.hero-sub { color: var(--muted); font-size: 1.03rem; }
.section-title { font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.2rem; margin: .2rem 0 .7rem; }
.timeline { display:flex; gap:10px; flex-wrap:wrap; }
.timeline-step { padding: 8px 12px; border-radius: 999px; border:1px solid var(--border); background: rgba(255,255,255,.03); font-size:.84rem; }
.kpi-label {color:var(--muted); font-size:.84rem;} .kpi-val{font-size:1.6rem;font-weight:800;}
.kpi-pos {color:#3bd98c;font-weight:700;font-size:.85rem;} .kpi-neg{color:#ff5a7d;font-weight:700;font-size:.85rem;}
.explain {color:var(--muted); font-size:.86rem; margin-top:.4rem;}
.micro-note { color: var(--muted); font-size: .82rem; line-height: 1.35; margin-top: .35rem; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:5px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; height:40px; }
.stButton button, .stDownloadButton button { border-radius:12px !important; border:1px solid rgba(255,138,0,.55)!important; }
.stButton button:hover, .stDownloadButton button:hover { box-shadow:0 0 24px rgba(255,138,0,.28); }
.pill-row { display:flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.pill { border:1px solid var(--border); color:var(--muted); background:rgba(255,255,255,.03); border-radius:999px; padding:6px 10px; font-size:.78rem; }
.big-cta { text-align: center; margin-top: .3rem; }
.big-cta h3 { margin: .25rem 0 .5rem; font-weight: 700; }
.mock-shell {
  border: 1px solid rgba(255,255,255,.28);
  border-radius: 28px;
  padding: 24px;
  background: linear-gradient(180deg, rgba(8,9,22,.95), rgba(7,8,20,.92));
  box-shadow: inset 0 0 45px rgba(228,0,255,.08), 0 15px 45px rgba(0,0,0,.45);
}
.mock-top {
  display:flex; align-items:center; justify-content:space-between; margin-bottom: 16px;
}
.badge-id {
  border:1px solid rgba(255,255,255,.23); border-radius: 13px; padding: 8px 14px; font-weight:600; color:#D9DEEF; background: rgba(255,255,255,.03);
}
.hero-center { text-align:center; padding: 10px 0 6px; }
.hero-sub2 { color:#cfd5ec; opacity:.9; max-width:660px; margin: 0 auto 12px; }
.cta-glow button {
  background: linear-gradient(90deg, #ffcc66, #ff8a00 40%, #f96af7 70%, #8f5cff) !important;
  color:#111 !important; font-weight: 800 !important; border: none !important;
  box-shadow: 0 0 28px rgba(255,138,0,.38), 0 0 30px rgba(228,0,255,.34) !important;
}
.feature-grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; }
.fcard { border:1px solid rgba(255,255,255,.16); border-radius: 13px; padding: 10px; background: rgba(255,255,255,.05); min-height: 94px; }
.fcard h5 { margin:0 0 3px; font-size:.93rem; }
.fcard p { margin:0; font-size:.75rem; color:var(--muted); line-height: 1.28; }
.stage-row { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-top:8px; flex-wrap:wrap; }
.stage-dot { width:42px; height:42px; border-radius:999px; display:flex; align-items:center; justify-content:center; background: radial-gradient(circle at 30% 20%, #ffd96c, #ff8a00 45%, #ff4fd8 100%); color:#241625; font-weight:800; border:1px solid rgba(255,255,255,.24); box-shadow:0 0 20px rgba(255,138,0,.35);}
.stage-item { text-align:center; min-width:74px; }
.stage-item span { display:block; margin-top:4px; font-size:.76rem; color:#d5daf0; }
.tech-list { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.tech-pill { border:1px solid rgba(255,255,255,.2); border-radius:10px; padding:5px 9px; font-size:.78rem; background: rgba(255,255,255,.04); }
.dashboard-grid { display:grid; grid-template-columns: 1.25fr .95fr; gap: 14px; }
.hero-kpis { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
.hero-kpi { border:1px solid rgba(255,255,255,.14); border-radius: 14px; padding: 12px; background: rgba(255,255,255,.03); }
.hero-kpi h4 { margin:0 0 4px; font-size:.92rem; font-weight:700; color:#e9edf8; }
.hero-kpi p { margin:0; font-size:.78rem; color:var(--muted); line-height:1.35; }
.deliverables-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px; }
.deliver-item { border:1px solid rgba(255,255,255,.16); border-radius: 12px; padding: 10px; background: rgba(255,255,255,.04);}
.deliver-item h5 { margin:0 0 4px; font-size:.9rem; }
.deliver-item p { margin:0; color:var(--muted); font-size:.77rem; line-height:1.35; }
.guide-box { border-left:3px solid var(--cyan); padding:10px 12px; border-radius:10px; background: rgba(0,212,255,.08); margin: 10px 0 4px; }
.guide-box ol { margin:0; padding-left: 18px; }
.guide-box li { margin: 4px 0; color:#d7def4; font-size:.82rem; }
@keyframes fadeScale { from{opacity:0;transform:translateY(8px) scale(.98);} to{opacity:1;transform:translateY(0) scale(1);} }
@media (max-width: 980px) {
  .block-container { padding-top: .6rem; padding-left: .8rem; padding-right: .8rem; }
  .glass-card { padding: 14px; border-radius: 14px; }
  .kpi-val { font-size: 1.25rem; }
  .hero-sub { font-size: .95rem; }
  .feature-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .dashboard-grid { grid-template-columns: 1fr; }
  .hero-kpis, .deliverables-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


def _base_plotly_layout(title: str) -> Dict[str, Any]:
    return dict(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E7EAF2"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=20, r=20, t=56, b=25),
        transition=dict(duration=380, easing="cubic-in-out"),
    )


def _mini_sparkline(values: pd.Series, color: str = "#ff8a00") -> go.Figure:
    y = pd.to_numeric(values, errors="coerce").ffill().fillna(0).tail(35)
    fig = go.Figure(go.Scatter(x=list(range(len(y))), y=y, mode="lines", line=dict(color=color, width=2.3), fill="tozeroy", fillcolor="rgba(255,138,0,0.14)", hoverinfo="skip"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=52, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _metric_card(title: str, value: str, delta_text: str, delta_positive: bool, spark: pd.Series, color: str) -> None:
    cls = "kpi-pos" if delta_positive else "kpi-neg"
    st.markdown('<div class="glass-card metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-label">{title}</div><div class="kpi-val">{value}</div><div class="{cls}">{delta_text}</div>', unsafe_allow_html=True)
    st.plotly_chart(_mini_sparkline(spark, color), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="micro-note">Наведите на точки графика для деталей.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_upload_block() -> Dict[str, Any]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Загрузка данных</div>', unsafe_allow_html=True)
    st.caption("Поддерживаются универсальный формат (1 CSV) и legacy-формат Olist (3 CSV).")
    load_mode = st.radio("Режим загрузки", ["Universal CSV", "Legacy Olist (3 CSV)"], horizontal=True)

    orders_file = items_file = products_file = reviews_file = None
    universal_file = None

    if load_mode == "Universal CSV":
        universal_file = st.file_uploader("Universal transactions CSV", type=["csv"], key="universal_file")
        st.progress(1.0 if universal_file else 0.0, text=f"Обязательные файлы: {1 if universal_file else 0}/1")
    else:
        orders_file = st.file_uploader("Orders CSV", type=["csv"], key="orders_file")
        items_file = st.file_uploader("Items CSV", type=["csv"], key="items_file")
        products_file = st.file_uploader("Products CSV", type=["csv"], key="products_file")
        reviews_file = st.file_uploader("Reviews CSV (опционально)", type=["csv"], key="reviews_file")
        required_files = [orders_file, items_file, products_file]
        st.progress(sum(x is not None for x in required_files) / 3.0, text=f"Обязательные файлы: {sum(x is not None for x in required_files)}/3")

    raw_for_select = None
    universal_txn = None
    universal_quality = {}
    universal_mapping: Dict[str, Optional[str]] = {}
    orders_col_map: Dict[str, Optional[str]] = {}
    items_col_map: Dict[str, Optional[str]] = {}
    products_col_map: Dict[str, Optional[str]] = {}
    reviews_col_map: Dict[str, Optional[str]] = {}

    if load_mode == "Universal CSV" and universal_file is not None:
        try:
            preview = pd.read_csv(universal_file)
            auto_map = build_auto_mapping(list(preview.columns))
            with st.expander("⚙️ Сопоставление колонок (каноническая схема)", expanded=True):
                for f in CANONICAL_FIELDS:
                    choices = ["<не использовать>"] + list(preview.columns)
                    guessed = auto_map.get(f.name)
                    idx = choices.index(guessed) if guessed in choices else 0
                    selected = st.selectbox(
                        f"{f.name} {'*' if f.required else ''} — {f.description}",
                        choices,
                        index=idx,
                        key=f"map_universal_{f.name}",
                    )
                    universal_mapping[f.name] = None if selected == "<не использовать>" else selected
            universal_txn, universal_quality = normalize_transactions(preview, universal_mapping)
            if universal_quality.get("errors"):
                st.error(" ; ".join(universal_quality["errors"]))
            else:
                st.success(f"Нормализация завершена: {len(universal_txn):,} строк.")
                for w in universal_quality.get("warnings", []):
                    st.warning(w)
                raw_for_select = universal_txn.copy().rename(columns={"category": "product_category_name", "product_id": "product_id"})
        except Exception as e:
            st.error(f"Ошибка предобработки universal CSV: {e}")
    elif load_mode != "Universal CSV" and all([orders_file, items_file, products_file]):
        try:
            orders_preview = pd.read_csv(orders_file)
            items_preview = pd.read_csv(items_file)
            products_preview = pd.read_csv(products_file)
            reviews_preview = pd.read_csv(reviews_file) if reviews_file else pd.DataFrame()

            with st.expander("⚙️ Сопоставление колонок", expanded=False):
                orders_cols, items_cols, products_cols = list(orders_preview.columns), list(items_preview.columns), list(products_preview.columns)
                reviews_cols = list(reviews_preview.columns) if len(reviews_preview) else []

                def pick(label: str, cols: List[str], aliases: List[str], key: str, required: bool = True) -> Optional[str]:
                    guessed = _suggest_column(cols, aliases)
                    choices = cols if required else ["<не использовать>"] + cols
                    idx = choices.index(guessed) if guessed in choices else 0
                    val = st.selectbox(label, choices, index=idx, key=key)
                    return None if (not required and val == "<не использовать>") else val

                orders_col_map = {
                    "order_id": pick("Orders → ID заказа", orders_cols, ["order_id", "orderid", "id_order", "invoice_id"], "map_orders_order_id"),
                    "order_purchase_timestamp": pick("Orders → Дата", orders_cols, ["order_purchase_timestamp", "purchase_timestamp", "order_date", "date"], "map_orders_ts"),
                }
                items_col_map = {
                    "order_id": pick("Items → ID заказа", items_cols, ["order_id", "orderid", "id_order", "invoice_id"], "map_items_order_id"),
                    "product_id": pick("Items → SKU", items_cols, ["product_id", "sku", "item_id", "product_sku"], "map_items_product_id"),
                    "price": pick("Items → Цена", items_cols, ["price", "unit_price", "sale_price"], "map_items_price"),
                    "freight_value": pick("Items → Freight", items_cols, ["freight_value", "shipping_cost", "delivery_cost"], "map_items_freight", required=False),
                    "order_item_id": pick("Items → ID позиции", items_cols, ["order_item_id", "item_line_id", "line_id"], "map_items_line", required=False),
                }
                products_col_map = {
                    "product_id": pick("Products → SKU", products_cols, ["product_id", "sku", "item_id", "product_sku"], "map_products_product_id"),
                    "product_category_name": pick("Products → Категория", products_cols, ["product_category_name", "category", "category_name"], "map_products_category"),
                }
                reviews_col_map = {
                    "order_id": pick("Reviews → ID", reviews_cols, ["order_id", "orderid", "id_order"], "map_reviews_order_id", required=False),
                    "review_score": pick("Reviews → Score", reviews_cols, ["review_score", "rating", "score"], "map_reviews_score", required=False),
                    "review_creation_date": pick("Reviews → Date", reviews_cols, ["review_creation_date", "review_date", "date"], "map_reviews_date", required=False),
                } if len(reviews_preview) else {}

            orders_norm = _rename_with_mapping(orders_preview, orders_col_map)
            items_norm = _rename_with_mapping(items_preview, items_col_map)
            products_norm = _rename_with_mapping(products_preview, products_col_map)
            reviews_norm = _rename_with_mapping(reviews_preview, reviews_col_map) if len(reviews_preview) else pd.DataFrame()
            if "order_item_id" not in items_norm.columns:
                items_norm["order_item_id"] = np.arange(1, len(items_norm) + 1)
            if "freight_value" not in items_norm.columns:
                items_norm["freight_value"] = 0.0

            raw_for_select = build_raw_frame(orders_norm, items_norm, products_norm, reviews_norm)
            st.success(f"Предпросмотр готов: {len(raw_for_select):,} строк")
        except Exception as e:
            st.error(f"Ошибка предобработки: {e}")

    target_category, target_sku = None, None
    if raw_for_select is not None and len(raw_for_select) > 0:
        category_col = "product_category_name" if "product_category_name" in raw_for_select.columns else "category"
        sku_col = "product_id"
        categories = sorted(raw_for_select[category_col].dropna().astype(str).unique())
        target_category = st.selectbox("Категория", categories, key="input_target_category") if categories else None
        sku_search = st.text_input("Поиск SKU", placeholder="Введите SKU")
        if target_category is not None:
            skus_all = raw_for_select[raw_for_select[category_col].astype(str) == str(target_category)][sku_col].astype(str).dropna().unique().tolist()
            skus_filtered = sorted([s for s in skus_all if sku_search.lower() in s.lower()]) if sku_search else sorted(skus_all)
            target_sku = st.selectbox("SKU", skus_filtered, key="input_target_sku") if skus_filtered else None
    run_requested = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    return {
        "orders_file": orders_file,
        "items_file": items_file,
        "products_file": products_file,
        "reviews_file": reviews_file,
        "universal_file": universal_file,
        "universal_txn": universal_txn,
        "universal_quality": universal_quality,
        "universal_mapping": universal_mapping,
        "load_mode": load_mode,
        "target_category": target_category,
        "target_sku": target_sku,
        "run_requested": run_requested,
        "orders_col_map": orders_col_map,
        "items_col_map": items_col_map,
        "products_col_map": products_col_map,
        "reviews_col_map": reviews_col_map,
        "raw_for_select": raw_for_select,
    }


ctx: Dict[str, Any] = {}

if st.session_state.app_stage == "landing" and st.session_state.results is None:
    st.markdown('<div class="mock-shell">', unsafe_allow_html=True)
    st.markdown('<div class="mock-top"><div style="font-weight:700;">SaaS</div><div class="badge-id">0A0A12</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-center"><div class="hero-title">AI What-if Engine</div><div class="hero-sub2">Сценарный прогноз спроса, выручки и прибыли на основе ML и эластичности</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="cta-glow">', unsafe_allow_html=True)
    if st.button("▶ Перейти к загрузке данных", key="hero_start", use_container_width=False, type="primary"):
        st.session_state.app_stage = "upload"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="hero-kpis">
  <div class="hero-kpi"><h4>🧠 Цель системы</h4><p>Построить честный what-if прогноз по выбранной серии и показать эффект сценариев на спрос, выручку и прибыль.</p></div>
  <div class="hero-kpi"><h4>⚡ Скорость принятия решения</h4><p>Полный цикл от загрузки данных до KPI и сценарного анализа занимает несколько минут.</p></div>
  <div class="hero-kpi"><h4>🛡️ Контроль риска</h4><p>Сценарии учитывают штрафы за нестабильность стратегии и неопределённость моделей.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Что делает модель</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="feature-grid">
  <div class="fcard"><h5>🧹 Нормализация данных</h5><p>Автоматическая очистка выбросов, проверка схемы и подготовка стабильных признаков.</p></div>
  <div class="fcard"><h5>📈 ML-прогноз спроса</h5><p>Ансамбль моделей оценивает спрос при разных ценах с учётом сезонности и лагов.</p></div>
  <div class="fcard"><h5>🎯 Сценарная прибыль</h5><p>Оценка прибыльности заданных сценариев и контроль волатильности.</p></div>
  <div class="fcard"><h5>🧪 What-if сценарии</h5><p>Сравнение гипотез в реальном времени без переобучения модели.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:14px;">Ключевые преимущества</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="feature-grid">
  <div class="fcard"><h5>💸 Эффект сценария</h5><p>Сценарные параметры оцениваются на основе прогнозируемого спроса и ценовой эластичности.</p></div>
  <div class="fcard"><h5>🧭 Прозрачные решения</h5><p>Все KPI, графики и отчёт доступны в одном интерфейсе без ручной сборки аналитики.</p></div>
  <div class="fcard"><h5>📦 Готовый отчёт</h5><p>Excel-выгрузка содержит прогнозы, эластичность, метрики качества и кривую прибыли.</p></div>
  <div class="fcard"><h5>🚀 Быстрое внедрение</h5><p>Поддержка разных входных схем CSV и ручного маппинга колонок без доработок кода.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title" style="margin-top:16px;">Этапы работы модели</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="stage-row">
  <div class="stage-item"><div class="stage-dot">1</div><span>Загрузка CSV</span></div>
  <div class="stage-item"><div class="stage-dot">2</div><span>Валидация данных</span></div>
  <div class="stage-item"><div class="stage-dot">3</div><span>Обучение ML</span></div>
  <div class="stage-item"><div class="stage-dot">4</div><span>Сценарный расчёт</span></div>
  <div class="stage-item"><div class="stage-dot">5</div><span>What-if анализ</span></div>
  <div class="stage-item"><div class="stage-dot">6</div><span>Экспорт отчёта</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="deliverables-grid">
  <div>
    <div class="section-title">Используемые технологии</div>
    <div class="tech-list">
      <span class="tech-pill">🐍 Python 3</span><span class="tech-pill">🧠 CatBoost</span><span class="tech-pill">🌲 RandomForest</span><span class="tech-pill">📉 Ridge/Huber</span><span class="tech-pill">📊 Plotly</span><span class="tech-pill">🖥️ Streamlit</span>
    </div>
  </div>
  <div>
    <div class="section-title">Что вы получите после анализа</div>
    <div class="deliver-item"><h5>1) Базовый и as-is прогноз</h5><p>Чёткое разделение нейтрального baseline и текущего сценария управления.</p></div>
    <div class="deliver-item"><h5>2) Прогноз спроса</h5><p>Динамика спроса и фактических продаж с учётом ограничений и сценарных overrides.</p></div>
    <div class="deliver-item"><h5>3) Эластичность и риск</h5><p>Помесячная чувствительность спроса к цене и комментарии для интерпретации.</p></div>
  </div>
</div>
<div class="guide-box">
  <ol>
    <li>Загрузите 3 обязательных CSV (Orders, Items, Products).</li>
    <li>Проверьте маппинг колонок в блоке «Сопоставление колонок».</li>
    <li>Выберите категорию и SKU, затем нажмите «🚀 Запустить анализ».</li>
  </ol>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    ctx = render_upload_block()
else:
    left_toolbar, right_toolbar = st.columns([1, 1])
    with left_toolbar:
        if st.button("🔄 Новая загрузка", use_container_width=True):
            st.session_state.results = None
            st.session_state.what_if_result = None
            st.session_state.scenario_table = None
            st.session_state.sensitivity_df = None
            st.session_state.app_stage = "landing"
            st.rerun()
    with right_toolbar:
        has_results = st.session_state.results is not None
        st.download_button("📥 Скачать полный Excel-отчёт", data=st.session_state.results["excel_buffer"] if has_results else b"", file_name=f"pricing_report_{st.session_state.get('selected_sku_for_results', 'report')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, disabled=not has_results)
        if has_results:
            st.download_button("🧾 run_summary.json", data=st.session_state.results.get("run_summary_json", b""), file_name="run_summary.json", mime="application/json", use_container_width=True)
            st.download_button("📈 holdout_predictions.csv", data=st.session_state.results.get("holdout_predictions_csv", b""), file_name="holdout_predictions.csv", mime="text/csv", use_container_width=True)
            st.download_button("🧪 scenario_daily_output.csv", data=st.session_state.results.get("scenario_daily_output_csv", b""), file_name="scenario_daily_output.csv", mime="text/csv", use_container_width=True)
            st.download_button("🧩 feature_report.csv", data=st.session_state.results.get("feature_report_csv", b""), file_name="feature_report.csv", mime="text/csv", use_container_width=True)

    if st.session_state.results is None:
        ctx = render_upload_block()

if ctx and ctx.get("run_requested"):
    load_mode = ctx.get("load_mode", "Legacy Olist (3 CSV)")
    orders_file = ctx.get("orders_file")
    items_file = ctx.get("items_file")
    products_file = ctx.get("products_file")
    reviews_file = ctx.get("reviews_file")
    target_category = ctx["target_category"]
    target_sku = ctx["target_sku"]
    if load_mode == "Universal CSV" and ctx.get("universal_txn") is None:
        st.error("Загрузите и сопоставьте universal CSV.")
    elif load_mode != "Universal CSV" and not (orders_file and items_file and products_file):
        st.error("Загрузите минимум 3 обязательных файла.")
    elif target_category is None or target_sku is None:
        st.error("Выберите категорию и SKU для анализа.")
    else:
        with st.spinner("Модель обучается…"):
            try:
                if load_mode == "Universal CSV":
                    results = run_full_pricing_analysis_universal(
                        ctx["universal_txn"],
                        target_category,
                        target_sku,
                    )
                else:
                    orders_file.seek(0); items_file.seek(0); products_file.seek(0)
                    if reviews_file: reviews_file.seek(0)
                    orders = pd.read_csv(orders_file)
                    items = pd.read_csv(items_file)
                    products = pd.read_csv(products_file)
                    reviews = pd.read_csv(reviews_file) if reviews_file else pd.DataFrame()
                    orders = _rename_with_mapping(orders, ctx["orders_col_map"]) if ctx["orders_col_map"] else orders
                    items = _rename_with_mapping(items, ctx["items_col_map"]) if ctx["items_col_map"] else items
                    products = _rename_with_mapping(products, ctx["products_col_map"]) if ctx["products_col_map"] else products
                    reviews = _rename_with_mapping(reviews, ctx["reviews_col_map"]) if ctx["reviews_col_map"] and len(reviews) else reviews
                    if "order_item_id" not in items.columns: items["order_item_id"] = np.arange(1, len(items) + 1)
                    if "freight_value" not in items.columns: items["freight_value"] = 0.0
                    results = run_full_pricing_analysis(orders, items, products, reviews, target_category, target_sku)
                st.session_state.results = results
                st.session_state.selected_category_for_results = target_category
                st.session_state.selected_sku_for_results = target_sku
                st.session_state.what_if_result = None
                st.session_state.scenario_table = None
                st.session_state.sensitivity_df = None
                st.session_state.app_stage = "dashboard"
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка анализа: {e}")

if st.session_state.results is not None:
    r = st.session_state.results
    if "history_daily" in r:
        history_daily = r["history_daily"]
        current_forecast = r["as_is_forecast"]
        scenario_forecast = r["scenario_forecast"] if r["scenario_forecast"] is not None else current_forecast
        baseline_forecast = r["neutral_baseline_forecast"]
    else:
        history_daily = r["daily"]
        current_forecast = r["forecast_current"]
        scenario_forecast = r.get("forecast_scenario", r.get("forecast_optimal", current_forecast))
        baseline_forecast = current_forecast
    st.markdown('<div class="section-title">KPI и ключевой вывод</div>', unsafe_allow_html=True)
    st.markdown('<div class="micro-note">Цель интерфейса — показать baseline/as-is и эффект what-if сценариев без ложной оптимизационной ветки.</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    current_price = float(r["current_price"])
    best_price = float(r.get("scenario_price", r.get("current_price", 0.0)))
    delta_pct = ((best_price - current_price) / current_price * 100) if current_price else 0.0

    with k1: _metric_card("Текущая цена", f"₽ {current_price:,.2f}", "База", True, history_daily["price"], "#FF8A00")
    with k2: _metric_card("Scenario price", f"₽ {best_price:,.2f}", f"{delta_pct:+.2f}%", delta_pct >= 0, scenario_forecast["actual_sales"], "#00D4FF")
    with k3:
        abs_lift = float(r["best_profit"] - r["current_profit"])
        _metric_card("Scenario Δ profit", f"{r['profit_lift_pct']:.2f}%", f"≈ ₽ {abs_lift:,.0f}", r["profit_lift_pct"] >= 0, r["profit_curve"]["adjusted_profit"], "#E400FF")
    with k4:
        elast = list(r["elasticity_map"].values())[-1] if len(r["elasticity_map"]) else np.nan
        _metric_card("Эластичность", f"{elast:.2f}" if np.isfinite(elast) else "n/a", "последний месяц", True, history_daily["sales"], "#FFFFFF")

    left, center, right = st.columns([1.35, 1.35, 1.0], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=baseline_forecast["date"], y=baseline_forecast["actual_sales"], name="Neutral baseline", line=dict(color="#FFFFFF", width=2.6)))
        fig_f.add_trace(go.Scatter(x=current_forecast["date"], y=current_forecast["actual_sales"], name="As-is", line=dict(color="#00D4FF", width=2.2)))
        fig_f.add_trace(go.Scatter(x=scenario_forecast["date"], y=scenario_forecast["actual_sales"], name="Scenario", line=dict(color="#FF8A00", width=3)))
        fig_f.update_layout(**_base_plotly_layout("Прогноз спроса на 30 дней"), dragmode="pan")
        st.plotly_chart(fig_f, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)
        st.caption("Сравнение baseline и as-is по фактическим продажам на горизонте.")
        table = current_forecast[["date", "actual_sales"]].rename(columns={"actual_sales":"current"}).merge(scenario_forecast[["date", "actual_sales"]].rename(columns={"actual_sales":"scenario"}), on="date", how="outer")
        table["delta_sales"] = table["scenario"] - table["current"]
        st.dataframe(table, use_container_width=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    with center:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        em = pd.DataFrame(list(r["elasticity_map"].items()), columns=["month", "elasticity"])
        elasticity_mean = float(em["elasticity"].mean()) if len(em) else np.nan
        elasticity_zone = "эластичный спрос" if np.isfinite(elasticity_mean) and elasticity_mean <= -1 else "слабоэластичный спрос"
        fig_e = go.Figure()
        fig_e.add_trace(go.Bar(x=em["month"], y=em["elasticity"], name="Эластичность", marker_color="#E400FF", opacity=.58))
        fig_e.add_trace(go.Scatter(x=em["month"], y=em["elasticity"], mode="lines+markers", name="Тренд", line=dict(color="#00D4FF", width=2.4)))
        if np.isfinite(elasticity_mean):
            fig_e.add_hline(y=elasticity_mean, line_dash="dot", line_color="#FFD166", annotation_text=f"Среднее: {elasticity_mean:.2f}")
        fig_e.add_hrect(y0=-5, y1=-1, fillcolor="rgba(0,212,255,0.08)", line_width=0)
        fig_e.add_hrect(y0=-1, y1=0, fillcolor="rgba(255,138,0,0.08)", line_width=0)
        fig_e.update_layout(**_base_plotly_layout("Эластичность спроса по месяцам"), dragmode="pan")
        st.plotly_chart(fig_e, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)
        st.caption(f"Интерпретация: зона [-5; -1] — эластичный спрос, зона (-1; 0] — слабоэластичный. Сейчас: {elasticity_zone}.")

        fig_p = px.line(r["profit_curve"], x="price", y="adjusted_profit", template="plotly_dark")
        fig_p.update_traces(line_color="#FF8A00", line_width=3)
        fig_p.add_vline(x=r["current_price"], line_dash="dash", line_color="#ffffff", annotation_text="Текущая")
        fig_p.add_vline(x=r.get("scenario_price", r["current_price"]), line_dash="solid", line_color="#00D4FF", annotation_text="Scenario")
        fig_p.update_layout(**_base_plotly_layout("Кривая прибыли"), dragmode="pan")
        st.plotly_chart(fig_p, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)
        st.caption("Вершина кривой — ценовой диапазон с максимумом ожидаемой прибыли.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### What-if: ручной сценарий")
        st.markdown('<div class="micro-note">Сценарий считается в реальном времени на уже обученном бандле, без переобучения.</div>', unsafe_allow_html=True)
        manual_price = st.number_input("Новая цена (₽)", min_value=0.01, value=float(r["current_price"]), step=1.0, key="what_if_price")
        freight_mult = st.slider("Коэффициент freight", 0.5, 1.5, 1.0, 0.05, key="what_if_freight")
        demand_mult = st.slider("Manual shock multiplier", 0.7, 1.3, 1.0, 0.05, key="what_if_demand")
        horizon_days = st.slider("Горизонт прогноза (дней)", 7, 90, int(CONFIG["HORIZON_DAYS_DEFAULT"]), 1)

        if st.button("Пересчитать сценарий", use_container_width=True):
            st.session_state.what_if_result = run_what_if_projection(r["_trained_bundle"], manual_price=float(manual_price), freight_multiplier=float(freight_mult), demand_multiplier=float(demand_mult), horizon_days=int(horizon_days))

        if st.session_state.what_if_result is not None:
            w = st.session_state.what_if_result
            base = current_forecast[["date", "actual_sales"]].rename(columns={"actual_sales": "base"})
            wf = w["daily"][["date", "actual_sales", "profit"]].rename(columns={"actual_sales": "what_if"})
            wt = base.merge(wf, on="date", how="outer").sort_values("date")
            fig_wd = go.Figure()
            fig_wd.add_trace(go.Scatter(x=wt["date"], y=wt["base"], name="Базовый", line=dict(color="#FFFFFF", width=2)))
            fig_wd.add_trace(go.Scatter(x=wt["date"], y=wt["what_if"], name="What-if", line=dict(color="#00D4FF", width=2.8)))
            fig_wd.update_layout(**_base_plotly_layout("Спрос: What-if vs База"), height=260, dragmode="pan")
            st.plotly_chart(fig_wd, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)

            fig_wp = px.line(wf, x="date", y="profit", template="plotly_dark")
            fig_wp.update_traces(line_color="#FF8A00", line_width=2.8)
            fig_wp.update_layout(**_base_plotly_layout("Прогноз прибыли"), height=240, dragmode="pan")
            st.plotly_chart(fig_wp, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)

            delta_d = float(w["demand_total"] - base["base"].sum())
            delta_p = float(w["profit_total"] - r["current_profit"])
            delta_r = float(w["revenue_total"] - current_forecast["revenue"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Δ Спрос", f"{delta_d:+.1f}")
            m2.metric("Δ Прибыль", f"₽ {delta_p:+,.0f}")
            m3.metric("Δ Выручка", f"₽ {delta_r:+,.0f}")
            st.markdown('<div class="micro-note">Положительные дельты показывают улучшение относительно базового сценария.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### What-if панель: baseline + 3 сценария")
    base_price = float(r["current_price"])
    scen_cols = st.columns(4)
    scenario_inputs = []
    default_names = ["Baseline", "Scenario A", "Scenario B", "Scenario C"]
    default_price_mult = [1.0, 1.03, 0.97, 1.08]
    for i, col in enumerate(scen_cols):
        with col:
            st.markdown(f"**{default_names[i]}**")
            p = st.number_input(f"Price x{i}", value=base_price * default_price_mult[i], min_value=0.01, step=0.1, key=f"sc_price_{i}")
            d = st.slider(f"Demand x{i}", 0.7, 1.3, 1.0, 0.05, key=f"sc_demand_{i}")
            f = st.slider(f"Freight x{i}", 0.5, 1.5, 1.0, 0.05, key=f"sc_freight_{i}")
            c = st.slider(f"Cost x{i}", 0.7, 1.3, 1.0, 0.05, key=f"sc_cost_{i}")
            h = st.slider(f"Horizon x{i}", 7, 90, 30, 1, key=f"sc_hor_{i}")
            scenario_inputs.append({"name": default_names[i], "price": p, "demand_multiplier": d, "freight_multiplier": f, "cost_multiplier": c, "discount_multiplier": 1.0, "stock_cap": 0.0, "horizon_days": h, "overrides": {"discount_multiplier": 1.0}})
    if st.button("Сравнить 4 сценария", use_container_width=True):
        st.session_state.scenario_table = run_scenario_set(r["_trained_bundle"], scenario_inputs, run_what_if_projection)
        st.session_state.sensitivity_df = build_sensitivity_grid(r["_trained_bundle"], base_price=base_price, runner=run_what_if_projection)
    if st.session_state.get("scenario_table") is not None:
        sc_df = st.session_state.scenario_table.copy()
        st.dataframe(sc_df, use_container_width=True, height=210)
        fig_sc = px.bar(sc_df, x="scenario", y=["profit", "revenue", "sales"], barmode="group", template="plotly_dark")
        fig_sc.update_layout(**_base_plotly_layout("Сравнение сценариев"))
        st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)
    if st.session_state.get("sensitivity_df") is not None:
        sens = st.session_state.sensitivity_df.copy()
        fig_heat = px.density_heatmap(sens, x="price", y="discount_multiplier", z="profit", nbinsx=14, nbinsy=14, template="plotly_dark", color_continuous_scale="RdYlGn")
        fig_heat.update_layout(**_base_plotly_layout("Sensitivity: price x discount (profit heatmap)"))
        st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["История продаж и цены", "Метрики качества", "Полная таблица прогноза", "Детальная эластичность"])
    with tabs[0]:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=history_daily["date"], y=history_daily["sales"], name="Продажи", line=dict(color="#00D4FF", width=2.6), yaxis="y1"))
        fig_h.add_trace(go.Scatter(x=history_daily["date"], y=history_daily["price"], name="Цена", line=dict(color="#FF8A00", width=2.2), yaxis="y2"))
        fig_h.update_layout(**_base_plotly_layout("История продаж и цены"), yaxis=dict(title="Продажи"), yaxis2=dict(title="Цена", overlaying="y", side="right"), dragmode="pan")
        st.plotly_chart(fig_h, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)
        st.caption("Динамика цены и объёма продаж на одной шкале времени.")
    with tabs[1]:
        metric_df = r["holdout_metrics"].copy()
        st.dataframe(metric_df, use_container_width=True)
        st.caption("MAPE/WAPE/RMSE показывают среднюю ошибку модели на отложенной выборке.")
        if "feature_usage_report" in r:
            st.markdown("**Feature usage diagnostics**")
            st.dataframe(r["feature_usage_report"], use_container_width=True, height=220)
            st.caption("`eligible_for_model` — кандидатность (наличие+вариативность), `listed_in_feature_set` — только включение в шаблон признаков, не причинная важность.")
    with tabs[2]:
        full_table = current_forecast[["date", "actual_sales", "revenue", "profit"]].rename(
            columns={"actual_sales": "sales_current", "revenue": "revenue_current", "profit": "profit_current"}
        ).merge(
            scenario_forecast[["date", "actual_sales", "revenue", "profit"]].rename(
                columns={"actual_sales": "sales_scenario", "revenue": "revenue_scenario", "profit": "profit_scenario"}
            ),
            on="date",
            how="outer",
        )
        full_table["delta_sales"] = full_table["sales_scenario"] - full_table["sales_current"]
        full_table["delta_revenue"] = full_table["revenue_scenario"] - full_table["revenue_current"]
        full_table["delta_profit"] = full_table["profit_scenario"] - full_table["profit_current"]
        st.dataframe(full_table, use_container_width=True, height=320)
    with tabs[3]:
        elast_df = pd.DataFrame(list(r["elasticity_map"].items()), columns=["month", "elasticity"])
        st.dataframe(elast_df, use_container_width=True, height=260)
        fig_ed = px.bar(elast_df, x="month", y="elasticity", template="plotly_dark")
        fig_ed.update_traces(marker_color="#E400FF")
        fig_ed.update_layout(**_base_plotly_layout("Детальная эластичность"), dragmode="pan")
        st.plotly_chart(fig_ed, use_container_width=True, config=PLOTLY_WORKSPACE_CONFIG)

    cta1, cta2 = st.columns(2)
    with cta1:
        st.markdown('<div class="big-cta"><h3>Готово к внедрению</h3></div>', unsafe_allow_html=True)
        if st.button("✅ Применить сценарные параметры", use_container_width=True):
            st.toast("Scenario price отмечена к применению", icon="🎯")
    with cta2:
        if st.button("🔄 Новая загрузка", use_container_width=True, key="new_upload_bottom"):
            st.session_state.results = None
            st.session_state.what_if_result = None
            st.session_state.scenario_table = None
            st.session_state.sensitivity_df = None
            st.session_state.app_stage = "landing"
            st.rerun()

st.caption("AI What-if Engine • Streamlit • Plotly")
