from __future__ import annotations

import copy
import gc
import hashlib
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
from data_schema import CANONICAL_FIELDS, canonical_required_fields
from scenario_engine import run_scenario
from v1_runtime_helpers import (
    build_backend_warning,
    compute_scenario_price_inputs,
    evaluate_net_price_support,
    get_model_backend_status,
    select_weekly_baseline_candidate,
)
from what_if import build_sensitivity_grid, run_scenario_set
from ui.theme import apply_theme

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
ARTIFACT_SCHEMA_VERSION = "v39.1"
APP_GENERATION = "39"


def robust_clean_dirty_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["price", "freight_value", "cost"]:
        if col in df.columns and len(df) > 0:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = np.where(df[col] < 0, np.nan, df[col])
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(safe_median(df["price"], 1.0)).clip(lower=0.01)
    if "freight_value" in df.columns:
        df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(df.get("price", 0.0) * CONFIG["COST_PROXY_RATIO"]).clip(lower=0.0)
    return df


def detect_small_mode_info(df: pd.DataFrame) -> Dict[str, Any]:
    sales = pd.to_numeric(df.get("sales"), errors="coerce").fillna(0.0)
    price = pd.to_numeric(df.get("price"), errors="coerce")

    n_days = int(len(df))
    positive_days = int((sales > 0).sum())
    unique_prices = int(price.dropna().nunique())

    price_ffill = price.ffill()
    price_changes = int(price_ffill.diff().abs().gt(1e-9).sum()) if len(price_ffill) else 0

    price_non_null = price.dropna()
    if len(price_non_null):
        median_price = float(price_non_null.median())
        price_span = float((price_non_null.max() - price_non_null.min()) / max(median_price, 1e-9))
    else:
        price_span = 0.0
    promotion = pd.to_numeric(df.get("promotion", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0)
    promotion_positive_share = float((promotion > 0).mean()) if len(promotion) else 0.0
    discount = pd.to_numeric(df.get("discount", pd.Series(np.zeros(len(df)))), errors="coerce")
    discount_unique_count = int(discount.dropna().nunique())

    reasons = []
    if n_days < 180:
        reasons.append(f"short_history:{n_days}_days")
    if positive_days < 60:
        reasons.append(f"few_positive_days:{positive_days}")
    if unique_prices < 10:
        reasons.append(f"few_unique_prices:{unique_prices}")
    if price_changes < 8:
        reasons.append(f"few_price_changes:{price_changes}")
    if price_span < 0.12:
        reasons.append(f"narrow_price_span:{price_span:.3f}")
    if promotion_positive_share < 0.08:
        reasons.append(f"low_promotion_share:{promotion_positive_share:.3f}")
    if discount_unique_count < 8:
        reasons.append(f"few_discount_points:{discount_unique_count}")

    return {
        "small_mode": bool(len(reasons) > 0),
        "reasons": reasons,
        "n_days": n_days,
        "positive_days": positive_days,
        "unique_prices": unique_prices,
        "price_changes": price_changes,
        "price_span": price_span,
        "promotion_positive_share": promotion_positive_share,
        "discount_unique_count": discount_unique_count,
    }


def safe_median(series: pd.Series, default: float = 0.0) -> float:
    try:
        x = float(pd.Series(series).median())
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def read_uploaded_csv_safely(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Файл не загружен.")
    raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not raw_bytes:
        raise ValueError("Загруженный CSV пустой.")
    parse_errors: List[str] = []
    for enc in ["utf-8-sig", "cp1251", "latin1"]:
        for sep in [None, ",", ";", "\t", "|"]:
            try:
                buf = BytesIO(raw_bytes)
                kwargs: Dict[str, Any] = {"encoding": enc}
                if sep is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs.update({"sep": sep})
                df = pd.read_csv(buf, **kwargs)
                if len(df.columns) <= 1 and sep in (",", ";", "\t", "|"):
                    continue
                return df
            except Exception as e:
                parse_errors.append(f"{enc}/{repr(sep)}: {e}")
    msg = " ; ".join(parse_errors[-4:]) if parse_errors else "не удалось определить разделитель/кодировку"
    raise ValueError(f"Не удалось прочитать CSV ({msg})")


def read_uploaded_table_safely(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Файл не загружен.")
    file_name = str(getattr(uploaded_file, "name", "")).lower()
    if file_name.endswith(".xlsx"):
        raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if not raw_bytes:
            raise ValueError("Загруженный XLSX пустой.")
        try:
            return pd.read_excel(BytesIO(raw_bytes), engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Не удалось прочитать XLSX: {e}") from e
    return read_uploaded_csv_safely(uploaded_file)


def validate_mapping_required_columns(mapping: Dict[str, Optional[str]]) -> List[str]:
    required = canonical_required_fields()
    missing_required = [c for c in required if not mapping.get(c)]
    return missing_required


def build_data_sufficiency_status(txn: pd.DataFrame) -> Dict[str, Any]:
    if txn is None or len(txn) == 0:
        return {
            "status": "poor",
            "label": "poor",
            "message": "После нормализации нет валидных строк: прогноз ненадёжен.",
            "warnings": ["Недостаточно данных для расчёта."],
        }
    date_series = pd.to_datetime(txn.get("date"), errors="coerce")
    valid_dates = date_series.dropna()
    history_days = int((valid_dates.max() - valid_dates.min()).days + 1) if len(valid_dates) else 0
    rows = int(len(txn))
    sku_count = int(txn["product_id"].nunique()) if "product_id" in txn.columns else 0
    if history_days >= 120 and rows >= 250 and sku_count >= 2:
        status = "enough"
        message = "Истории и объёма данных достаточно для v1."
    elif history_days >= 60 and rows >= 100:
        status = "limited"
        message = "Данные ограничены: используйте результат как ориентир, а не как гарантию."
    else:
        status = "poor"
        message = "Данных мало, рекомендации будут нестабильными."
    warnings_local: List[str] = []
    if history_days < 60:
        warnings_local.append("История короче 60 дней.")
    if rows < 100:
        warnings_local.append("Меньше 100 наблюдений.")
    if sku_count < 2:
        warnings_local.append("Только один SKU — сравнительная устойчивость ниже.")
    return {
        "status": status,
        "label": status,
        "message": message,
        "rows": rows,
        "history_days": history_days,
        "sku_count": sku_count,
        "warnings": warnings_local,
    }


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


def fit_feature_stats(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for c in features:
        stats[c] = safe_median(df[c], 0.0) if c in df.columns and len(df) > 0 else 0.0
    return stats


def _safe_git_commit() -> str:
    code_sig = code_signature()
    for env_key in ("GIT_COMMIT", "RENDER_GIT_COMMIT", "COMMIT_SHA"):
        value = str(os.getenv(env_key, "")).strip()
        if value:
            return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return f"nogit:{code_sig}"


def code_signature() -> str:
    hasher = hashlib.sha256()
    for name in ["app.py", "what_if.py", "data_adapter.py", "data_schema.py"]:
        try:
            with open(name, "rb") as f:
                hasher.update(f.read())
        except Exception:
            hasher.update(f"missing:{name}".encode("utf-8"))
    return hasher.hexdigest()[:12]


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


BASELINE_FEATURES: List[str] = [
    "sales_lag1",
    "sales_lag7",
    "sales_lag14",
    "sales_lag28",
    "sales_ma7",
    "sales_ma28",
    "sales_ma90",
    "sales_std28",
    "sales_momentum_7_28",
    "sales_momentum_28_90",
    "dow",
    "month",
    "weekofyear",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "month_sin",
    "month_cos",
    "time_index",
    "time_index_norm",
]

UPLIFT_FEATURES: List[str] = [
    "promotion",
    "freight_value",
    "dow",
    "is_weekend",
    "month_sin",
    "month_cos",
    "time_index_norm",
    "baseline_log_feature",
    "promo_x_weekend",
]

LEGACY_WEEKLY_BASELINE_FEATURES: List[str] = [
    "sales_lag1w",
    "sales_lag2w",
    "sales_lag4w",
    "sales_lag8w",
    "sales_ma4w",
    "sales_ma8w",
    "sales_ma12w",
    "sales_std4w",
    "sales_std8w",
    "week_sin",
    "week_cos",
    "trend_idx",
    "stock_mean",
    "stockout_share",
]
WEEKLY_BASELINE_BUNDLES: Dict[str, List[str]] = {
    "legacy_baseline": LEGACY_WEEKLY_BASELINE_FEATURES,
    "price_only_baseline": LEGACY_WEEKLY_BASELINE_FEATURES + ["price_idx"],
    "price_promo_baseline": LEGACY_WEEKLY_BASELINE_FEATURES + ["price_idx", "promotion_share"],
    "price_promo_freight_baseline": LEGACY_WEEKLY_BASELINE_FEATURES + ["price_idx", "promotion_share", "freight_mean"],
}
WEEKLY_BASELINE_FEATURES: List[str] = list(dict.fromkeys(
    feature
    for bundle_features in WEEKLY_BASELINE_BUNDLES.values()
    for feature in bundle_features
))

WEEKLY_UPLIFT_FEATURES: List[str] = [
    "price_gap_ref_8w",
    "promotion_share",
    "promo_any",
    "freight_pct_change_1w",
]

WEEKLY_MODEL_FEATURES: List[str] = list(dict.fromkeys(WEEKLY_BASELINE_FEATURES + WEEKLY_UPLIFT_FEATURES + ["net_price_mean", "price_ref_8w"]))
WEEKLY_EXOGENOUS_FEATURES: List[str] = ["price_idx", "promotion_share", "freight_mean"]
MONOTONE_MAP: Dict[str, int] = {
    "price_idx": -1,
    "price_gap_ref_8w": -1,
    "promotion_share": 1,
    "freight_mean": -1,
    "freight_pct_change_1w": -1,
    "stock_mean": 1,
    "stockout_share": -1,
}

UPLIFT_MIN_ROWS = 40
WEEKLY_UPLIFT_CLIP_LOW = -0.10
WEEKLY_UPLIFT_CLIP_HIGH = 0.10
LEARNED_UPLIFT_MODE = "diagnostic_only"  # diagnostic_only / gated_active
MIN_UPLIFT_KEEP_FOLD_RATE = 0.40
MAX_SUPPORT_TOO_LOW_FOLD_RATE = 0.50
UPLIFT_MIN_PRICE_WEEKS = 2
UPLIFT_MIN_PROMO_WEEKS = 2
UPLIFT_MIN_FREIGHT_WEEKS = 3
UPLIFT_SIGNIFICANT_PRICE_GAP = 0.01
UPLIFT_SIGNIFICANT_FREIGHT_CHANGE = 0.02
UPLIFT_NEUTRAL_BIAS_THRESHOLD = 0.015
NONLEGACY_BASELINE_MODE = "active_production"
AMPLITUDE_SCALE_GRID: List[float] = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50]


def select_eligible_features(df: pd.DataFrame, candidates: List[str], min_unique: int = 2) -> List[str]:
    keep: List[str] = []
    for c in candidates:
        if c in df.columns and df[c].notna().any() and df[c].nunique(dropna=True) >= min_unique:
            keep.append(c)
    return keep


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
        weight_data = min(0.35 if small_mode else 0.8, len(data) / 500.0)
        coef = weight_data * coef + (1 - weight_data) * CONFIG["PRIOR_ELASTICITY"]
    floor, ceil = (CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
    if small_mode:
        floor, ceil = -2.5, -0.2
    return float(np.clip(coef, floor, ceil))


def compute_monthly_group_elasticities(df_all: pd.DataFrame, pooled_prior: float, small_mode: bool = False) -> Tuple[Dict[str, float], pd.DataFrame]:
    df_all = df_all.copy()
    if "elasticity_bucket" not in df_all.columns:
        df_all["elasticity_bucket"] = df_all["date"].dt.to_period("M").astype(str)
    if small_mode and int(df_all["price"].nunique(dropna=True)) < 10:
        all_buckets = sorted(df_all["elasticity_bucket"].dropna().astype(str).unique().tolist())
        shrunk = {b: float(pooled_prior) for b in all_buckets}
        diag_df = pd.DataFrame([{"bucket": b, "n": int((df_all["elasticity_bucket"] == b).sum()), "price_std": float(df_all.loc[df_all["elasticity_bucket"] == b, "price"].std(ddof=0) or 0.0), "price_unique": int(df_all.loc[df_all["elasticity_bucket"] == b, "price"].nunique()), "raw_slope": float(pooled_prior), "s2": float("nan"), "lambda": 0.0, "shrunk": float(pooled_prior), "note": "small_mode_pooled_only"} for b in all_buckets])
        return shrunk, diag_df
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
        floor, ceil = (CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
        if small_mode:
            floor, ceil = -2.5, -0.2
        val = float(np.clip(val, floor, ceil))
        shrunk[bucket] = val
        lam_map[bucket] = lam

    diag_rows = []
    for row in group_stats:
        b = row["bucket"]
        diag_rows.append({"bucket": b, "n": int(row["n"]), "price_std": float(row["price_std"]), "price_unique": int(row["price_unique"]), "raw_slope": float(raw_slopes[b]), "s2": float(raw_s2[b]), "lambda": float(lam_map[b]), "shrunk": float(shrunk[b]), "note": row["note"]})

    diag_df = pd.DataFrame(diag_rows).sort_values("bucket").reset_index(drop=True)
    return shrunk, diag_df


def _make_uplift_monotone_constraints(feature_names: List[str]) -> List[int]:
    return [MONOTONE_MAP.get(fname, 0) for fname in feature_names]


def build_models(X: pd.DataFrame, y: pd.Series, feature_names: List[str], n_models: int = CONFIG["ENSEMBLE_SIZE"], kind: str = "direct", small_mode: bool = False) -> List[Any]:
    ensemble: List[Any] = []
    if len(X) == 0:
        raise ValueError("Пустая обучающая выборка.")
    if kind == "direct":
        monotone = _make_direct_monotone_constraints(feature_names)
    elif kind == "uplift":
        monotone = _make_uplift_monotone_constraints(feature_names)
    else:
        monotone = [0] * len(feature_names)
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


def predict_baseline_log(frame: pd.DataFrame, models_local: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    return ensemble_predict(models_local, frame[BASELINE_FEATURES].astype(float))


def predict_baseline_log_bundle(frame: pd.DataFrame, baseline_bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    feat = clean_feature_frame(frame.copy(), baseline_bundle["features"], baseline_bundle.get("feature_stats", {}))
    return ensemble_predict(baseline_bundle["models"], feat[baseline_bundle["features"]].astype(float))


def predict_uplift_log_bundle(frame: pd.DataFrame, uplift_bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    feat = clean_feature_frame(frame.copy(), uplift_bundle["features"], uplift_bundle.get("feature_stats", {}))
    return ensemble_predict(uplift_bundle["models"], feat[uplift_bundle["features"]].astype(float))


def calc_observed_price_effect_log(frame: pd.DataFrame, elasticity_map: Dict[str, float], pooled_elasticity: float, ref_net_price: Optional[float] = None) -> np.ndarray:
    price = frame.get("net_unit_price", frame.get("price", pd.Series(np.ones(len(frame))))).astype(float).values
    if ref_net_price is None:
        ref_net_price = safe_median(pd.Series(price), 1.0)
    ratio = np.clip(np.maximum(price, 1e-9) / max(ref_net_price, 1e-9), 0.20, 5.0)
    months = [str(pd.Timestamp(d).to_period("M")) for d in frame["date"]] if "date" in frame.columns else [None] * len(price)
    elasticity = np.array([elasticity_map.get(m, pooled_elasticity) if m is not None else pooled_elasticity for m in months], dtype=float)
    elasticity = np.clip(elasticity, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
    effect = elasticity * np.log(ratio)
    return np.clip(effect, -1.2, 1.2)


def calc_scenario_price_effect_log(current_net_price: float, scenario_net_price: float, future_months: List[str], elasticity_map: Dict[str, float], pooled_elasticity: float) -> np.ndarray:
    ratio = float(np.clip(max(scenario_net_price, 1e-9) / max(current_net_price, 1e-9), 0.20, 5.0))
    elasticity = np.array([elasticity_map.get(m, pooled_elasticity) for m in future_months], dtype=float)
    elasticity = np.clip(elasticity, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
    effect = elasticity * np.log(ratio)
    return np.clip(effect, -1.2, 1.2)


def forecast_future_dates(last_date: pd.Timestamp, n_days: int = CONFIG["HORIZON_DAYS_DEFAULT"]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=n_days, freq="D")})


def current_price_context(base_df: pd.DataFrame) -> Dict[str, Any]:
    return base_df.iloc[-1].to_dict()


def build_weekly_model_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    frame = daily_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["date"] - pd.to_timedelta(frame["date"].dt.dayofweek, unit="D")
    frame["net_unit_price"] = pd.to_numeric(frame.get("net_unit_price", frame.get("price", 0.0)), errors="coerce")
    frame["stock"] = pd.to_numeric(frame.get("stock", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0.0)
    frame["promotion"] = pd.to_numeric(frame.get("promotion", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0.0)
    agg = (
        frame.groupby("week_start", as_index=False)
        .agg(
            observed_days=("date", "nunique"),
            sales=("sales", "sum"),
            revenue=("revenue", "sum"),
            price_mean=("price", "mean"),
            discount_mean=("discount", "mean"),
            net_price_mean=("net_unit_price", "mean"),
            promotion_share=("promotion", "mean"),
            freight_mean=("freight_value", "mean"),
            stock_mean=("stock", "mean"),
            stock_min=("stock", "min"),
            stockout_share=("stock", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) <= 0).mean())),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    agg["weekofyear"] = agg["week_start"].dt.isocalendar().week.astype(int)
    agg["is_full_week"] = (pd.to_numeric(agg["observed_days"], errors="coerce").fillna(0) >= 7).astype(int)
    agg["month"] = agg["week_start"].dt.month.astype(int)
    agg["week_sin"] = np.sin(2 * np.pi * agg["weekofyear"] / 52.0)
    agg["week_cos"] = np.cos(2 * np.pi * agg["weekofyear"] / 52.0)
    agg["month_sin"] = np.sin(2 * np.pi * agg["month"] / 12.0)
    agg["month_cos"] = np.cos(2 * np.pi * agg["month"] / 12.0)
    agg["trend_idx"] = np.arange(len(agg), dtype=float)
    agg["promo_any"] = (pd.to_numeric(agg["promotion_share"], errors="coerce").fillna(0.0) > 0.0).astype(float)
    price_ref = pd.to_numeric(agg["net_price_mean"], errors="coerce").shift(1).rolling(8, min_periods=3).median()
    agg["price_ref_8w"] = price_ref
    agg["price_ref_8w"] = agg["price_ref_8w"].fillna(pd.to_numeric(agg["net_price_mean"], errors="coerce").expanding().median())
    agg["price_ref_8w"] = agg["price_ref_8w"].replace(0.0, np.nan)
    agg["price_ref_8w"] = agg["price_ref_8w"].fillna(max(float(pd.to_numeric(agg["net_price_mean"], errors="coerce").median()), 1e-6))
    agg["price_idx"] = (pd.to_numeric(agg["net_price_mean"], errors="coerce") / agg["price_ref_8w"]).clip(0.5, 1.5)
    return agg


def add_weekly_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    out = weekly_df.copy().sort_values("week_start").reset_index(drop=True)
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").fillna(0.0)
    out["sales_lag1w"] = out["sales"].shift(1)
    out["sales_lag2w"] = out["sales"].shift(2)
    out["sales_lag4w"] = out["sales"].shift(4)
    out["sales_lag8w"] = out["sales"].shift(8)
    out["sales_lag52w"] = out["sales"].shift(52)
    out["sales_ma4w"] = out["sales"].shift(1).rolling(4, min_periods=2).mean()
    out["sales_ma8w"] = out["sales"].shift(1).rolling(8, min_periods=4).mean()
    out["sales_ma12w"] = out["sales"].shift(1).rolling(12, min_periods=6).mean()
    out["sales_std4w"] = out["sales"].shift(1).rolling(4, min_periods=2).std()
    out["sales_std8w"] = out["sales"].shift(1).rolling(8, min_periods=4).std()
    if "price_idx" in out.columns:
        out["price_gap_ref_8w"] = (
            pd.to_numeric(out["price_idx"], errors="coerce") - 1.0
        ).clip(-0.5, 0.5).fillna(0.0)
    if "freight_mean" in out.columns:
        out["freight_pct_change_1w"] = (
            pd.to_numeric(out["freight_mean"], errors="coerce")
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-0.5, 0.5)
        )
    fill_cols = [c for c in ["net_price_mean", "discount_mean", "promotion_share", "promo_any", "freight_mean", "freight_pct_change_1w", "stock_mean", "stockout_share", "price_ref_8w", "price_idx", "price_gap_ref_8w", "week_sin", "week_cos", "trend_idx", "observed_days", "is_full_week"] if c in out.columns]
    for c in fill_cols:
        med = safe_median(out[c], 0.0)
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
    return out


def compute_seasonal_anchor_weight(train_weekly: pd.DataFrame) -> float:
    if "sales_lag52w" not in train_weekly.columns:
        return 0.0
    if "is_full_week" in train_weekly.columns:
        full_train = train_weekly[train_weekly["is_full_week"] >= 1].copy()
    else:
        full_train = train_weekly.copy()
    if len(full_train) < 60:
        return 0.0
    pairs = full_train[["sales", "sales_lag52w"]].dropna()
    if len(pairs) < 12:
        return 0.0
    corr = float(np.corrcoef(pairs["sales"].astype(float), pairs["sales_lag52w"].astype(float))[0, 1]) if len(pairs) > 1 else 0.0
    if not np.isfinite(corr) or corr < 0.20:
        return 0.0
    return float(np.clip(0.15 + 0.20 * corr, 0.15, 0.30))


def apply_seasonal_anchor(pred_core: float, seasonal_anchor: Optional[float], weight: float) -> float:
    anchor = float(seasonal_anchor) if seasonal_anchor is not None and np.isfinite(seasonal_anchor) else float("nan")
    if weight <= 0.0 or not np.isfinite(anchor) or anchor < 0:
        return float(max(0.0, pred_core))
    return float(max(0.0, (1.0 - weight) * float(pred_core) + weight * anchor))


def resolve_scenario_driver_mode(selected_forecaster: str, baseline_has_exog: bool) -> str:
    if selected_forecaster == "weekly_model" and baseline_has_exog:
        return "weekly_ml_exogenous"
    if selected_forecaster == "weekly_model":
        return "weekly_ml_legacy_plus_rule_based_multiplier"
    return "naive_plus_rule_based_multiplier"


def resolve_weekly_driver_mode(selected_forecaster: str, learned_uplift_active: bool, fallback_multiplier_used: bool) -> str:
    if selected_forecaster != "weekly_model":
        return "naive_core_only"
    if learned_uplift_active and not fallback_multiplier_used:
        return "weekly_ml_plus_learned_uplift"
    if fallback_multiplier_used:
        return "weekly_ml_plus_rule_based_multiplier"
    return "weekly_ml_core_only"


def train_weekly_core_model(weekly_train: pd.DataFrame, feature_names: List[str]) -> Any:
    X = weekly_train[feature_names].astype(float)
    y = np.log1p(weekly_train["sales"].astype(float))
    monotone = [MONOTONE_MAP.get(f, 0) for f in feature_names]
    if USE_CATBOOST and CatBoostRegressor is not None:
        model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            monotone_constraints=monotone,
            thread_count=1,
        )
        model.fit(X, y)
        setattr(model, "model_backend", "catboost")
        setattr(model, "backend_reason", "catboost_available")
        return model
    class DeterministicWeeklyModel:
        def predict(self, X_local):
            xdf = pd.DataFrame(X_local).copy()
            if "sales_ma4w" in xdf.columns:
                return np.log1p(pd.to_numeric(xdf["sales_ma4w"], errors="coerce").fillna(0.0).values.clip(min=0.0))
            if "sales_lag1w" in xdf.columns:
                return np.log1p(pd.to_numeric(xdf["sales_lag1w"], errors="coerce").fillna(0.0).values.clip(min=0.0))
            return np.zeros(len(xdf), dtype=float)

    model = DeterministicWeeklyModel()
    setattr(model, "model_backend", "deterministic_fallback")
    setattr(model, "backend_reason", "catboost_unavailable_or_disabled")
    return model


def fit_weekly_uplift_model(weekly_train: pd.DataFrame, baseline_pred_train: np.ndarray, small_mode: bool) -> Dict[str, Any]:
    frame_raw = weekly_train.copy()
    frame = frame_raw.copy()
    frame["baseline_log_feature"] = np.log1p(np.clip(np.asarray(baseline_pred_train, dtype=float), 0.0, None))
    frame["uplift_target"] = np.log1p(np.clip(frame["sales"].astype(float), 0.0, None)) - frame["baseline_log_feature"]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["uplift_target"]).reset_index(drop=True)
    core_driver_features = ["price_gap_ref_8w", "promotion_share", "freight_pct_change_1w"]
    aux_features = ["promo_any"]
    candidate_features = [f for f in core_driver_features + aux_features if f in frame.columns]
    dynamic_core = [
        f for f in core_driver_features
        if f in frame.columns and pd.to_numeric(frame[f], errors="coerce").nunique(dropna=True) > 1
    ]
    dynamic_aux = []
    if "promotion_share" not in dynamic_core and "promo_any" in frame.columns:
        if pd.to_numeric(frame["promo_any"], errors="coerce").nunique(dropna=True) > 1:
            dynamic_aux = ["promo_any"]
    features = dynamic_core + dynamic_aux

    debug_info = {
        "train_rows_raw": int(len(frame_raw)),
        "train_rows_after_target_filter": int(len(frame)),
        "candidate_features": list(candidate_features),
        "dynamic_core_features": list(dynamic_core),
        "dynamic_aux_features": list(dynamic_aux),
        "selected_features_before_clean": list(features),
    }
    signal_info = {
        "train_rows": int(len(frame)),
        "train_rows_raw": int(len(frame_raw)),
        "candidate_features": list(candidate_features),
        "dynamic_core_features": list(dynamic_core),
        "dynamic_aux_features": list(dynamic_aux),
        "price_idx_nunique": int(pd.to_numeric(frame["price_idx"], errors="coerce").nunique(dropna=True)) if "price_idx" in frame.columns else 0,
        "promo_weeks": int(pd.to_numeric(frame["promotion_share"], errors="coerce").fillna(0.0).gt(0).sum()) if "promotion_share" in frame.columns else 0,
        "freight_nunique": int(pd.to_numeric(frame["freight_mean"], errors="coerce").nunique(dropna=True)) if "freight_mean" in frame.columns else 0,
    }
    if len(dynamic_core) == 0:
        return {
            "models": [],
            "features": [],
            "feature_stats": {},
            "disabled": True,
            "reason": "uplift_no_exogenous_features",
            "signal_info": signal_info,
            "debug_info": debug_info,
            "neutral_reference_log": 0.0,
        }
    if len(frame) < UPLIFT_MIN_ROWS:
        return {
            "models": [],
            "features": [],
            "feature_stats": {},
            "disabled": True,
            "reason": "uplift_not_enough_rows",
            "signal_info": signal_info,
            "debug_info": debug_info,
            "neutral_reference_log": 0.0,
        }

    feature_stats = fit_feature_stats(frame, features)
    frame = clean_feature_frame(frame, features, feature_stats)
    final_feature_list = [f for f in features if f in frame.columns]
    debug_info["selected_features_after_clean"] = list(final_feature_list)
    debug_info["feature_nunique_after_clean"] = {
        f: int(pd.to_numeric(frame[f], errors="coerce").nunique(dropna=True))
        for f in final_feature_list
    }
    debug_info["feature_std_after_clean"] = {
        f: float(pd.to_numeric(frame[f], errors="coerce").std(ddof=0))
        for f in final_feature_list
    }
    models = build_models(
        frame[features].astype(float),
        frame["uplift_target"].astype(float),
        features,
        kind="uplift",
        small_mode=small_mode,
    )
    neutral_row = {f: 0.0 for f in features}
    neutral_frame = clean_feature_frame(pd.DataFrame([neutral_row]), features, feature_stats)
    neutral_preds = [float(model.predict(neutral_frame[features].astype(float))[0]) for model in models] if len(models) else [0.0]
    neutral_reference_log = float(np.mean(neutral_preds)) if len(neutral_preds) else 0.0
    return {
        "models": models,
        "features": features,
        "feature_stats": feature_stats,
        "disabled": False,
        "reason": "",
        "signal_info": signal_info,
        "debug_info": debug_info,
        "neutral_reference_log": neutral_reference_log,
    }


def _predict_weekly_from_history(model: Any, history_weekly: pd.DataFrame, feature_names: List[str], scenario_overrides: Dict[str, float], n_steps: int, seasonal_anchor_weight: float = 0.0) -> pd.DataFrame:
    hist = history_weekly.copy().sort_values("week_start").reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for step in range(n_steps):
        next_week = pd.Timestamp(hist["week_start"].iloc[-1]) + pd.Timedelta(days=7)
        row = {
            "week_start": next_week,
            "sales": np.nan,
            "revenue": np.nan,
            "price_mean": float(scenario_overrides["price_mean"]),
            "discount_mean": float(scenario_overrides["discount_mean"]),
            "net_price_mean": float(scenario_overrides["net_price_mean"]),
            "promotion_share": float(scenario_overrides["promotion_share"]),
            "freight_mean": float(scenario_overrides["freight_mean"]),
            "stock_mean": float(scenario_overrides["stock_mean"]),
            "stock_min": float(scenario_overrides["stock_mean"]),
            "stockout_share": float(1.0 if scenario_overrides["stock_mean"] <= 0 else 0.0),
        }
        row["weekofyear"] = int(next_week.isocalendar().week)
        row["month"] = int(next_week.month)
        row["week_sin"] = float(np.sin(2 * np.pi * row["weekofyear"] / 52.0))
        row["week_cos"] = float(np.cos(2 * np.pi * row["weekofyear"] / 52.0))
        row["month_sin"] = float(np.sin(2 * np.pi * row["month"] / 12.0))
        row["month_cos"] = float(np.cos(2 * np.pi * row["month"] / 12.0))
        row["trend_idx"] = float(len(hist))
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        hist = add_weekly_features(hist)
        pred_log = float(model.predict(hist.iloc[[-1]][feature_names].astype(float))[0])
        pred_core = float(np.expm1(pred_log))
        seasonal_anchor = pd.to_numeric(hist.iloc[-1].get("sales_lag52w", np.nan), errors="coerce")
        pred = apply_seasonal_anchor(pred_core, seasonal_anchor, seasonal_anchor_weight)
        hist.loc[hist.index[-1], "sales"] = max(0.0, pred)
        rows.append({"week_start": next_week, "pred_weekly_sales": max(0.0, pred)})
    return pd.DataFrame(rows)


def predict_weekly_holdout_with_actual_exog(model: Any, train_weekly: pd.DataFrame, test_weekly: pd.DataFrame, feature_names: List[str], seasonal_anchor_weight: float = 0.0) -> pd.DataFrame:
    hist = train_weekly.copy().sort_values("week_start").reset_index(drop=True)
    preds: List[Dict[str, Any]] = []
    for _, test_row in test_weekly.sort_values("week_start").iterrows():
        next_row = test_row.to_dict()
        next_row["sales"] = np.nan
        next_row["revenue"] = np.nan
        next_row["trend_idx"] = float(len(hist))
        hist = pd.concat([hist, pd.DataFrame([next_row])], ignore_index=True)
        hist = add_weekly_features(hist)
        pred_log = float(model.predict(hist.iloc[[-1]][feature_names].astype(float))[0])
        pred_core = max(0.0, float(np.expm1(pred_log)))
        seasonal_anchor = pd.to_numeric(hist.iloc[-1].get("sales_lag52w", np.nan), errors="coerce")
        pred = apply_seasonal_anchor(pred_core, seasonal_anchor, seasonal_anchor_weight)
        hist.loc[hist.index[-1], "sales"] = pred
        preds.append({"week_start": pd.Timestamp(test_row["week_start"]), "pred_weekly_sales": pred})
    return pd.DataFrame(preds)


def predict_naive_ma4w_recursive(history_weekly: pd.DataFrame, n_steps: int) -> np.ndarray:
    history_sales = history_weekly["sales"].astype(float).tolist()
    preds: List[float] = []
    for _ in range(n_steps):
        pred = float(np.mean(history_sales[-4:])) if len(history_sales) >= 4 else (float(np.mean(history_sales)) if history_sales else 0.0)
        pred = max(0.0, pred)
        preds.append(pred)
        history_sales.append(pred)
    return np.array(preds, dtype=float)


def apply_fallback_holdout_predictions(holdout_predictions: pd.DataFrame, final_pred_test: np.ndarray) -> pd.DataFrame:
    out = holdout_predictions.copy()
    out["baseline_pred_sales"] = np.asarray(final_pred_test, dtype=float)
    out["final_pred_sales"] = np.asarray(final_pred_test, dtype=float)
    out["uplift_log_pred"] = 0.0
    out["uplift_multiplier"] = 1.0
    out["signed_error"] = out["final_pred_sales"] - out["actual_sales"]
    out["abs_error"] = out["signed_error"].abs()
    denom = out["actual_sales"].abs().replace(0.0, np.nan)
    out["ape"] = (out["abs_error"] / denom) * 100.0
    return out


def uplift_passes_gate(actual: np.ndarray, core_pred: np.ndarray, final_pred: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    wape_core = float(calculate_wape(actual, core_pred))
    wape_final = float(calculate_wape(actual, final_pred))

    corr_core = float(np.corrcoef(actual, core_pred)[0, 1]) if len(actual) > 1 else float("nan")
    corr_final = float(np.corrcoef(actual, final_pred)[0, 1]) if len(actual) > 1 else float("nan")

    std_core = float(np.std(core_pred, ddof=0) / max(np.std(actual, ddof=0), 1e-9)) if len(actual) > 1 else float("nan")
    std_final = float(np.std(final_pred, ddof=0) / max(np.std(actual, ddof=0), 1e-9)) if len(actual) > 1 else float("nan")

    keep = bool(
        np.isfinite(wape_core)
        and np.isfinite(wape_final)
        and (wape_final <= wape_core - 0.2)
        and (not np.isfinite(corr_core) or not np.isfinite(corr_final) or corr_final >= corr_core - 0.02)
        and (not np.isfinite(std_core) or not np.isfinite(std_final) or std_final >= max(std_core * 1.05, 0.35))
    )
    return keep, {
        "wape_core": wape_core,
        "wape_final": wape_final,
        "corr_core": corr_core,
        "corr_final": corr_final,
        "std_ratio_core": std_core,
        "std_ratio_final": std_final,
    }


def calibrate_weekly_baseline_amplitude(
    actual: np.ndarray,
    pred: np.ndarray,
    scale_grid: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scales = scale_grid or AMPLITUDE_SCALE_GRID
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    center = float(np.mean(pred)) if len(pred) else 0.0
    baseline_wape = float(calculate_wape(actual, pred)) if len(pred) else float("nan")
    baseline_corr = float(np.corrcoef(actual, pred)[0, 1]) if len(pred) > 1 else float("nan")
    baseline_std_ratio = float(np.std(pred, ddof=0) / max(np.std(actual, ddof=0), 1e-9)) if len(pred) > 1 else float("nan")
    candidates: List[Dict[str, Any]] = []
    for scale in scales:
        pred_adj = np.clip(center + float(scale) * (pred - center), 0.0, None)
        wape = float(calculate_wape(actual, pred_adj)) if len(pred_adj) else float("nan")
        corr = float(np.corrcoef(actual, pred_adj)[0, 1]) if len(pred_adj) > 1 else float("nan")
        std_ratio = float(np.std(pred_adj, ddof=0) / max(np.std(actual, ddof=0), 1e-9)) if len(pred_adj) > 1 else float("nan")
        std_distance = abs(std_ratio - 1.0) if np.isfinite(std_ratio) else float("inf")
        baseline_std_distance = abs(baseline_std_ratio - 1.0) if np.isfinite(baseline_std_ratio) else float("inf")
        eligible = bool(
            np.isfinite(wape)
            and np.isfinite(corr)
            and np.isfinite(std_ratio)
            and (not np.isfinite(baseline_wape) or wape <= baseline_wape + 0.3)
            and (not np.isfinite(baseline_corr) or corr >= baseline_corr - 0.03)
            and std_distance < baseline_std_distance
        )
        candidates.append(
            {
                "scale": float(scale),
                "wape": wape,
                "corr": corr,
                "std_ratio": std_ratio,
                "std_ratio_distance_to_1": std_distance,
                "eligible": eligible,
            }
        )
    eligible_candidates = [row for row in candidates if row["eligible"]]
    if eligible_candidates:
        best = min(eligible_candidates, key=lambda row: row["wape"])
        best_scale = float(best["scale"])
        pred_final = np.clip(center + best_scale * (pred - center), 0.0, None)
        info = {
            "enabled": bool(abs(best_scale - 1.0) > 1e-9),
            "scale": best_scale,
            "center_mode": "forecast_mean",
            "selection_metrics": {
                "baseline": {
                    "wape": baseline_wape,
                    "corr": baseline_corr,
                    "std_ratio": baseline_std_ratio,
                },
                "candidates": candidates,
                "selected": best,
            },
            "reason": "selected_best_eligible_scale",
        }
        return pred_final, info
    info = {
        "enabled": False,
        "scale": 1.0,
        "center_mode": "forecast_mean",
        "selection_metrics": {
            "baseline": {
                "wape": baseline_wape,
                "corr": baseline_corr,
                "std_ratio": baseline_std_ratio,
            },
            "candidates": candidates,
            "selected": None,
        },
        "reason": "amplitude_calibration_not_helpful",
    }
    return pred.copy(), info


def apply_weekly_amplitude_calibrator(pred: np.ndarray, calibrator_info: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(pred, dtype=float)
    arr = np.clip(arr, 0.0, None)
    if not calibrator_info or not calibrator_info.get("enabled", False):
        return arr
    scale = float(calibrator_info.get("scale", 1.0))
    center_mode = str(calibrator_info.get("center_mode", "forecast_mean"))
    center = float(np.mean(arr)) if center_mode == "forecast_mean" and len(arr) else 0.0
    return np.clip(center + scale * (arr - center), 0.0, None)


def select_amplitude_calibrator_from_train_backtest(
    weekly_train: pd.DataFrame,
    feature_names: List[str],
    n_folds: int = 4,
    horizon_weeks: int = 4,
) -> Dict[str, Any]:
    frame = weekly_train.sort_values("week_start").reset_index(drop=True)
    n = len(frame)
    if n < max(16, horizon_weeks * 3):
        return {
            "enabled": False,
            "scale": 1.0,
            "center_mode": "forecast_mean",
            "selection_metrics": {"folds": [], "selected_scale_source": "insufficient_train_history"},
            "reason": "insufficient_train_history_for_amplitude_backtest",
        }
    min_train = max(12, n - (n_folds * horizon_weeks))
    fold_rows: List[Dict[str, Any]] = []
    fold_scales: List[float] = []
    for i in range(n_folds):
        train_end = min_train + i * horizon_weeks
        test_end = train_end + horizon_weeks
        if test_end > n:
            break
        train_fold = frame.iloc[:train_end].copy()
        test_fold = frame.iloc[train_end:test_end].copy()
        if len(train_fold) < 8 or len(test_fold) == 0:
            continue
        model_fold = train_weekly_core_model(train_fold, feature_names)
        seasonal_anchor_weight_fold = compute_seasonal_anchor_weight(train_fold)
        pred_fold = predict_weekly_holdout_with_actual_exog(
            model_fold,
            train_fold,
            test_fold,
            feature_names,
            seasonal_anchor_weight=seasonal_anchor_weight_fold,
        )
        pred_raw = pred_fold["pred_weekly_sales"].astype(float).values
        actual = test_fold["sales"].astype(float).values
        _, fold_info = calibrate_weekly_baseline_amplitude(actual, pred_raw)
        fold_scale = float(fold_info.get("scale", 1.0))
        fold_scales.append(fold_scale)
        fold_rows.append(
            {
                "fold": int(i + 1),
                "scale": fold_scale,
                "enabled": bool(fold_info.get("enabled", False)),
                "reason": str(fold_info.get("reason", "")),
            }
        )
    if not fold_scales:
        return {
            "enabled": False,
            "scale": 1.0,
            "center_mode": "forecast_mean",
            "selection_metrics": {"folds": fold_rows, "selected_scale_source": "no_valid_folds"},
            "reason": "insufficient_train_history_for_amplitude_backtest",
        }
    median_scale = float(np.median(np.asarray(fold_scales, dtype=float)))
    snapped_scale = float(min(AMPLITUDE_SCALE_GRID, key=lambda x: abs(float(x) - median_scale)))
    enabled = bool(abs(snapped_scale - 1.0) > 1e-9)
    return {
        "enabled": enabled,
        "scale": snapped_scale,
        "center_mode": "forecast_mean",
        "selection_metrics": {
            "folds": fold_rows,
            "median_scale_raw": median_scale,
            "selected_scale_source": "train_backtest_median_scale",
        },
        "reason": "selected_from_train_backtest",
    }


def evaluate_uplift_holdout_support(frame: pd.DataFrame) -> Dict[str, Any]:
    price_gap = pd.to_numeric(frame.get("price_gap_ref_8w", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    promotion = pd.to_numeric(frame.get("promotion_share", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    freight_change = pd.to_numeric(frame.get("freight_pct_change_1w", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    non_neutral_price_weeks = int(price_gap.abs().gt(UPLIFT_SIGNIFICANT_PRICE_GAP).sum()) if len(price_gap) else 0
    promo_weeks = int(promotion.gt(0.0).sum()) if len(promotion) else 0
    freight_change_weeks = int(freight_change.abs().gt(UPLIFT_SIGNIFICANT_FREIGHT_CHANGE).sum()) if len(freight_change) else 0
    support_too_low = bool(
        non_neutral_price_weeks < UPLIFT_MIN_PRICE_WEEKS
        and promo_weeks < UPLIFT_MIN_PROMO_WEEKS
        and freight_change_weeks < UPLIFT_MIN_FREIGHT_WEEKS
    )
    return {
        "non_neutral_price_weeks": non_neutral_price_weeks,
        "promo_weeks": promo_weeks,
        "freight_change_weeks": freight_change_weeks,
        "support_too_low": support_too_low,
        "thresholds": {
            "price_gap_abs_min": float(UPLIFT_SIGNIFICANT_PRICE_GAP),
            "freight_pct_change_abs_min": float(UPLIFT_SIGNIFICANT_FREIGHT_CHANGE),
            "min_price_weeks": int(UPLIFT_MIN_PRICE_WEEKS),
            "min_promo_weeks": int(UPLIFT_MIN_PROMO_WEEKS),
            "min_freight_weeks": int(UPLIFT_MIN_FREIGHT_WEEKS),
        },
    }


def compute_uplift_neutral_bias(frame: pd.DataFrame, uplift_multiplier: np.ndarray) -> Dict[str, Any]:
    price_gap = pd.to_numeric(frame.get("price_gap_ref_8w", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    promotion = pd.to_numeric(frame.get("promotion_share", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    freight_change = pd.to_numeric(frame.get("freight_pct_change_1w", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    neutral_mask = (
        price_gap.abs().le(UPLIFT_SIGNIFICANT_PRICE_GAP)
        & promotion.eq(0.0)
        & freight_change.abs().le(UPLIFT_SIGNIFICANT_FREIGHT_CHANGE)
    )
    neutral_indices = np.asarray(neutral_mask.values if hasattr(neutral_mask, "values") else neutral_mask, dtype=bool)
    if neutral_indices.size == 0 or int(neutral_indices.sum()) == 0:
        return {"neutral_weeks": int(neutral_indices.sum()), "neutral_bias": 0.0, "threshold": float(UPLIFT_NEUTRAL_BIAS_THRESHOLD), "failed": False}
    multipliers = np.asarray(uplift_multiplier, dtype=float)
    neutral_bias = float(np.mean(np.abs(multipliers[neutral_indices] - 1.0)))
    return {
        "neutral_weeks": int(neutral_indices.sum()),
        "neutral_bias": neutral_bias,
        "threshold": float(UPLIFT_NEUTRAL_BIAS_THRESHOLD),
        "failed": bool(neutral_bias > UPLIFT_NEUTRAL_BIAS_THRESHOLD),
    }


def build_uplift_support_snapshot(frame: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(frame))}
    for col in feature_names:
        series = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(dtype=float)
        out[f"{col}_nunique"] = int(series.nunique(dropna=True)) if len(series) else 0
        out[f"{col}_min"] = float(series.min()) if len(series.dropna()) else float("nan")
        out[f"{col}_max"] = float(series.max()) if len(series.dropna()) else float("nan")
    promo = pd.to_numeric(frame["promotion_share"], errors="coerce").fillna(0.0) if "promotion_share" in frame.columns else pd.Series(dtype=float)
    price_gap = pd.to_numeric(frame["price_gap_ref_8w"], errors="coerce").fillna(0.0) if "price_gap_ref_8w" in frame.columns else pd.Series(dtype=float)
    freight_change = pd.to_numeric(frame["freight_pct_change_1w"], errors="coerce").fillna(0.0) if "freight_pct_change_1w" in frame.columns else pd.Series(dtype=float)
    out["raw"] = {
        "promo_weeks": int(promo.gt(1e-6).sum()) if len(promo) else 0,
        "non_neutral_price_weeks": int(price_gap.abs().gt(1e-6).sum()) if len(price_gap) else 0,
        "freight_change_weeks": int(freight_change.abs().gt(1e-6).sum()) if len(freight_change) else 0,
    }
    out["significant"] = {
        "promo_weeks": int(promo.gt(0.0).sum()) if len(promo) else 0,
        "non_neutral_price_weeks": int(price_gap.abs().gt(UPLIFT_SIGNIFICANT_PRICE_GAP).sum()) if len(price_gap) else 0,
        "freight_change_weeks": int(freight_change.abs().gt(UPLIFT_SIGNIFICANT_FREIGHT_CHANGE).sum()) if len(freight_change) else 0,
    }
    out["thresholds"] = {
        "price_gap_abs_min": float(UPLIFT_SIGNIFICANT_PRICE_GAP),
        "freight_pct_change_abs_min": float(UPLIFT_SIGNIFICANT_FREIGHT_CHANGE),
    }
    return out


def decide_uplift_activation(
    activation_mode: str,
    support_info_holdout: Dict[str, Any],
    neutral_bias_info: Dict[str, Any],
    uplift_gate_metrics: Dict[str, Any],
    backtest_summary: Dict[str, Any],
) -> Dict[str, Any]:
    mode_allows_activation = activation_mode == "gated_active"
    holdout_support_ok = not bool(support_info_holdout.get("support_too_low", False))
    neutral_bias_ok = not bool(neutral_bias_info.get("failed", False))
    holdout_gate_ok = bool(uplift_gate_metrics.get("passed", False))
    fold_keep_rate = float(backtest_summary.get("uplift_keep_fold_rate", float("nan")))
    support_too_low_fold_rate = float(backtest_summary.get("support_too_low_fold_rate", float("nan")))
    fold_keep_rate_ok = bool(np.isfinite(fold_keep_rate) and fold_keep_rate >= MIN_UPLIFT_KEEP_FOLD_RATE)
    support_too_low_fold_rate_ok = bool(np.isfinite(support_too_low_fold_rate) and support_too_low_fold_rate <= MAX_SUPPORT_TOO_LOW_FOLD_RATE)
    checks = {
        "mode_allows_activation": bool(mode_allows_activation),
        "holdout_support_ok": bool(holdout_support_ok),
        "neutral_bias_ok": bool(neutral_bias_ok),
        "holdout_gate_ok": bool(holdout_gate_ok),
        "fold_keep_rate_ok": bool(fold_keep_rate_ok),
        "support_too_low_fold_rate_ok": bool(support_too_low_fold_rate_ok),
    }
    if activation_mode == "diagnostic_only":
        return {"active": False, "reason": "diagnostic_only_mode", "checks": checks}
    if activation_mode != "gated_active":
        return {"active": False, "reason": "unknown_activation_mode", "checks": checks}
    if all(checks.values()):
        return {"active": True, "reason": "gated_active_passed", "checks": checks}
    for failed_key, reason in [
        ("holdout_support_ok", "holdout_support_too_low"),
        ("neutral_bias_ok", "uplift_non_neutral_bias"),
        ("holdout_gate_ok", "uplift_holdout_failed"),
        ("fold_keep_rate_ok", "uplift_fold_keep_rate_too_low"),
        ("support_too_low_fold_rate_ok", "support_too_low_fold_rate_too_high"),
    ]:
        if not checks[failed_key]:
            return {"active": False, "reason": reason, "checks": checks}
    return {"active": False, "reason": "uplift_activation_failed", "checks": checks}


def resolve_final_active_path(
    selected_forecaster: str,
    production_selected_candidate: str,
) -> str:
    if selected_forecaster == "weekly_model":
        candidate = str(production_selected_candidate or "legacy_baseline")
        return f"{candidate}+scenario_recompute"
    if selected_forecaster in {"naive_lag1w", "naive_ma4w"}:
        return f"{selected_forecaster}+scenario_recompute"
    return "deterministic_fallback+scenario_recompute"


def allocate_weekly_to_daily(weekly_forecast: pd.DataFrame, daily_history: pd.DataFrame, future_daily_context: pd.DataFrame) -> pd.DataFrame:
    hist = daily_history.sort_values("date").copy().tail(84)
    hist["dow"] = pd.to_datetime(hist["date"]).dt.dayofweek
    dow_sum = hist.groupby("dow")["sales"].sum()
    total = float(dow_sum.sum())
    if total <= 0:
        dow_share = {d: 1.0 / 7.0 for d in range(7)}
    else:
        dow_share = {d: float(dow_sum.get(d, 0.0) / total) for d in range(7)}
        share_total = sum(dow_share.values())
        dow_share = {d: (v / share_total if share_total > 0 else 1.0 / 7.0) for d, v in dow_share.items()}

    out = future_daily_context.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["week_start"] = out["date"] - pd.to_timedelta(out["date"].dt.dayofweek, unit="D")
    out["dow"] = out["date"].dt.dayofweek
    weekly_map = dict(zip(pd.to_datetime(weekly_forecast["week_start"]), weekly_forecast["pred_weekly_sales"]))
    out["weekly_pred_sales"] = out["week_start"].map(weekly_map).fillna(0.0)
    out["share"] = out["dow"].map(dow_share).fillna(1.0 / 7.0)
    if "promotion" in out.columns:
        promo_factor = 1.20
        out["share"] = np.where(pd.to_numeric(out["promotion"], errors="coerce").fillna(0.0) > 0, out["share"] * promo_factor, out["share"])
        week_share_sum = out.groupby("week_start")["share"].transform("sum").replace(0.0, 1.0)
        out["share"] = out["share"] / week_share_sum
    out["pred_sales"] = out["weekly_pred_sales"] * out["share"]
    return out


def evaluate_weekly_backtest(weekly_full: pd.DataFrame, feature_names: List[str], small_mode: bool, n_folds: int = 5, horizon_weeks: int = 4) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    n = len(weekly_full)
    min_train = max(16, int(n * 0.5))
    for i in range(n_folds):
        train_end = min_train + i * horizon_weeks
        test_end = train_end + horizon_weeks
        if test_end > n:
            break
        train = weekly_full.iloc[:train_end].copy()
        test = weekly_full.iloc[train_end:test_end].copy()
        model = train_weekly_core_model(train, feature_names)
        seasonal_anchor_weight = compute_seasonal_anchor_weight(train)
        pred = predict_weekly_holdout_with_actual_exog(model, train, test, feature_names, seasonal_anchor_weight=seasonal_anchor_weight)
        core_pred_raw = pred["pred_weekly_sales"].astype(float).values.copy()
        fold_calibrator_info = select_amplitude_calibrator_from_train_backtest(
            train,
            feature_names,
            n_folds=min(4, max(1, i + 1)),
            horizon_weeks=horizon_weeks,
        )
        core_pred = apply_weekly_amplitude_calibrator(core_pred_raw, fold_calibrator_info)
        baseline_train_raw = np.expm1(model.predict(train[feature_names].astype(float))).clip(min=0.0)
        baseline_train = apply_weekly_amplitude_calibrator(baseline_train_raw, fold_calibrator_info)
        uplift_bundle = fit_weekly_uplift_model(train, baseline_train, small_mode)
        attempted_pred = core_pred.copy()
        active_pred = core_pred.copy()
        uplift_keep_fold = False
        uplift_gate_reason_fold = "not_run"
        support_info_fold = evaluate_uplift_holdout_support(test)
        neutral_bias_info_fold = {"neutral_weeks": 0, "neutral_bias": 0.0, "threshold": float(UPLIFT_NEUTRAL_BIAS_THRESHOLD), "failed": False}
        if len(uplift_bundle.get("models", [])):
            if support_info_fold.get("support_too_low", False):
                uplift_keep_fold = False
                uplift_gate_reason_fold = "holdout_support_too_low"
                active_pred = core_pred.copy()
            else:
                uplift_frame = test.copy()
                uplift_log, _ = predict_uplift_log_bundle(uplift_frame, uplift_bundle)
                uplift_log = np.asarray(uplift_log, dtype=float) - float(uplift_bundle.get("neutral_reference_log", 0.0))
                uplift_log = np.clip(uplift_log, WEEKLY_UPLIFT_CLIP_LOW, WEEKLY_UPLIFT_CLIP_HIGH)
                attempted_pred = np.expm1(np.log1p(np.clip(core_pred, 0.0, None)) + uplift_log).clip(min=0.0)
                active_pred = attempted_pred.copy()
                uplift_multiplier_attempted = attempted_pred / np.clip(core_pred, 1e-9, None)
                neutral_bias_info_fold = compute_uplift_neutral_bias(test, uplift_multiplier_attempted)
                uplift_keep_fold, _ = uplift_passes_gate(test["sales"].astype(float).values, core_pred, attempted_pred)
                if neutral_bias_info_fold.get("failed", False):
                    uplift_keep_fold = False
                    uplift_gate_reason_fold = "uplift_non_neutral_bias"
                elif uplift_keep_fold:
                    uplift_gate_reason_fold = "passed"
                else:
                    uplift_gate_reason_fold = "uplift_holdout_failed"
                if not uplift_keep_fold:
                    active_pred = core_pred.copy()
        else:
            uplift_gate_reason_fold = str(uplift_bundle.get("reason", "no_uplift_model")) or "no_uplift_model"
        rows.append(
            {
                "fold": i + 1,
                "core_weekly_wape": calculate_wape(test["sales"].values, core_pred),
                "attempted_weekly_wape": calculate_wape(test["sales"].values, attempted_pred),
                "active_weekly_wape": calculate_wape(test["sales"].values, active_pred),
                "core_weekly_mae": mean_absolute_error(test["sales"].values, core_pred),
                "attempted_weekly_mae": mean_absolute_error(test["sales"].values, attempted_pred),
                "active_weekly_mae": mean_absolute_error(test["sales"].values, active_pred),
                "final_weekly_wape": calculate_wape(test["sales"].values, active_pred),
                "uplift_keep_fold": bool(uplift_keep_fold),
                "uplift_gate_reason_fold": str(uplift_gate_reason_fold),
                "amplitude_scale": float(fold_calibrator_info.get("scale", 1.0)),
                "support_too_low": bool(support_info_fold.get("support_too_low", False)),
                "neutral_bias": float(neutral_bias_info_fold.get("neutral_bias", 0.0)),
                "neutral_bias_failed": bool(neutral_bias_info_fold.get("failed", False)),
                "weekly_wape": calculate_wape(test["sales"].values, active_pred),
                "weekly_mae": mean_absolute_error(test["sales"].values, active_pred),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "weekly_wape": float(df["weekly_wape"].mean()) if len(df) else float("nan"),
        "weekly_mae": float(df["weekly_mae"].mean()) if len(df) else float("nan"),
        "core_weekly_wape": float(df["core_weekly_wape"].mean()) if len(df) and "core_weekly_wape" in df.columns else float("nan"),
        "core_weekly_mae": float(df["core_weekly_mae"].mean()) if len(df) and "core_weekly_mae" in df.columns else float("nan"),
        "attempted_weekly_wape": float(df["attempted_weekly_wape"].mean()) if len(df) and "attempted_weekly_wape" in df.columns else float("nan"),
        "attempted_weekly_mae": float(df["attempted_weekly_mae"].mean()) if len(df) and "attempted_weekly_mae" in df.columns else float("nan"),
        "active_weekly_wape": float(df["active_weekly_wape"].mean()) if len(df) and "active_weekly_wape" in df.columns else float("nan"),
        "active_weekly_mae": float(df["active_weekly_mae"].mean()) if len(df) and "active_weekly_mae" in df.columns else float("nan"),
        "final_weekly_wape": float(df["final_weekly_wape"].mean()) if len(df) and "final_weekly_wape" in df.columns else float("nan"),
        "uplift_keep_fold_rate": float(df["uplift_keep_fold"].mean()) if len(df) and "uplift_keep_fold" in df.columns else float("nan"),
        "amplitude_scale_median": float(df["amplitude_scale"].median()) if len(df) and "amplitude_scale" in df.columns else float("nan"),
        "support_too_low_fold_rate": float(df["support_too_low"].mean()) if len(df) and "support_too_low" in df.columns else float("nan"),
        "neutral_bias_mean": float(df["neutral_bias"].mean()) if len(df) and "neutral_bias" in df.columns else float("nan"),
    }
    return df, summary


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


def make_walk_forward_oof_baseline(
    train_df: pd.DataFrame,
    features: List[str],
    n_splits: int = 4,
    small_mode: bool = True,
) -> np.ndarray:
    n = len(train_df)
    if n < 20:
        return np.full(n, float(train_df["log_sales"].median()) if "log_sales" in train_df.columns else 0.0, dtype=float)
    oof = np.full(n, np.nan, dtype=float)
    split_points = np.linspace(0.5, 1.0, n_splits + 1)
    for i in range(n_splits):
        train_end = int(n * split_points[i])
        valid_end = int(n * split_points[i + 1])
        if train_end < 8 or valid_end <= train_end:
            continue
        fold_train = train_df.iloc[:train_end].copy()
        fold_valid = train_df.iloc[train_end:valid_end].copy()
        fold_stats = fit_feature_stats(fold_train, features)
        fold_train = clean_feature_frame(fold_train, features, fold_stats)
        fold_valid = clean_feature_frame(fold_valid, features, fold_stats)
        fold_models = build_models(
            fold_train[features].astype(float),
            fold_train["log_sales"].astype(float),
            features,
            kind="baseline",
            small_mode=small_mode,
        )
        pred, _ = ensemble_predict(fold_models, fold_valid[features].astype(float))
        oof[train_end:valid_end] = pred
    fallback = float(train_df["log_sales"].median())
    oof = np.where(np.isfinite(oof), oof, fallback)
    return oof


def recursive_baseline_forecast(base_history: pd.DataFrame, horizon_df: pd.DataFrame, baseline_models: List[Any], base_ctx: Dict[str, Any], feature_names: Optional[List[str]] = None, feature_stats: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    feature_names = feature_names or BASELINE_FEATURES
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
        feat = clean_feature_frame(feat, feature_names, feature_stats)
        X = feat[feature_names].astype(float)
        pred_log_mean, pred_log_std = ensemble_predict(baseline_models, X)
        pred_log_mean = float(pred_log_mean[0]); pred_log_std = float(pred_log_std[0])
        pred_sales = max(0.0, float(np.expm1(pred_log_mean)))
        outputs.append(pd.DataFrame({"date": [current_date], "base_pred_log_sales": [pred_log_mean], "base_pred_sales": [pred_sales], "base_pred_std_log": [pred_log_std], "year_month": [str(current_date.to_period("M"))]}))
        history = pd.concat([history, pd.DataFrame({"date": [current_date], "sales": [pred_sales], "freight_value": [float(base_ctx.get("freight_value", 0.0))], "review_score": [float(base_ctx.get("review_score", 4.5))]})], ignore_index=True)
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


def _monotone_price_multiplier(price_candidate: float, current_price: float, elasticity: float) -> float:
    elasticity = float(np.clip(elasticity, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"]))
    ratio = max(float(price_candidate), 1e-9) / max(float(current_price), 1e-9)
    ratio = float(np.clip(ratio, 0.20, 5.0))
    mult = np.exp(elasticity * np.log(ratio))
    return float(np.clip(mult, 0.15, 3.0))


def simulate_horizon_profit(base_row: Dict[str, Any], price_candidate: float, future_dates_df: pd.DataFrame, baseline_bundle: Dict[str, Any], uplift_bundle: Dict[str, Any], base_history: pd.DataFrame, base_ctx: Dict[str, Any], elasticity_map: Dict[str, float], pooled_elasticity: float, allow_extrapolate: bool = False, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = overrides or {}
    weekly_model = baseline_bundle["model"] if isinstance(baseline_bundle, dict) and "model" in baseline_bundle else None
    feature_names = baseline_bundle.get("features", WEEKLY_BASELINE_FEATURES) if isinstance(baseline_bundle, dict) else WEEKLY_BASELINE_FEATURES
    selected_forecaster = baseline_bundle.get("selected_forecaster", "weekly_model") if isinstance(baseline_bundle, dict) else "weekly_model"
    baseline_features = set(feature_names) if isinstance(feature_names, (list, tuple, set)) else set()
    baseline_has_exog = any(f in baseline_features for f in WEEKLY_EXOGENOUS_FEATURES)
    scenario_driver_mode = resolve_scenario_driver_mode(selected_forecaster, baseline_has_exog)
    seasonal_anchor_weight = float(baseline_bundle.get("seasonal_anchor_weight", 0.0)) if isinstance(baseline_bundle, dict) else 0.0
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
    weekly_history = add_weekly_features(build_weekly_model_frame(base_history))
    weekly_history = weekly_history.dropna(subset=["sales"]).reset_index(drop=True)
    future_dates_local = future_dates_df.copy()
    future_dates_local["date"] = pd.to_datetime(future_dates_local["date"])
    future_dates_local["week_start"] = future_dates_local["date"] - pd.to_timedelta(future_dates_local["date"].dt.dayofweek, unit="D")
    n_weeks = int(max(1, future_dates_local["week_start"].nunique()))
    last_history_week = pd.Timestamp(weekly_history["week_start"].max()) if len(weekly_history) else pd.Timestamp(base_history["date"].max()).floor("D")
    if selected_forecaster == "naive_lag1w":
        naive_val = float(weekly_history["sales"].iloc[-1]) if len(weekly_history) else 0.0
        weekly_pred = pd.DataFrame({"week_start": [last_history_week + pd.Timedelta(days=7 * (i + 1)) for i in range(n_weeks)], "pred_weekly_sales": [naive_val] * n_weeks})
    elif selected_forecaster == "naive_ma4w":
        naive_preds = predict_naive_ma4w_recursive(weekly_history, n_weeks)
        weekly_pred = pd.DataFrame({"week_start": [last_history_week + pd.Timedelta(days=7 * (i + 1)) for i in range(n_weeks)], "pred_weekly_sales": naive_preds})
    elif weekly_model is None:
        naive_val = float(weekly_history["sales"].tail(4).mean()) if len(weekly_history) else 0.0
        weekly_pred = pd.DataFrame({"week_start": [last_history_week + pd.Timedelta(days=7 * (i + 1)) for i in range(n_weeks)], "pred_weekly_sales": [naive_val] * n_weeks})
    else:
        weekly_pred = _predict_weekly_from_history(
            weekly_model,
            weekly_history,
            feature_names,
            {
                "price_mean": float(price_for_model),
                "discount_mean": float(discount_base),
                "net_price_mean": float(scenario_net_price_model),
                "promotion_share": float(promo_val),
                "freight_mean": float(freight_val),
                "stock_mean": float(stock_cap if stock_cap > 0 else safe_median(base_history.get("stock", pd.Series([0.0])), 0.0)),
            },
            n_weeks,
            seasonal_anchor_weight=seasonal_anchor_weight,
        )
    amplitude_calibrator = baseline_bundle.get("amplitude_calibrator", {}) if isinstance(baseline_bundle, dict) else {}
    if selected_forecaster == "weekly_model" and weekly_model is not None:
        weekly_pred["pred_weekly_sales"] = apply_weekly_amplitude_calibrator(
            weekly_pred["pred_weekly_sales"].astype(float).values,
            amplitude_calibrator,
        )
    if len(future_dates_local) and pd.Timestamp(future_dates_local["week_start"].min()) == last_history_week:
        if len(weekly_pred) and pd.Timestamp(weekly_pred["week_start"].min()) > last_history_week:
            hist_tail = base_history.sort_values("date").tail(84).copy()
            hist_tail["dow"] = pd.to_datetime(hist_tail["date"]).dt.dayofweek
            dow_sum = hist_tail.groupby("dow")["sales"].sum()
            dow_total = float(dow_sum.sum())
            dow_share = {d: (float(dow_sum.get(d, 0.0) / dow_total) if dow_total > 0 else 1.0 / 7.0) for d in range(7)}
            remaining_days = future_dates_local[future_dates_local["week_start"] == last_history_week]["date"]
            remaining_share = float(sum(dow_share.get(int(pd.Timestamp(d).dayofweek), 0.0) for d in remaining_days))
            if remaining_share <= 0:
                remaining_share = float(len(remaining_days) / 7.0) if len(remaining_days) else 1.0
            weekly_reference = float(weekly_history["sales"].iloc[-1]) if len(weekly_history) else float(weekly_pred["pred_weekly_sales"].iloc[0])
            promo_current = float(base_ctx.get("promotion", 0.0))
            bridge_mult_price = _monotone_price_multiplier(scenario_net_price_model, current_net_price_model, pooled_elasticity)
            bridge_mult_promo = float(np.clip(1.0 + 0.15 * (float(promo_val) - promo_current), 0.7, 1.3))
            bridge_scenario_mult = float(np.clip(bridge_mult_price * bridge_mult_promo, 0.5, 1.8))
            bridge_val = max(0.0, weekly_reference * remaining_share * bridge_scenario_mult)
            weekly_pred = pd.concat(
                [pd.DataFrame({"week_start": [last_history_week], "pred_weekly_sales": [max(0.0, bridge_val)]}), weekly_pred],
                ignore_index=True,
            )

    future_ctx = future_dates_df[["date"]].copy()
    future_ctx["promotion"] = float(promo_val)
    future_ctx["stock"] = float(stock_cap if stock_cap > 0 else safe_median(base_history.get("stock", pd.Series([np.inf])), np.inf))
    future_ctx["price"] = float(price_for_model)
    future_ctx["discount"] = float(discount_base)
    future_ctx["freight_value"] = float(freight_val)
    future_ctx["net_unit_price"] = (future_ctx["price"] * (1.0 - future_ctx["discount"])).clip(lower=0.01)
    history_tail = weekly_history.sort_values("week_start").tail(12).copy()
    future_weekly_ctx = build_weekly_model_frame(future_ctx.assign(sales=0.0, revenue=0.0))
    combine_cols = [
        "week_start",
        "sales",
        "revenue",
        "observed_days",
        "price_mean",
        "discount_mean",
        "net_price_mean",
        "promotion_share",
        "freight_mean",
        "stock_mean",
        "stock_min",
        "stockout_share",
        "weekofyear",
        "is_full_week",
        "month",
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
        "trend_idx",
        "promo_any",
        "price_ref_8w",
        "price_idx",
    ]
    history_slice = history_tail[[c for c in combine_cols if c in history_tail.columns]].copy()
    future_slice = future_weekly_ctx[[c for c in combine_cols if c in future_weekly_ctx.columns]].copy()
    combined = pd.concat([history_slice, future_slice], ignore_index=True).sort_values("week_start").reset_index(drop=True)
    combined = add_weekly_features(combined).drop_duplicates(subset=["week_start"], keep="last").reset_index(drop=True)
    future_weeks = set(pd.to_datetime(future_weekly_ctx["week_start"]))
    future_weekly = combined[combined["week_start"].isin(future_weeks)].copy()
    future_weekly = future_weekly.merge(weekly_pred.rename(columns={"pred_weekly_sales": "baseline_pred_sales"}), on="week_start", how="left")
    future_weekly["baseline_pred_sales"] = pd.to_numeric(future_weekly["baseline_pred_sales"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weekly_uplift_log = np.zeros(len(future_weekly), dtype=float)
    fallback_multiplier_used = False
    fallback_reason = ""
    weekly_driver_mode = resolve_weekly_driver_mode(
        selected_forecaster,
        False,
        bool(fallback_multiplier_used),
    )
    future_weekly["pred_weekly_sales"] = np.expm1(np.log1p(np.clip(future_weekly["baseline_pred_sales"].values, 0.0, None)) + weekly_uplift_log).clip(min=0.0)
    weekly_pred_baseline = future_weekly[["week_start", "baseline_pred_sales"]].rename(columns={"baseline_pred_sales": "pred_weekly_sales"}).copy()
    weekly_pred_final = future_weekly[["week_start", "pred_weekly_sales"]].copy()
    baseline_daily_raw = allocate_weekly_to_daily(weekly_pred_baseline, base_history, future_ctx)
    final_daily = allocate_weekly_to_daily(weekly_pred_final, base_history, future_ctx)
    base_pred_sales = baseline_daily_raw["pred_sales"].astype(float).values
    pred_sales = final_daily["pred_sales"].astype(float).values * max(manual_shock, 1e-9)
    pred_log_sales = np.log1p(pred_sales.clip(min=0.0))
    pred_std_log = np.full(len(pred_sales), 0.15, dtype=float)
    weekly_uplift_map = dict(zip(pd.to_datetime(future_weekly["week_start"]), weekly_uplift_log)) if len(future_weekly) else {}
    uplift_log = (pd.to_datetime(future_dates_local["week_start"]).map(weekly_uplift_map).fillna(0.0).astype(float).values
                  if len(future_dates_local) else np.zeros(len(pred_sales), dtype=float))
    uplift_std = np.zeros(len(pred_sales), dtype=float)
    price_effect_log = np.zeros(len(pred_sales), dtype=float)
    future_months = [str(pd.Timestamp(d).to_period("M")) for d in future_dates_df["date"]]

    daily = pd.DataFrame({
        "date": future_dates_df["date"].values,
        "price": float(price_for_model),
        "base_pred_sales": base_pred_sales,
        "uplift_log": uplift_log,
        "pred_sales": pred_sales,
        "pred_log_sales": pred_log_sales,
        "pred_std_log": pred_std_log,
        "elasticity": np.array([pooled_elasticity for _ in future_months], dtype=float),
        "price_effect_log": price_effect_log,
        "discount": discount_base,
        "promotion": promo_val,
        "freight_value": freight_val,
        "cost": float(unit_cost),
    })
    daily["net_unit_price"] = (daily["price"] * (1.0 - daily["discount"])).clip(lower=0.01)
    daily["unconstrained_demand"] = daily["pred_sales"].clip(lower=0.0)
    stock_series = pd.to_numeric(base_history.set_index("date").get("stock", pd.Series(dtype=float)), errors="coerce")
    stock_lookup = stock_series.reindex(pd.to_datetime(daily["date"])).fillna(future_ctx.set_index("date").reindex(pd.to_datetime(daily["date"]))["stock"]).values
    if stock_cap > 0:
        stock_lookup = np.minimum(stock_lookup, stock_cap)
    daily["actual_sales"] = np.where(stock_lookup <= 0, 0.0, np.minimum(daily["unconstrained_demand"], stock_lookup))
    daily["lost_sales"] = (daily["unconstrained_demand"] - daily["actual_sales"]).clip(lower=0.0)
    daily["revenue"] = daily["net_unit_price"] * daily["actual_sales"]
    daily["profit"] = (daily["net_unit_price"] - daily["cost"] - daily["freight_value"]) * daily["actual_sales"]

    std_mean = float(np.nanmean(daily["pred_std_log"].values)) if len(daily) else 0.0
    total_demand = float(np.nansum(daily["actual_sales"].values)) if len(daily) else 0.0
    uncertainty_penalty = CONFIG["UNCERTAINTY_MULTIPLIER"] * std_mean * max((price_for_model - unit_cost), 0.0) * max(total_demand, 1.0)
    disagreement_penalty = 0.0
    raw_profit = float(np.nansum(daily["profit"].values))
    adjusted_profit = float(raw_profit - uncertainty_penalty - disagreement_penalty)

    return {
        "requested_price": float(requested_price),
        "price": float(requested_price),
        "price_for_model": float(price_for_model),
        "current_price_raw": float(current_price_raw),
        "current_price_for_model": float(current_price_model),
        "clip_applied": bool(abs(float(requested_price) - float(price_for_model)) > 1e-9),
        "clip_reason": (
            "price_below_train_min_weekly_baseline_clipped"
            if float(requested_price) < float(train_min)
            else "price_above_train_max_weekly_baseline_clipped"
            if float(requested_price) > float(train_max)
            else ""
        ),
        "scenario_price_effect_source": "requested_price_over_current_price_raw",
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
        "price_multiplier": np.ones(len(daily), dtype=float),
        "elasticity_used": daily["elasticity"].values if len(daily) else np.array([]),
        "direct_current_daily": None,
        "baseline_daily": baseline_daily_raw,
        "fallback_multiplier_used": bool(fallback_multiplier_used),
        "fallback_reason": str(fallback_reason),
        "learned_uplift_active": False,
        "scenario_driver_mode": str(scenario_driver_mode),
        "weekly_driver_mode": str(weekly_driver_mode),
        "baseline_has_exogenous_driver": bool(baseline_has_exog),
    }


def run_full_pricing_analysis_universal(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    region: Optional[str] = None,
    channel: Optional[str] = None,
    segment: Optional[str] = None,
):
    txn = normalized_txn.copy()
    raw_columns = set(normalized_txn.columns.tolist())
    if len(txn) == 0:
        raise ValueError("Пустой датасет после нормализации.")
    daily_base = build_daily_from_transactions(txn, target_sku, target_category=target_category, region=region, channel=channel, segment=segment)
    daily_base = robust_clean_dirty_data(daily_base)
    daily_base = build_feature_matrix(daily_base).dropna(subset=["sales", "price"]).reset_index(drop=True)
    if len(daily_base) < 56:
        raise ValueError("Недостаточно данных для weekly модели (нужно минимум 56 дней).")
    daily_columns = set(daily_base.columns.tolist())
    small_mode_info = detect_small_mode_info(daily_base)
    small_mode = bool(small_mode_info["small_mode"])

    weekly_raw = build_weekly_model_frame(daily_base)
    weekly_full = add_weekly_features(weekly_raw)
    weekly_full = weekly_full.dropna(subset=["sales"]).reset_index(drop=True)
    weekly_eval = weekly_full[weekly_full.get("is_full_week", 1) >= 1].copy().reset_index(drop=True)
    if len(weekly_eval) < 16:
        weekly_eval = weekly_full.copy()
    usable = weekly_eval.dropna(subset=["sales_lag1w", "sales_lag2w", "sales_lag4w", "sales_lag8w"]).reset_index(drop=True)
    use_weekly_ml = len(usable) >= 16
    if use_weekly_ml:
        model_frame = usable.copy()
    else:
        model_frame = weekly_eval.copy()

    split = max(4, int(len(model_frame) * 0.8))
    split = min(split, len(model_frame) - 1)
    train_weekly = model_frame.iloc[:split].copy()
    test_weekly = model_frame.iloc[split:].copy()
    baseline_feature_names = select_eligible_features(train_weekly, WEEKLY_BASELINE_BUNDLES["legacy_baseline"])
    if not baseline_feature_names:
        baseline_feature_names = [f for f in WEEKLY_BASELINE_BUNDLES["legacy_baseline"] if f in train_weekly.columns]
    legacy_baseline_feature_names = list(baseline_feature_names)
    seasonal_anchor_weight_train = compute_seasonal_anchor_weight(train_weekly) if use_weekly_ml else 0.0
    weekly_model = None
    uplift_bundle = {"models": [], "features": [], "feature_stats": {}, "disabled": True, "reason": "weekly_core_only", "signal_info": {}, "neutral_reference_log": 0.0}
    uplift_bundle_attempted = dict(uplift_bundle)

    weekly_baseline_candidate_comparison = {
        "selected_candidate": "legacy_baseline",
        "selection_reason": "weekly_ml_not_used",
        "nonlegacy_baseline_mode": NONLEGACY_BASELINE_MODE,
        "selection_rule": {
            "wape_tolerance_pp": 0.5,
            "corr_tolerance_down": 0.05,
            "std_ratio_min_improvement": 0.02,
            "primary_rank_metric": "std_ratio",
            "secondary_rank_metric": "holdout_wape",
            "tertiary_rank_metric": "corr",
        },
        "legacy_reference": {},
        "candidates": [],
    }
    selected_candidate_name = "legacy_baseline"
    amplitude_calibrator_info: Dict[str, Any] = {
        "enabled": False,
        "scale": 1.0,
        "center_mode": "forecast_mean",
        "selection_metrics": {},
        "reason": "not_run",
    }
    legacy_candidate_pred = np.array([], dtype=float)
    price_only_candidate_pred = np.array([], dtype=float)
    price_promo_candidate_pred = np.array([], dtype=float)
    price_promo_freight_candidate_pred = np.array([], dtype=float)
    candidate_predictions: Dict[str, np.ndarray] = {}

    if use_weekly_ml:
        def _bundle_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
            return {
                "holdout_wape": float(calculate_wape(actual, pred)),
                "holdout_mape": float(calculate_mape(actual, pred)),
                "corr": float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 1 else float("nan"),
                "std_ratio": float(np.std(pred, ddof=0) / max(np.std(actual, ddof=0), 1e-9)) if len(actual) > 1 else float("nan"),
                "mean_bias": float((pred - actual).mean()) if len(actual) else float("nan"),
                "mae": float(mean_absolute_error(actual, pred)) if len(actual) else float("nan"),
                "rmse": float(np.sqrt(mean_squared_error(actual, pred))) if len(actual) else float("nan"),
            }

        actual_holdout = test_weekly["sales"].astype(float).values
        bundle_results: List[Dict[str, Any]] = []
        bundle_models: Dict[str, Any] = {}
        bundle_features_selected: Dict[str, List[str]] = {}
        bundle_pred_frames: Dict[str, pd.DataFrame] = {}
        for bundle_name, bundle_candidates in WEEKLY_BASELINE_BUNDLES.items():
            eligible_features = select_eligible_features(train_weekly, bundle_candidates)
            if not eligible_features:
                continue
            bundle_model = train_weekly_core_model(train_weekly, eligible_features)
            bundle_pred = predict_weekly_holdout_with_actual_exog(
                bundle_model,
                train_weekly,
                test_weekly,
                eligible_features,
                seasonal_anchor_weight=seasonal_anchor_weight_train,
            )
            bundle_pred_arr = bundle_pred["pred_weekly_sales"].astype(float).values
            candidate_predictions[bundle_name] = bundle_pred_arr
            candidate_record = {"name": bundle_name, "features": list(eligible_features)}
            candidate_record.update(_bundle_metrics(actual_holdout, bundle_pred_arr))
            bundle_results.append(candidate_record)
            bundle_models[bundle_name] = bundle_model
            bundle_features_selected[bundle_name] = list(eligible_features)
            bundle_pred_frames[bundle_name] = bundle_pred

        legacy_reference = next((row for row in bundle_results if row["name"] == "legacy_baseline"), None)
        if legacy_reference is None and bundle_results:
            legacy_reference = bundle_results[0]
        weekly_baseline_candidate_comparison["legacy_reference"] = dict(legacy_reference) if legacy_reference else {}

        selection_result = select_weekly_baseline_candidate(
            bundle_results=bundle_results,
            bundle_models=bundle_models,
            bundle_features_selected=bundle_features_selected,
            baseline_bundle_name="legacy_baseline",
            nonlegacy_mode=NONLEGACY_BASELINE_MODE,
            wape_tol_pp=0.5,
            corr_tol=0.05,
            std_ratio_floor=0.02,
            std_ratio_cap=0.80,
        )
        diagnostic_selected_candidate_name = str(selection_result["selected_candidate_name"])
        weekly_baseline_candidate_comparison = selection_result["comparison_payload"]
        selected_candidate_name = "legacy_baseline"
        selected_bundle = next((row for row in bundle_results if str(row.get("name")) == selected_candidate_name), None)
        if selected_bundle is not None and selected_candidate_name in bundle_models:
            weekly_model = bundle_models[selected_candidate_name]
            baseline_feature_names = bundle_features_selected[selected_candidate_name]
            test_pred_weekly = bundle_pred_frames[selected_candidate_name]
        else:
            weekly_model = train_weekly_core_model(train_weekly, baseline_feature_names)
            test_pred_weekly = predict_weekly_holdout_with_actual_exog(
                weekly_model,
                train_weekly,
                test_weekly,
                baseline_feature_names,
                seasonal_anchor_weight=seasonal_anchor_weight_train,
            )
        weekly_baseline_candidate_comparison["production_selected_candidate"] = "legacy_baseline"
        weekly_baseline_candidate_comparison["diagnostic_selected_candidate"] = diagnostic_selected_candidate_name
        weekly_baseline_candidate_comparison["selection_mode"] = "diagnostic_comparison_runtime_frozen_to_legacy"
        amplitude_calibrator_info = select_amplitude_calibrator_from_train_backtest(
            train_weekly,
            baseline_feature_names,
            n_folds=4,
            horizon_weeks=4,
        )

        legacy_candidate_pred = candidate_predictions.get("legacy_baseline", test_pred_weekly["pred_weekly_sales"].astype(float).values.copy())
        price_only_candidate_pred = candidate_predictions.get("price_only_baseline", legacy_candidate_pred.copy())
        price_promo_candidate_pred = candidate_predictions.get("price_promo_baseline", legacy_candidate_pred.copy())
        price_promo_freight_candidate_pred = candidate_predictions.get("price_promo_freight_baseline", legacy_candidate_pred.copy())
        baseline_pred_train_raw = np.expm1(weekly_model.predict(train_weekly[baseline_feature_names].astype(float))).clip(min=0.0)
        final_pred_test_raw = test_pred_weekly["pred_weekly_sales"].astype(float).values.copy()
        final_pred_test = apply_weekly_amplitude_calibrator(final_pred_test_raw, amplitude_calibrator_info)
        baseline_pred_train = apply_weekly_amplitude_calibrator(baseline_pred_train_raw, amplitude_calibrator_info)
        uplift_bundle_attempted = fit_weekly_uplift_model(train_weekly, baseline_pred_train, small_mode)
        uplift_bundle_attempted.setdefault("debug_info", {})
        uplift_bundle_attempted["debug_info"]["baseline_train_calibrated"] = True
        uplift_bundle_attempted["debug_info"]["baseline_train_mean"] = float(np.mean(baseline_pred_train)) if len(baseline_pred_train) else 0.0
        uplift_bundle_attempted["debug_info"]["baseline_train_raw_mean"] = float(np.mean(baseline_pred_train_raw)) if len(baseline_pred_train_raw) else 0.0
        uplift_bundle = dict(uplift_bundle_attempted)
        test_pred_weekly["pred_weekly_sales"] = final_pred_test.copy()
        core_pred_test = test_pred_weekly["pred_weekly_sales"].astype(float).values.copy()
        uplift_multiplier = np.ones(len(final_pred_test), dtype=float)
        uplift_log_pred = np.zeros(len(final_pred_test), dtype=float)
        uplift_log_raw = np.zeros(len(final_pred_test), dtype=float)
        final_pred_attempted = final_pred_test.copy()
        uplift_multiplier_attempted = uplift_multiplier.copy()
        uplift_log_clipped_attempted = uplift_log_pred.copy()
        uplift_log_raw_attempted = uplift_log_raw.copy()
        if len(uplift_bundle.get("models", [])):
            uplift_frame = test_weekly.copy()
            uplift_log_raw, _ = predict_uplift_log_bundle(uplift_frame, uplift_bundle)
            neutral_reference = float(uplift_bundle.get("neutral_reference_log", 0.0))
            uplift_log_raw = np.asarray(uplift_log_raw, dtype=float) - neutral_reference
            uplift_log_pred = np.clip(uplift_log_raw, WEEKLY_UPLIFT_CLIP_LOW, WEEKLY_UPLIFT_CLIP_HIGH)
            final_pred_test = np.expm1(np.log1p(np.clip(final_pred_test, 0.0, None)) + uplift_log_pred).clip(min=0.0)
            uplift_multiplier = final_pred_test / np.clip(test_pred_weekly["pred_weekly_sales"].astype(float).values, 1e-9, None)
            final_pred_attempted = final_pred_test.copy()
            uplift_multiplier_attempted = uplift_multiplier.copy()
            uplift_log_clipped_attempted = uplift_log_pred.copy()
            uplift_log_raw_attempted = uplift_log_raw.copy()
    else:
        naive_fallback = predict_naive_ma4w_recursive(train_weekly, len(test_weekly))
        test_pred_weekly = pd.DataFrame({"week_start": test_weekly["week_start"].values, "pred_weekly_sales": naive_fallback})
        final_pred_test = naive_fallback.copy()
        core_pred_test = naive_fallback.copy()
        legacy_candidate_pred = final_pred_test.copy()
        price_only_candidate_pred = final_pred_test.copy()
        price_promo_candidate_pred = final_pred_test.copy()
        price_promo_freight_candidate_pred = final_pred_test.copy()
        uplift_multiplier = np.ones(len(final_pred_test), dtype=float)
        uplift_log_pred = np.zeros(len(final_pred_test), dtype=float)
        uplift_log_raw = np.zeros(len(final_pred_test), dtype=float)
        final_pred_attempted = final_pred_test.copy()
        uplift_multiplier_attempted = uplift_multiplier.copy()
        uplift_log_clipped_attempted = uplift_log_pred.copy()
        uplift_log_raw_attempted = uplift_log_raw.copy()
        uplift_bundle_attempted = dict(uplift_bundle)
        amplitude_calibrator_info = {
            "enabled": False,
            "scale": 1.0,
            "center_mode": "forecast_mean",
            "selection_metrics": {},
            "reason": "weekly_ml_not_used",
        }
        fallback_metrics = {
            "holdout_wape": float(calculate_wape(test_weekly["sales"].values, legacy_candidate_pred)) if len(test_weekly) else float("nan"),
            "holdout_mape": float(calculate_mape(test_weekly["sales"].values, legacy_candidate_pred)) if len(test_weekly) else float("nan"),
            "corr": float(np.corrcoef(test_weekly["sales"].values, legacy_candidate_pred)[0, 1]) if len(test_weekly) > 1 else float("nan"),
            "std_ratio": float(np.std(legacy_candidate_pred, ddof=0) / max(np.std(test_weekly["sales"].values, ddof=0), 1e-9)) if len(test_weekly) > 1 else float("nan"),
            "mean_bias": float((legacy_candidate_pred - test_weekly["sales"].values).mean()) if len(test_weekly) else float("nan"),
            "mae": float(mean_absolute_error(test_weekly["sales"].values, legacy_candidate_pred)) if len(test_weekly) else float("nan"),
            "rmse": float(np.sqrt(mean_squared_error(test_weekly["sales"].values, legacy_candidate_pred))) if len(test_weekly) else float("nan"),
        }
        weekly_baseline_candidate_comparison["legacy_reference"] = {"name": "legacy_baseline", "features": list(legacy_baseline_feature_names), **fallback_metrics}
        weekly_baseline_candidate_comparison["candidates"] = [
            {"name": "legacy_baseline", "features": list(legacy_baseline_feature_names), **fallback_metrics, "eligible_under_selection_rule": True, "rejection_reason": None},
            {"name": "price_only_baseline", "features": list(legacy_baseline_feature_names), **fallback_metrics, "eligible_under_selection_rule": False, "rejection_reason": "weekly_ml_not_used"},
            {"name": "price_promo_baseline", "features": list(legacy_baseline_feature_names), **fallback_metrics, "eligible_under_selection_rule": False, "rejection_reason": "weekly_ml_not_used"},
            {"name": "price_promo_freight_baseline", "features": list(legacy_baseline_feature_names), **fallback_metrics, "eligible_under_selection_rule": False, "rejection_reason": "weekly_ml_not_used"},
        ]
    weekly_baseline_candidate_comparison["selected_candidate"] = selected_candidate_name
    support_snapshot_train = build_uplift_support_snapshot(train_weekly, ["price_gap_ref_8w", "promotion_share", "freight_pct_change_1w"])
    support_snapshot_holdout = build_uplift_support_snapshot(test_weekly, ["price_gap_ref_8w", "promotion_share", "freight_pct_change_1w"])
    uplift_enabled_holdout = bool(len(uplift_bundle_attempted.get("models", [])) > 0)
    wape_core_holdout = float(calculate_wape(test_weekly["sales"].values, test_pred_weekly["pred_weekly_sales"].values)) if len(test_weekly) else float("nan")
    wape_attempted_holdout = float(calculate_wape(test_weekly["sales"].values, final_pred_attempted)) if len(test_weekly) else float("nan")
    uplift_gate_diagnostics = {
        "wape_core": wape_core_holdout,
        "wape_attempted": wape_attempted_holdout,
        "corr_core": float("nan"),
        "corr_attempted": float("nan"),
        "std_ratio_core": float("nan"),
        "std_ratio_attempted": float("nan"),
    }
    uplift_keep = False
    uplift_gate_result = "not_run"
    holdout_support = evaluate_uplift_holdout_support(test_weekly) if uplift_enabled_holdout else {
        "non_neutral_price_weeks": 0, "promo_weeks": 0, "freight_change_weeks": 0, "support_too_low": True, "thresholds": {}
    }
    neutral_bias_info = {"neutral_weeks": 0, "neutral_bias": 0.0, "threshold": float(UPLIFT_NEUTRAL_BIAS_THRESHOLD), "failed": False}
    uplift_gate_reason = str(uplift_bundle_attempted.get("reason", ""))
    uplift_gate_passed = False
    if uplift_enabled_holdout:
        uplift_gate_diagnostics["holdout_support"] = holdout_support
        if holdout_support.get("support_too_low", False):
            uplift_keep = False
            uplift_gate_result = "failed"
            uplift_gate_reason = "holdout_support_too_low"
        else:
            uplift_keep, uplift_gate_diagnostics = uplift_passes_gate(
                test_weekly["sales"].astype(float).values,
                test_pred_weekly["pred_weekly_sales"].astype(float).values,
                final_pred_attempted.astype(float),
            )
            uplift_gate_diagnostics["holdout_support"] = holdout_support
            neutral_bias_info = compute_uplift_neutral_bias(test_weekly, uplift_multiplier_attempted)
            uplift_gate_diagnostics["neutral_bias"] = neutral_bias_info
            if uplift_keep and neutral_bias_info.get("failed", False):
                uplift_keep = False
                uplift_gate_result = "failed"
                uplift_gate_reason = "uplift_non_neutral_bias"
            elif uplift_keep:
                uplift_gate_result = "passed"
                uplift_gate_reason = "passed"
                uplift_gate_passed = True
            else:
                uplift_gate_result = "failed"
                uplift_gate_reason = "uplift_holdout_failed"
    if not uplift_enabled_holdout:
        uplift_gate_reason = str(uplift_bundle_attempted.get("reason", "no_uplift_model")) or "no_uplift_model"
    _, backtest_summary = evaluate_weekly_backtest(usable, baseline_feature_names, small_mode, n_folds=5, horizon_weeks=4) if use_weekly_ml else (pd.DataFrame(), {"weekly_wape": float("nan"), "weekly_mae": float("nan")})
    uplift_gate_metrics = {"result": uplift_gate_result, "reason": uplift_gate_reason, "passed": bool(uplift_gate_passed)}
    uplift_activation = decide_uplift_activation(
        activation_mode=LEARNED_UPLIFT_MODE,
        support_info_holdout=holdout_support,
        neutral_bias_info=neutral_bias_info,
        uplift_gate_metrics=uplift_gate_metrics,
        backtest_summary=backtest_summary if isinstance(backtest_summary, dict) else {},
    )
    uplift_activation = {
        "active": False,
        "reason": "production_runtime_disabled_diagnostic_only",
        "checks": dict(uplift_activation.get("checks", {})),
        "diagnostic_gate_result": dict(uplift_gate_metrics),
    }
    uplift_keep = False
    if uplift_keep:
        uplift_bundle = dict(uplift_bundle_attempted)
        final_pred_test = np.asarray(final_pred_attempted, dtype=float)
        uplift_multiplier = np.asarray(uplift_multiplier_attempted, dtype=float)
        uplift_log_pred = np.asarray(uplift_log_clipped_attempted, dtype=float)
        uplift_log_raw = np.asarray(uplift_log_raw_attempted, dtype=float)
    else:
        final_pred_test = test_pred_weekly["pred_weekly_sales"].astype(float).values.copy()
        uplift_multiplier = np.ones(len(final_pred_test), dtype=float)
        uplift_log_pred = np.zeros(len(final_pred_test), dtype=float)
        uplift_log_raw = np.zeros(len(final_pred_test), dtype=float)
        uplift_bundle = {
            "models": [],
            "features": [],
            "feature_stats": {},
            "disabled": True,
            "reason": str(uplift_gate_reason) if str(uplift_gate_reason) else "uplift_holdout_failed",
            "signal_info": dict(uplift_bundle_attempted.get("signal_info", {})),
            "debug_info": dict(uplift_bundle_attempted.get("debug_info", {})),
            "neutral_reference_log": float(uplift_bundle_attempted.get("neutral_reference_log", 0.0)),
        }
    holdout_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(test_weekly["sales"].values, final_pred_test))),
        "mae": float(mean_absolute_error(test_weekly["sales"].values, final_pred_test)),
        "mape": float(calculate_mape(test_weekly["sales"].values, final_pred_test)),
        "smape": float(calculate_smape(test_weekly["sales"].values, final_pred_test)),
        "wape": float(calculate_wape(test_weekly["sales"].values, final_pred_test)),
        "sigma_log": float("nan"),
    }
    holdout_predictions = pd.DataFrame({
        "date": pd.to_datetime(test_weekly["week_start"]).dt.strftime("%Y-%m-%d"),
        "series_id": str(target_sku),
        "actual_sales": test_weekly["sales"].values,
        "is_full_week": test_weekly.get("is_full_week", pd.Series(np.ones(len(test_weekly), dtype=int))).values,
        "baseline_pred_sales": test_pred_weekly["pred_weekly_sales"].values,
        "price_effect_multiplier": 1.0,
        "uplift_multiplier": uplift_multiplier,
        "final_pred_sales": final_pred_test,
        "uplift_log_pred": uplift_log_pred,
    })
    holdout_predictions["abs_error"] = np.abs(holdout_predictions["actual_sales"] - holdout_predictions["final_pred_sales"])
    holdout_predictions["signed_error"] = holdout_predictions["final_pred_sales"] - holdout_predictions["actual_sales"]

    naive_lag_val = float(train_weekly["sales"].iloc[-1]) if len(train_weekly) else 0.0
    naive_lag = np.full(len(test_weekly), naive_lag_val, dtype=float)
    naive_ma4 = predict_naive_ma4w_recursive(train_weekly, len(test_weekly))
    best_naive_wape = float(min(calculate_wape(test_weekly["sales"].values, naive_lag), calculate_wape(test_weekly["sales"].values, naive_ma4)))
    corr_final = float(np.corrcoef(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"])[0, 1]) if len(holdout_predictions) > 1 else float("nan")
    std_ratio_final = float(np.std(holdout_predictions["final_pred_sales"], ddof=0) / max(np.std(holdout_predictions["actual_sales"], ddof=0), 1e-9)) if len(holdout_predictions) > 1 else float("nan")
    shape_quality_low = bool(
        (not np.isfinite(corr_final) or corr_final < 0.45)
        or (not np.isfinite(std_ratio_final) or std_ratio_final < 0.40)
    )
    model_enabled = bool(use_weekly_ml)
    lag1_wape = float(calculate_wape(test_weekly["sales"].values, naive_lag))
    ma4_wape = float(calculate_wape(test_weekly["sales"].values, naive_ma4))
    if model_enabled:
        selected_forecaster = "weekly_model"
    else:
        selected_forecaster = "naive_lag1w" if lag1_wape <= ma4_wape else "naive_ma4w"
        weekly_model = None
        amplitude_calibrator_info = {
            "enabled": False,
            "scale": 1.0,
            "center_mode": "forecast_mean",
            "selection_metrics": {},
            "reason": "benchmark_gate_failed",
        }
        uplift_bundle = {
            "models": [],
            "features": [],
            "feature_stats": {},
            "disabled": True,
            "reason": "benchmark_gate_failed",
            "signal_info": dict(uplift_bundle.get("signal_info", {})),
            "neutral_reference_log": 0.0,
        }
        fallback_weekly_pred = naive_lag if selected_forecaster == "naive_lag1w" else naive_ma4
        final_pred_test = np.asarray(fallback_weekly_pred, dtype=float)
        holdout_predictions = apply_fallback_holdout_predictions(holdout_predictions, final_pred_test)
        holdout_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(test_weekly["sales"].values, final_pred_test))),
            "mae": float(mean_absolute_error(test_weekly["sales"].values, final_pred_test)),
            "mape": float(calculate_mape(test_weekly["sales"].values, final_pred_test)),
            "smape": float(calculate_smape(test_weekly["sales"].values, final_pred_test)),
            "wape": float(calculate_wape(test_weekly["sales"].values, final_pred_test)),
            "sigma_log": float("nan"),
        }

    corr_final = float(np.corrcoef(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"])[0, 1]) if len(holdout_predictions) > 1 else float("nan")
    std_ratio_final = float(np.std(holdout_predictions["final_pred_sales"], ddof=0) / max(np.std(holdout_predictions["actual_sales"], ddof=0), 1e-9)) if len(holdout_predictions) > 1 else float("nan")
    corr_baseline = float(np.corrcoef(holdout_predictions["actual_sales"], holdout_predictions["baseline_pred_sales"])[0, 1]) if len(holdout_predictions) > 1 else float("nan")
    std_ratio_baseline = float(np.std(holdout_predictions["baseline_pred_sales"], ddof=0) / max(np.std(holdout_predictions["actual_sales"], ddof=0), 1e-9)) if len(holdout_predictions) > 1 else float("nan")
    shape_quality_low = bool(
        (not np.isfinite(corr_final) or corr_final < 0.45)
        or (not np.isfinite(std_ratio_final) or std_ratio_final < 0.40)
    )

    zero_mask = holdout_predictions["actual_sales"] <= 0
    pos_mask = holdout_predictions["actual_sales"] > 0

    mean_pred_on_zero_days = (
        float(holdout_predictions.loc[zero_mask, "final_pred_sales"].mean())
        if zero_mask.any() else float("nan")
    )

    false_positive_rate_on_zero_days = (
        float((holdout_predictions.loc[zero_mask, "final_pred_sales"] > 0.1).mean())
        if zero_mask.any() else float("nan")
    )

    wape_positive_days = (
        float(calculate_wape(
            holdout_predictions.loc[pos_mask, "actual_sales"],
            holdout_predictions.loc[pos_mask, "final_pred_sales"],
        ))
        if pos_mask.any() else float("nan")
    )
    holdout_weekly_diagnostics = pd.DataFrame({
        "week_start": pd.to_datetime(test_weekly["week_start"]).dt.strftime("%Y-%m-%d"),
        "actual_sales": test_weekly["sales"].astype(float).values,
        "legacy_pred_sales": np.asarray(legacy_candidate_pred, dtype=float),
        "price_only_pred_sales": np.asarray(price_only_candidate_pred, dtype=float),
        "price_promo_pred_sales": np.asarray(price_promo_candidate_pred, dtype=float),
        "price_promo_freight_pred_sales": np.asarray(price_promo_freight_candidate_pred, dtype=float),
        "selected_pred_sales": holdout_predictions["baseline_pred_sales"].astype(float).values,
        "naive_pred_sales": np.asarray(naive_ma4, dtype=float),
        "final_pred_sales": holdout_predictions["final_pred_sales"].astype(float).values,
    })
    holdout_weekly_diagnostics["legacy_error"] = holdout_weekly_diagnostics["legacy_pred_sales"] - holdout_weekly_diagnostics["actual_sales"]
    holdout_weekly_diagnostics["price_only_error"] = holdout_weekly_diagnostics["price_only_pred_sales"] - holdout_weekly_diagnostics["actual_sales"]
    holdout_weekly_diagnostics["price_promo_error"] = holdout_weekly_diagnostics["price_promo_pred_sales"] - holdout_weekly_diagnostics["actual_sales"]
    holdout_weekly_diagnostics["price_promo_freight_error"] = holdout_weekly_diagnostics["price_promo_freight_pred_sales"] - holdout_weekly_diagnostics["actual_sales"]
    holdout_weekly_diagnostics["selected_error"] = holdout_weekly_diagnostics["selected_pred_sales"] - holdout_weekly_diagnostics["actual_sales"]
    holdout_weekly_diagnostics["final_error"] = holdout_weekly_diagnostics["final_pred_sales"] - holdout_weekly_diagnostics["actual_sales"]
    holdout_weekly_diagnostics["actual_wow_change"] = holdout_weekly_diagnostics["actual_sales"].pct_change().replace([np.inf, -np.inf], np.nan)
    holdout_weekly_diagnostics["legacy_wow_change"] = holdout_weekly_diagnostics["legacy_pred_sales"].pct_change().replace([np.inf, -np.inf], np.nan)
    holdout_weekly_diagnostics["price_only_wow_change"] = holdout_weekly_diagnostics["price_only_pred_sales"].pct_change().replace([np.inf, -np.inf], np.nan)
    holdout_weekly_diagnostics["price_promo_wow_change"] = holdout_weekly_diagnostics["price_promo_pred_sales"].pct_change().replace([np.inf, -np.inf], np.nan)
    holdout_weekly_diagnostics["price_promo_freight_wow_change"] = holdout_weekly_diagnostics["price_promo_freight_pred_sales"].pct_change().replace([np.inf, -np.inf], np.nan)
    holdout_weekly_diagnostics["selected_wow_change"] = holdout_weekly_diagnostics["selected_pred_sales"].pct_change().replace([np.inf, -np.inf], np.nan)
    for col in ["price_idx", "price_gap_ref_8w", "discount_mean", "promotion_share", "promo_any", "freight_mean", "freight_pct_change_1w", "stockout_share"]:
        if col in test_weekly.columns:
            holdout_weekly_diagnostics[col] = pd.to_numeric(test_weekly[col], errors="coerce").values
        else:
            holdout_weekly_diagnostics[col] = np.nan
    uplift_trace_columns = ["price_idx", "price_gap_ref_8w", "promotion_share", "promo_any", "freight_mean", "freight_pct_change_1w", "stockout_share"]
    uplift_holdout_trace = pd.DataFrame({
        "week_start": pd.to_datetime(test_weekly["week_start"]).dt.strftime("%Y-%m-%d"),
        "actual_sales": test_weekly["sales"].astype(float).values,
        "core_pred": np.asarray(core_pred_test, dtype=float),
        "uplift_log_raw_attempted": np.asarray(uplift_log_raw_attempted, dtype=float),
        "uplift_log_clipped_attempted": np.asarray(uplift_log_clipped_attempted, dtype=float),
        "uplift_multiplier_attempted": np.asarray(uplift_multiplier_attempted, dtype=float),
        "final_pred_attempted": np.asarray(final_pred_attempted, dtype=float),
        "uplift_log_raw_active": np.asarray(uplift_log_raw, dtype=float),
        "uplift_log_clipped_active": np.asarray(uplift_log_pred, dtype=float),
        "uplift_multiplier_active": np.asarray(uplift_multiplier, dtype=float),
        "final_pred_active": np.asarray(final_pred_test, dtype=float),
    })
    for col in uplift_trace_columns:
        uplift_holdout_trace[col] = pd.to_numeric(test_weekly[col], errors="coerce").values if col in test_weekly.columns else np.nan

    uplift_debug_rows: List[Dict[str, Any]] = []
    attempted_features_set = set(uplift_bundle_attempted.get("features", []))
    active_features_set = set(uplift_bundle.get("features", []))
    for feature_name in uplift_trace_columns:
        train_series = pd.to_numeric(train_weekly[feature_name], errors="coerce") if feature_name in train_weekly.columns else pd.Series(dtype=float)
        holdout_series = pd.to_numeric(test_weekly[feature_name], errors="coerce") if feature_name in test_weekly.columns else pd.Series(dtype=float)
        dropped_reason = ""
        if feature_name not in attempted_features_set:
            dropped_reason = "not_selected_for_attempted_uplift"
        elif feature_name not in active_features_set:
            dropped_reason = str(uplift_bundle.get("reason", "")) or "dropped_after_gate"
        uplift_debug_rows.append(
            {
                "feature_name": feature_name,
                "present_in_train_weekly": bool(feature_name in train_weekly.columns),
                "present_in_holdout_weekly": bool(feature_name in test_weekly.columns),
                "non_null_share_train": float(train_series.notna().mean()) if len(train_series) else 0.0,
                "non_null_share_holdout": float(holdout_series.notna().mean()) if len(holdout_series) else 0.0,
                "nunique_train": int(train_series.nunique(dropna=True)) if len(train_series) else 0,
                "nunique_holdout": int(holdout_series.nunique(dropna=True)) if len(holdout_series) else 0,
                "min_train": float(train_series.min()) if len(train_series.dropna()) else float("nan"),
                "max_train": float(train_series.max()) if len(train_series.dropna()) else float("nan"),
                "min_holdout": float(holdout_series.min()) if len(holdout_series.dropna()) else float("nan"),
                "max_holdout": float(holdout_series.max()) if len(holdout_series.dropna()) else float("nan"),
                "std_train": float(train_series.std(ddof=0)) if len(train_series.dropna()) else float("nan"),
                "std_holdout": float(holdout_series.std(ddof=0)) if len(holdout_series.dropna()) else float("nan"),
                "selected_for_attempted_uplift": bool(feature_name in attempted_features_set),
                "selected_for_active_uplift": bool(feature_name in active_features_set),
                "dropped_reason": dropped_reason,
            }
        )
    uplift_debug_report = pd.DataFrame(uplift_debug_rows)

    warnings = []
    partial_weeks_excluded = int(len(weekly_full) - len(weekly_eval))
    if partial_weeks_excluded > 0:
        warnings.append(f"Из weekly-оценки исключено неполных недель: {partial_weeks_excluded}.")
    if small_mode:
        warnings.append(
            "Датасет нестабилен для свободного обучения модели. "
            f"Причины: {', '.join(small_mode_info['reasons'])}"
        )
        warnings.append("Для блока эластичности включён weak-signal weekly fallback.")
    wape_val = holdout_metrics.get("wape", float("nan"))
    if pd.notna(wape_val) and float(wape_val) > 40:
        warnings.append(f"Высокий holdout WAPE: {float(wape_val):.1f}%")
    if not model_enabled:
        warnings.append(f"Weekly ML не прошла benchmark gate, используем {selected_forecaster}.")
    elif shape_quality_low or (np.isfinite(std_ratio_final) and float(std_ratio_final) < 0.5):
        warnings.append("Forecast shape is flatter than actual demand; use scenario results as directional, not exact.")
    if uplift_enabled_holdout and not uplift_keep:
        warnings.append(f"Learned uplift откатили: {str(uplift_gate_reason) or 'uplift_gate_failed'}.")
    if not use_weekly_ml:
        warnings.append("Недостаточно weekly history для ML: используем deterministic naive forecaster.")
    full_weekly = usable.copy()
    seasonal_anchor_weight_full = seasonal_anchor_weight_train
    if model_enabled and use_weekly_ml:
        weekly_model = train_weekly_core_model(full_weekly, baseline_feature_names)
        seasonal_anchor_weight_full = compute_seasonal_anchor_weight(full_weekly)
        full_baseline_pred_raw = np.expm1(weekly_model.predict(full_weekly[baseline_feature_names].astype(float))).clip(min=0.0)
        full_baseline_pred = apply_weekly_amplitude_calibrator(full_baseline_pred_raw, amplitude_calibrator_info)
        if uplift_keep:
            uplift_bundle = fit_weekly_uplift_model(full_weekly, full_baseline_pred, small_mode)
        else:
            uplift_bundle = {
                "models": [],
                "features": [],
                "feature_stats": {},
                "disabled": True,
                "reason": str(uplift_gate_reason) if uplift_gate_reason else "uplift_holdout_failed",
                "signal_info": dict(uplift_bundle.get("signal_info", {})),
                "neutral_reference_log": float(uplift_bundle.get("neutral_reference_log", 0.0)),
            }
    baseline_bundle = {
        "model": weekly_model,
        "features": baseline_feature_names,
        "mode": "weekly_baseline_core",
        "selected_forecaster": selected_forecaster,
        "selected_candidate": selected_candidate_name,
        "seasonal_anchor_weight": float(seasonal_anchor_weight_full),
        "amplitude_calibrator": dict(amplitude_calibrator_info),
    }
    backend_status = get_model_backend_status(weekly_model)
    baseline_bundle["model_backend"] = backend_status["model_backend"]
    baseline_bundle["backend_reason"] = backend_status["backend_reason"]
    backend_warning_text = build_backend_warning(backend_status["model_backend"], backend_status["backend_reason"])
    if backend_warning_text:
        warnings.append(backend_warning_text)
    baseline_has_exog = any(f in set(baseline_feature_names) for f in WEEKLY_EXOGENOUS_FEATURES)
    fixed_log_price_coef = CONFIG["PRIOR_ELASTICITY"]
    shrunk_random_effects = {}
    ref_net_price = safe_median(daily_base.get("net_unit_price", daily_base["price"]), safe_median(daily_base["price"], 1.0))

    base_ctx = current_price_context(daily_base)
    base_ctx["category"] = target_category
    base_ctx["product_id"] = target_sku
    latest_row = dict(base_ctx)
    latest_row["requested_price"] = float(base_ctx.get("price", safe_median(daily_base["price"], 1.0)))
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()))
    baseline_price = float(safe_median(daily_base["price"], float(base_ctx.get("price", 1.0))))
    as_is_sim = simulate_horizon_profit(latest_row, float(base_ctx.get("price")), future_dates, baseline_bundle, uplift_bundle, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef)
    neutral_overrides = {"discount": 0.0, "promotion": 0.0, "freight_value": float(safe_median(daily_base.get("freight_value", pd.Series([0.0])), 0.0))}
    baseline_sim = simulate_horizon_profit(latest_row, baseline_price, future_dates, baseline_bundle, uplift_bundle, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, overrides=neutral_overrides)
    scenario_sim = None
    scenario_forecast = scenario_sim["daily"] if scenario_sim else None
    confidence = float(1.0 / (1.0 + max(0.0, holdout_metrics.get("wape", 100.0) / 100.0)))
    holdout_core_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(test_weekly["sales"].values, test_pred_weekly["pred_weekly_sales"].values))),
        "mae": float(mean_absolute_error(test_weekly["sales"].values, test_pred_weekly["pred_weekly_sales"].values)),
        "mape": float(calculate_mape(test_weekly["sales"].values, test_pred_weekly["pred_weekly_sales"].values)),
        "smape": float(calculate_smape(test_weekly["sales"].values, test_pred_weekly["pred_weekly_sales"].values)),
        "wape": float(calculate_wape(test_weekly["sales"].values, test_pred_weekly["pred_weekly_sales"].values)),
    } if len(test_weekly) else {}
    attempted_uplift_metrics = {
        "holdout_wape": float(calculate_wape(test_weekly["sales"].values, final_pred_attempted)) if len(test_weekly) else float("nan"),
        "holdout_mae": float(mean_absolute_error(test_weekly["sales"].values, final_pred_attempted)) if len(test_weekly) else float("nan"),
        "holdout_rmse": float(np.sqrt(mean_squared_error(test_weekly["sales"].values, final_pred_attempted))) if len(test_weekly) else float("nan"),
        "active_possible": bool(uplift_enabled_holdout),
        "model_count": int(len(uplift_bundle_attempted.get("models", []))),
    }
    active_uplift_metrics = {
        "holdout_wape": float(calculate_wape(test_weekly["sales"].values, final_pred_test)) if len(test_weekly) else float("nan"),
        "holdout_mae": float(mean_absolute_error(test_weekly["sales"].values, final_pred_test)) if len(test_weekly) else float("nan"),
        "holdout_rmse": float(np.sqrt(mean_squared_error(test_weekly["sales"].values, final_pred_test))) if len(test_weekly) else float("nan"),
        "active": bool(uplift_activation.get("active", False)),
        "reason": str(uplift_activation.get("reason", "")),
        "model_count": int(len(uplift_bundle.get("models", []))),
    }
    attempted_vs_active_delta = {
        "holdout_wape_delta": float(active_uplift_metrics["holdout_wape"] - attempted_uplift_metrics["holdout_wape"]) if np.isfinite(active_uplift_metrics["holdout_wape"]) and np.isfinite(attempted_uplift_metrics["holdout_wape"]) else float("nan"),
        "holdout_mae_delta": float(active_uplift_metrics["holdout_mae"] - attempted_uplift_metrics["holdout_mae"]) if np.isfinite(active_uplift_metrics["holdout_mae"]) and np.isfinite(attempted_uplift_metrics["holdout_mae"]) else float("nan"),
        "holdout_rmse_delta": float(active_uplift_metrics["holdout_rmse"] - attempted_uplift_metrics["holdout_rmse"]) if np.isfinite(active_uplift_metrics["holdout_rmse"]) and np.isfinite(attempted_uplift_metrics["holdout_rmse"]) else float("nan"),
    }

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        daily_base.to_excel(writer, sheet_name="history", index=False)
        baseline_sim["daily"].to_excel(writer, sheet_name="neutral_baseline", index=False)
        as_is_sim["daily"].to_excel(writer, sheet_name="as_is", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="metrics", index=False)
    excel_buffer.seek(0)
    delta_vs_as_is = {
        "demand_total": float((scenario_sim["daily"]["actual_sales"].sum() - as_is_sim["daily"]["actual_sales"].sum())) if scenario_sim else float("nan"),
        "revenue_total": float((scenario_sim["daily"]["revenue"].sum() - as_is_sim["daily"]["revenue"].sum())) if scenario_sim else float("nan"),
        "profit_total": float((scenario_sim["daily"]["profit"].sum() - as_is_sim["daily"]["profit"].sum())) if scenario_sim else float("nan"),
    }
    delta_vs_baseline = {
        "demand_total": float(as_is_sim["daily"]["actual_sales"].sum() - baseline_sim["daily"]["actual_sales"].sum()),
        "revenue_total": float(as_is_sim["daily"]["revenue"].sum() - baseline_sim["daily"]["revenue"].sum()),
        "profit_total": float(as_is_sim["daily"]["profit"].sum() - baseline_sim["daily"]["profit"].sum()),
    }
    analysis_baseline_vs_as_is = pd.DataFrame({
        "date": pd.to_datetime(as_is_sim["daily"]["date"]).dt.strftime("%Y-%m-%d"),
        "series_id": str(target_sku),
        "baseline_demand": baseline_sim["daily"]["actual_sales"].values,
        "as_is_demand": as_is_sim["daily"]["actual_sales"].values,
        "baseline_revenue": baseline_sim["daily"]["revenue"].values,
        "as_is_revenue": as_is_sim["daily"]["revenue"].values,
        "baseline_profit": baseline_sim["daily"]["profit"].values,
        "as_is_profit": as_is_sim["daily"]["profit"].values,
    })
    feature_usage_rows = []
    report_features = list(dict.fromkeys(BASELINE_FEATURES + WEEKLY_MODEL_FEATURES + ["price", "discount", "cost", "stock", "freight_value", "promotion", "baseline_log_feature"]))
    engineered_features = {
        "sales_lag1", "sales_lag7", "sales_lag14",
        "sales_ma7", "sales_ma14", "sales_ma28",
        "sales_std28",
        "day_of_week", "month", "is_weekend",
        "month_sin", "month_cos",
        "time_index", "time_index_norm",
        "promo_x_weekend", "baseline_log_feature",
        "log_sales", "uplift_target_log", "net_unit_price",
    }
    for f in report_features:
        found_in_raw = f in raw_columns
        present_in_daily = f in daily_columns
        present_in_weekly = f in weekly_full.columns
        daily_missing_share = float(daily_base[f].isna().mean()) if present_in_daily else 1.0
        daily_unique_count = int(daily_base[f].nunique(dropna=True)) if present_in_daily else 0
        weekly_missing_share = float(weekly_full[f].isna().mean()) if present_in_weekly else 1.0
        weekly_unique_count = int(weekly_full[f].nunique(dropna=True)) if present_in_weekly else 0
        engineered_feature = f in engineered_features
        model_generated_feature = f in {"baseline_log_feature"}
        used_in_baseline = f in baseline_feature_names
        used_in_uplift_attempted = f in uplift_bundle_attempted.get("features", [])
        used_in_uplift_active = bool(len(uplift_bundle.get("models", [])) > 0) and (f in uplift_bundle.get("features", []))
        used_in_final_active_forecast = bool(used_in_baseline or used_in_uplift_active)
        if used_in_uplift_active:
            active_usage_reason = "active_learned_uplift"
        elif used_in_uplift_attempted and not used_in_uplift_active:
            active_usage_reason = "uplift_deactivated_by_gate"
        elif used_in_baseline:
            active_usage_reason = "active_baseline_feature"
        else:
            active_usage_reason = "not_used_in_active_path"
        used_in_uplift = used_in_uplift_active
        used_in_weekly_baseline = used_in_baseline
        used_in_weekly_uplift = used_in_uplift
        used_in_weekly_model = used_in_weekly_baseline or used_in_weekly_uplift
        used_in_price_effect = False
        used_only_in_economics = f in {"cost"}
        used_in_model = used_in_baseline or used_in_uplift or used_in_price_effect
        scenario_features = {
            "price", "discount", "promotion", "freight_value", "cost",
            "price_idx", "price_gap_ref_8w", "promotion_share", "promo_any",
            "freight_mean", "freight_pct_change_1w", "stockout_share", "baseline_log_feature",
        }
        used_in_scenario = f in scenario_features

        if used_in_weekly_baseline or used_in_weekly_uplift:
            reason_excluded = ""
        elif (not present_in_daily) and (not present_in_weekly):
            reason_excluded = "missing"
        elif (present_in_daily and daily_unique_count <= 1) or (present_in_weekly and weekly_unique_count <= 1):
            reason_excluded = "constant"
        elif f in set(WEEKLY_UPLIFT_FEATURES) and present_in_weekly and bool(uplift_bundle.get("disabled", False)):
            reason_excluded = "weekly_feature_available_but_uplift_disabled"
        elif f in {"discount_mean", "promo_any"} and present_in_weekly and not used_in_weekly_model:
            reason_excluded = "excluded_from_weekly_baseline_bundle_policy"
        elif present_in_weekly and not used_in_weekly_model:
            reason_excluded = "not_selected_in_active_weekly_model"
        elif not used_in_model:
            reason_excluded = "not_in_active_model"
        else:
            reason_excluded = ""

        feature_usage_rows.append(
            {
                "factor_name": f,
                "found_in_raw": bool(found_in_raw),
                "present_in_daily": bool(present_in_daily),
                "present_in_weekly": bool(present_in_weekly),
                "engineered_feature": bool(engineered_feature),
                "model_generated_feature": bool(model_generated_feature),
                "daily_missing_share": daily_missing_share,
                "daily_unique_count": daily_unique_count,
                "weekly_missing_share": weekly_missing_share,
                "weekly_unique_count": weekly_unique_count,
                "used_in_baseline": bool(used_in_baseline),
                "used_in_uplift": bool(used_in_uplift),
                "used_in_uplift_attempted": bool(used_in_uplift_attempted),
                "used_in_uplift_active": bool(used_in_uplift_active),
                "used_in_attempted_uplift": bool(used_in_uplift_attempted),
                "used_in_active_uplift": bool(used_in_uplift_active),
                "used_in_active_baseline": bool(used_in_baseline),
                "used_in_final_active_forecast": bool(used_in_final_active_forecast),
                "active_usage_reason": active_usage_reason,
                "used_in_weekly_baseline": bool(used_in_weekly_baseline),
                "used_in_weekly_uplift": bool(used_in_weekly_uplift),
                "used_in_weekly_uplift_attempted": bool(used_in_uplift_attempted),
                "used_in_weekly_uplift_active": bool(used_in_uplift_active),
                "weekly_uplift_attempted_reason": "" if used_in_uplift_attempted else "not_selected_for_attempted_uplift",
                "weekly_uplift_active_reason": "" if used_in_uplift_active else (str(uplift_bundle.get("reason", "")) if used_in_uplift_attempted else "not_selected_for_attempted_uplift"),
                "used_in_weekly_model": bool(used_in_weekly_model),
                "used_in_price_effect": bool(used_in_price_effect),
                "used_only_in_economics": bool(used_only_in_economics),
                "used_in_model": bool(used_in_model),
                "used_in_scenario": bool(used_in_scenario),
                "reason_excluded": reason_excluded,
            }
        )

    feature_report = pd.DataFrame(feature_usage_rows)
    active_model_factors = sorted(
        feature_report.loc[
            feature_report["used_in_final_active_forecast"].astype(bool),
            "factor_name",
        ].astype(str).unique().tolist()
    ) if len(feature_report) else []
    scenario_only_factors = sorted(
        feature_report.loc[
            feature_report["used_in_scenario"].astype(bool)
            & (~feature_report["used_in_final_active_forecast"].astype(bool)),
            "factor_name",
        ].astype(str).unique().tolist()
    ) if len(feature_report) else []
    if "manual_shock_multiplier" not in scenario_only_factors:
        scenario_only_factors.append("manual_shock_multiplier")
    attempted_but_disabled_factors = sorted(
        feature_report.loc[
            feature_report["used_in_weekly_uplift_attempted"].astype(bool)
            & (~feature_report["used_in_weekly_uplift_active"].astype(bool))
            & (~feature_report["used_in_final_active_forecast"].astype(bool)),
            "factor_name",
        ].astype(str).unique().tolist()
    ) if len(feature_report) else []
    candidate_feature_readiness = {}
    candidate_eligibility = {
        name: set(select_eligible_features(train_weekly, features))
        for name, features in WEEKLY_BASELINE_BUNDLES.items()
    }
    for col in ["price_idx", "price_gap_ref_8w", "promotion_share", "freight_mean", "freight_pct_change_1w", "discount_mean", "promo_any", "stockout_share"]:
        present_weekly = col in weekly_full.columns
        series_weekly = pd.to_numeric(weekly_full[col], errors="coerce") if present_weekly else pd.Series(dtype=float)
        used_in_baseline = bool(col in baseline_feature_names)
        used_in_attempted_uplift = bool(col in uplift_bundle_attempted.get("features", []))
        used_in_active_uplift = bool(len(uplift_bundle.get("models", [])) > 0 and col in uplift_bundle.get("features", []))
        used_in_final_active_forecast = bool(used_in_baseline or used_in_active_uplift)
        if used_in_active_uplift:
            active_usage_reason = "active_learned_uplift"
        elif used_in_attempted_uplift:
            active_usage_reason = "uplift_deactivated_by_gate"
        elif used_in_baseline:
            active_usage_reason = "active_baseline_feature"
        else:
            active_usage_reason = "not_used_in_active_path"
        feature_row = {
            "present_in_weekly": bool(present_weekly),
            "non_null_share": float(series_weekly.notna().mean()) if present_weekly and len(series_weekly) else 0.0,
            "nunique": int(series_weekly.nunique(dropna=True)) if present_weekly else 0,
            "used_in_attempted_uplift": used_in_attempted_uplift,
            "used_in_active_uplift": used_in_active_uplift,
            "used_in_active_baseline": used_in_baseline,
            "used_in_final_active_forecast": used_in_final_active_forecast,
            "active_usage_reason": active_usage_reason,
        }
        if col in {"price_idx", "promotion_share", "freight_mean"}:
            feature_row["eligible_for_price_only_baseline"] = bool(col in candidate_eligibility.get("price_only_baseline", set()))
            feature_row["eligible_for_price_promo_baseline"] = bool(col in candidate_eligibility.get("price_promo_baseline", set()))
            feature_row["eligible_for_price_promo_freight_baseline"] = bool(col in candidate_eligibility.get("price_promo_freight_baseline", set()))
            feature_row["eligible_for_weekly_uplift"] = bool(col in uplift_bundle.get("features", []))
            feature_row["used_in_weekly_uplift_attempted"] = bool(col in uplift_bundle_attempted.get("features", []))
            feature_row["used_in_weekly_uplift_active"] = bool(len(uplift_bundle.get("models", [])) > 0 and col in uplift_bundle.get("features", []))
            feature_row["weekly_uplift_attempted_reason"] = "" if feature_row["used_in_weekly_uplift_attempted"] else "not_selected_for_attempted_uplift"
            feature_row["weekly_uplift_active_reason"] = "" if feature_row["used_in_weekly_uplift_active"] else (str(uplift_bundle.get("reason", "")) if feature_row["used_in_weekly_uplift_attempted"] else "not_selected_for_attempted_uplift")
        elif col in {"price_gap_ref_8w", "freight_pct_change_1w", "stockout_share", "promo_any"}:
            feature_row["eligible_for_weekly_uplift"] = bool(col in uplift_bundle.get("features", []))
            feature_row["used_in_weekly_uplift_attempted"] = bool(col in uplift_bundle_attempted.get("features", []))
            feature_row["used_in_weekly_uplift_active"] = bool(len(uplift_bundle.get("models", [])) > 0 and col in uplift_bundle.get("features", []))
            feature_row["weekly_uplift_attempted_reason"] = "" if feature_row["used_in_weekly_uplift_attempted"] else "not_selected_for_attempted_uplift"
            feature_row["weekly_uplift_active_reason"] = "" if feature_row["used_in_weekly_uplift_active"] else (str(uplift_bundle.get("reason", "")) if feature_row["used_in_weekly_uplift_attempted"] else "not_selected_for_attempted_uplift")
            feature_row["excluded_by_design"] = False
        else:
            feature_row["excluded_by_design"] = True
            feature_row["exclusion_reason"] = "removed_from_weekly_baseline_bundle_policy"
            feature_row["used_in_weekly_uplift_attempted"] = False
            feature_row["used_in_weekly_uplift_active"] = False
            feature_row["weekly_uplift_attempted_reason"] = "excluded_by_design"
            feature_row["weekly_uplift_active_reason"] = "excluded_by_design"
        candidate_feature_readiness[col] = feature_row
    holdout_actual = holdout_predictions["actual_sales"].astype(float).values
    holdout_final = holdout_predictions["final_pred_sales"].astype(float).values
    holdout_actual_std = float(np.std(holdout_actual, ddof=0)) if len(holdout_actual) else float("nan")
    holdout_pred_std = float(np.std(holdout_final, ddof=0)) if len(holdout_final) else float("nan")
    sensitivity_trained_bundle = {
        "baseline_bundle": baseline_bundle,
        "uplift_bundle": uplift_bundle,
        "daily_base": daily_base,
        "base_ctx": base_ctx,
        "latest_row": latest_row,
        "future_dates": future_dates,
        "elasticity_map": shrunk_random_effects,
        "pooled_elasticity": fixed_log_price_coef,
        "confidence": confidence,
        "small_mode_info": small_mode_info,
    }
    sensitivity_base = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)),
    )
    sensitivity_price_minus_5 = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)) * 0.95,
    )
    sensitivity_price_plus_5 = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)) * 1.05,
    )
    sensitivity_promo_plus_10pp = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)),
        overrides={"promotion": min(1.0, float(base_ctx.get("promotion", 0.0)) + 0.10)},
    )
    sensitivity_freight_plus_10pct = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)),
        overrides={"freight_value": float(base_ctx.get("freight_value", 0.0)) * 1.10},
    )
    base_demand_total = float(sensitivity_base["demand_total"])
    scenario_sensitivity_diagnostics = {
        "selected_forecaster": selected_forecaster,
        "selected_candidate": selected_candidate_name,
        "selection_reason": str(weekly_baseline_candidate_comparison.get("selection_reason", "")),
        "baseline_has_exogenous_driver": bool(baseline_has_exog),
        "scenario_driver_mode": resolve_scenario_driver_mode(selected_forecaster, bool(baseline_has_exog)),
        "weekly_driver_mode": str(sensitivity_base.get("weekly_driver_mode", "naive_core_only")),
        "learned_uplift_active": False,
        "fallback_multiplier_used": bool(sensitivity_base.get("fallback_multiplier_used", False)),
        "fallback_reason": str(sensitivity_base.get("fallback_reason", "")),
        "model_backend": backend_status["model_backend"],
        "backend_reason": backend_status["backend_reason"],
        "source": "run_what_if_projection_runtime_path",
        "price_minus_5pct_demand_delta_pct": float(((sensitivity_price_minus_5["demand_total"] - base_demand_total) / max(base_demand_total, 1e-9)) * 100.0),
        "price_plus_5pct_demand_delta_pct": float(((sensitivity_price_plus_5["demand_total"] - base_demand_total) / max(base_demand_total, 1e-9)) * 100.0),
        "promo_plus_10pp_demand_delta_pct": float(((sensitivity_promo_plus_10pp["demand_total"] - base_demand_total) / max(base_demand_total, 1e-9)) * 100.0),
        "freight_plus_10pct_demand_delta_pct": float(((sensitivity_freight_plus_10pct["demand_total"] - base_demand_total) / max(base_demand_total, 1e-9)) * 100.0),
    }
    final_active_path = resolve_final_active_path(
        selected_forecaster=selected_forecaster,
        production_selected_candidate=selected_candidate_name,
    )
    run_summary = {
            "config": {
            "git_commit": _safe_git_commit(),
            "code_signature": code_signature(),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "app_generation": APP_GENERATION,
            "date_utc": str(pd.Timestamp.utcnow()),
            "series_id": str(target_sku),
            "category": str(target_category),
            "train_period": [str(train_weekly["week_start"].min()), str(train_weekly["week_start"].max())],
            "validation_period": [None, None],
            "holdout_period": [str(test_weekly["week_start"].min()), str(test_weekly["week_start"].max())],
            "fit_scope": "refit_full_history",
            "horizon_days": int(len(future_dates)),
            "baseline_features": baseline_feature_names,
            "seasonal_anchor_weight": float(seasonal_anchor_weight_full),
            "uplift_features": uplift_bundle.get("features", []),
            "uplift_features_used": uplift_bundle.get("features", []),
            "weekly_baseline_features_used": baseline_feature_names,
            "amplitude_calibrator": amplitude_calibrator_info,
            "weekly_uplift_features_used": uplift_bundle.get("features", []),
            "uplift_signal_info": uplift_bundle.get("signal_info", {}),
            "uplift_attempted_features": uplift_bundle_attempted.get("features", []),
            "uplift_attempted_model_count": int(len(uplift_bundle_attempted.get("models", []))),
            "uplift_attempted_signal_info": uplift_bundle_attempted.get("signal_info", {}),
            "uplift_active_features": uplift_bundle.get("features", []),
            "uplift_active_model_count": int(len(uplift_bundle.get("models", []))),
            "uplift_gate_result": str(uplift_gate_result),
            "uplift_gate_reason": str(uplift_gate_reason),
            "uplift_gate_diagnostics": uplift_gate_diagnostics,
            "uplift_debug_info": uplift_bundle_attempted.get("debug_info", {}),
            "uplift_support_train": support_snapshot_train,
            "uplift_support_holdout": support_snapshot_holdout,
            "support_snapshot_train": support_snapshot_train,
            "support_snapshot_holdout": support_snapshot_holdout,
            "uplift_enabled": bool(len(uplift_bundle.get("models", [])) > 0),
            "uplift_reason": str(uplift_bundle.get("reason", "")),
            "uplift_holdout_keep": bool(uplift_keep),
            "uplift_activation_mode": LEARNED_UPLIFT_MODE,
            "uplift_activation": uplift_activation,
            "uplift_mode": "diagnostic_only",
            "uplift_used_in_production": False,
            "quality_improvement_expected": False,
            "quality_improvement_expectation_reason": "diagnostic_only_modes_active_path_frozen",
            "final_active_path": final_active_path,
                "v1_contract": {
                    "active_path": final_active_path,
                    "learned_uplift_status": "inactive_production_diagnostic_only",
                    "factor_application": "weekly_baseline_recompute_plus_scenario_engine_no_double_counting",
                },
            "factor_contract": {
                "active_model_factors": active_model_factors,
                "scenario_only_factors": scenario_only_factors,
                "attempted_but_disabled_factors": attempted_but_disabled_factors,
            },
            "attempted_uplift_metrics": attempted_uplift_metrics,
            "active_uplift_metrics": active_uplift_metrics,
            "final_active_metrics": active_uplift_metrics,
            "attempted_vs_active_delta": attempted_vs_active_delta,
            "benchmark_gate_passed": bool(model_enabled),
            "shape_quality_low": bool(shape_quality_low),
            "selected_forecaster": selected_forecaster,
            "selected_candidate": selected_candidate_name,
            "production_selected_candidate": "legacy_baseline",
            "diagnostic_selected_candidate": str(weekly_baseline_candidate_comparison.get("selected_candidate", selected_candidate_name)),
            "selection_mode": "diagnostic_comparison_runtime_frozen_to_legacy",
            "production_selection_reason": "v1_contract_runtime_frozen_to_legacy",
            "selection_reason": str(weekly_baseline_candidate_comparison.get("selection_reason", "")),
            "model_backend": backend_status["model_backend"],
            "backend_reason": backend_status["backend_reason"],
            "baseline_has_exogenous_driver": bool(baseline_has_exog),
            "scenario_driver_mode": resolve_scenario_driver_mode(selected_forecaster, bool(baseline_has_exog)),
            "weekly_driver_mode": str(as_is_sim.get("weekly_driver_mode", "naive_core_only")),
            "learned_uplift_active": False,
            "fallback_multiplier_used": bool(as_is_sim.get("fallback_multiplier_used", False)),
            "fallback_reason": str(as_is_sim.get("fallback_reason", "")),
            "backend_warning": backend_warning_text,
        },
        "dataset_passport": _build_dataset_passport(txn),
        "metrics_summary": {
            "holdout": {
                "core": holdout_core_metrics,
                "attempted_uplift": attempted_uplift_metrics,
                "final_active": active_uplift_metrics,
            },
            "holdout_flat": dict(holdout_metrics),
            "rolling_weekly_backtest": {
                "active_fold_gate_weekly_wape": float(backtest_summary.get("active_weekly_wape", float("nan"))),
                "active_fold_gate_weekly_mae": float(backtest_summary.get("active_weekly_mae", float("nan"))),
                "amplitude_scale_median": float(backtest_summary.get("amplitude_scale_median", float("nan"))),
                "uplift_keep_fold_rate": float(backtest_summary.get("uplift_keep_fold_rate", float("nan"))),
                "support_too_low_fold_rate": float(backtest_summary.get("support_too_low_fold_rate", float("nan"))),
                "neutral_bias_mean": float(backtest_summary.get("neutral_bias_mean", float("nan"))),
                "core": {
                    "weekly_wape": float(backtest_summary.get("core_weekly_wape", float("nan"))),
                    "weekly_mae": float(backtest_summary.get("core_weekly_mae", float("nan"))),
                },
                "attempted_uplift": {
                    "weekly_wape": float(backtest_summary.get("attempted_weekly_wape", float("nan"))),
                    "weekly_mae": float(backtest_summary.get("attempted_weekly_mae", float("nan"))),
                    "uplift_keep_fold_rate": float(backtest_summary.get("uplift_keep_fold_rate", float("nan"))),
                    "support_too_low_fold_rate": float(backtest_summary.get("support_too_low_fold_rate", float("nan"))),
                    "neutral_bias_mean": float(backtest_summary.get("neutral_bias_mean", float("nan"))),
                },
                "final_active": {
                    "weekly_wape": float(backtest_summary.get("active_weekly_wape", float("nan"))) if bool(uplift_activation.get("active", False)) else float(backtest_summary.get("core_weekly_wape", float("nan"))),
                    "weekly_mae": float(backtest_summary.get("active_weekly_mae", float("nan"))) if bool(uplift_activation.get("active", False)) else float(backtest_summary.get("core_weekly_mae", float("nan"))),
                    "uplift_active": bool(uplift_activation.get("active", False)),
                    "activation_reason": str(uplift_activation.get("reason", "")),
                },
            },
        },
        "weekly_baseline_candidate_comparison": weekly_baseline_candidate_comparison,
        "scenario_sensitivity_diagnostics": scenario_sensitivity_diagnostics,
        "candidate_feature_readiness": candidate_feature_readiness,
        "warnings": warnings,
        "small_mode_info": small_mode_info,
        "feature_usage_report": feature_report.to_dict("records"),
        "scenario_inputs": {"as_is": {"price": float(base_ctx.get("price")), "discount": float(base_ctx.get("discount", 0.0))}, "neutral_baseline": neutral_overrides},
        "scenario_output_summary": {
            "artifact_scope": "analysis_only",
            "scenario_status": "as_is" if scenario_sim is None else "computed",
            "scenario_reason": "" if scenario_sim is None else "",
            "active_path_contract": final_active_path,
            "learned_uplift_contract": "inactive_production_diagnostic_only",
            "uplift_mode": "diagnostic_only",
            "uplift_used_in_production": False,
            "scenario_driver_mode": str(as_is_sim.get("scenario_driver_mode", "unknown")),
            "weekly_driver_mode": str(as_is_sim.get("weekly_driver_mode", "naive_core_only")),
            "fallback_multiplier_used": bool(as_is_sim.get("fallback_multiplier_used", False)),
            "fallback_reason": str(as_is_sim.get("fallback_reason", "")),
            "selected_candidate": selected_candidate_name,
            "selection_reason": str(weekly_baseline_candidate_comparison.get("selection_reason", "")),
            "model_backend": backend_status["model_backend"],
            "backend_reason": backend_status["backend_reason"],
            "backend_warning": backend_warning_text,
            "holdout_support_status": "low" if bool(holdout_support.get("support_too_low", True)) else "ok",
            "scenario_sensitivity_status": "computed",
            "baseline_demand_total": float(baseline_sim["daily"]["actual_sales"].sum()),
            "as_is_demand_total": float(as_is_sim["daily"]["actual_sales"].sum()),
            "scenario_demand_total": float(scenario_sim["daily"]["actual_sales"].sum()) if scenario_sim else float(as_is_sim["daily"]["actual_sales"].sum()),
            "baseline_revenue_total": float(baseline_sim["daily"]["revenue"].sum()),
            "as_is_revenue_total": float(as_is_sim["daily"]["revenue"].sum()),
            "scenario_revenue_total": float(scenario_sim["daily"]["revenue"].sum()) if scenario_sim else float(as_is_sim["daily"]["revenue"].sum()),
            "baseline_profit_total": float(baseline_sim["daily"]["profit"].sum()),
            "as_is_profit_total": float(as_is_sim["daily"]["profit"].sum()),
            "scenario_profit_total": float(scenario_sim["daily"]["profit"].sum()) if scenario_sim else float(as_is_sim["daily"]["profit"].sum()),
            "corr_final": float(np.corrcoef(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"])[0, 1]) if len(holdout_predictions) > 1 else float("nan"),
            "std_ratio_final": float(np.std(holdout_predictions["final_pred_sales"], ddof=0) / max(np.std(holdout_predictions["actual_sales"], ddof=0), 1e-9)) if len(holdout_predictions) > 1 else float("nan"),
            "corr_baseline": float(corr_baseline),
            "std_ratio_baseline": float(std_ratio_baseline),
            "shape_quality_low": bool(shape_quality_low),
            "bias_mean": float(holdout_predictions["signed_error"].mean()) if len(holdout_predictions) else float("nan"),
            "bias_median": float(holdout_predictions["signed_error"].median()) if len(holdout_predictions) else float("nan"),
            "wape_baseline": float(calculate_wape(holdout_predictions["actual_sales"], holdout_predictions["baseline_pred_sales"])) if len(holdout_predictions) else float("nan"),
            "wape_final": float(calculate_wape(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"])) if len(holdout_predictions) else float("nan"),
            "uplift_holdout_keep": bool(uplift_keep),
            "rmse_baseline": float(mean_squared_error(holdout_predictions["actual_sales"], holdout_predictions["baseline_pred_sales"]) ** 0.5) if len(holdout_predictions) else float("nan"),
            "rmse_final": float(mean_squared_error(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"]) ** 0.5) if len(holdout_predictions) else float("nan"),
            "mape_baseline": float(calculate_mape(holdout_predictions["actual_sales"], holdout_predictions["baseline_pred_sales"])) if len(holdout_predictions) else float("nan"),
            "mape_final": float(calculate_mape(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"])) if len(holdout_predictions) else float("nan"),
            "delta_wape_pct": float(
                ((calculate_wape(holdout_predictions["actual_sales"], holdout_predictions["baseline_pred_sales"]) - calculate_wape(holdout_predictions["actual_sales"], holdout_predictions["final_pred_sales"]))
                 / max(calculate_wape(holdout_predictions["actual_sales"], holdout_predictions["baseline_pred_sales"]), 1e-9)) * 100.0
            ) if len(holdout_predictions) else float("nan"),
            "mean_pred_on_zero_days": mean_pred_on_zero_days,
            "false_positive_rate_on_zero_days": false_positive_rate_on_zero_days,
            "wape_positive_days": wape_positive_days,
            "weekly_wape_backtest": backtest_summary.get("weekly_wape", float("nan")),
            "weekly_positive_periods_wape": wape_positive_days,
            "weekly_zero_actual_mean_pred": mean_pred_on_zero_days,
            "weekly_false_positive_rate_on_zero": false_positive_rate_on_zero_days,
        },
        "shape_diagnostics": {
            "actual_mean": float(np.mean(holdout_actual)) if len(holdout_actual) else float("nan"),
            "pred_mean": float(np.mean(holdout_final)) if len(holdout_final) else float("nan"),
            "actual_std": holdout_actual_std,
            "pred_std": holdout_pred_std,
            "std_ratio": float(holdout_pred_std / max(holdout_actual_std, 1e-9)) if np.isfinite(holdout_actual_std) and np.isfinite(holdout_pred_std) else float("nan"),
            "actual_min": float(np.min(holdout_actual)) if len(holdout_actual) else float("nan"),
            "actual_max": float(np.max(holdout_actual)) if len(holdout_actual) else float("nan"),
            "pred_min": float(np.min(holdout_final)) if len(holdout_final) else float("nan"),
            "pred_max": float(np.max(holdout_final)) if len(holdout_final) else float("nan"),
            "actual_peak_to_trough": float(np.max(holdout_actual) - np.min(holdout_actual)) if len(holdout_actual) else float("nan"),
            "pred_peak_to_trough": float(np.max(holdout_final) - np.min(holdout_final)) if len(holdout_final) else float("nan"),
        },
    }
    current_price = float(base_ctx.get("price"))
    return {
        "history_daily": daily_base,
        "quality_report": {"holdout_metrics": holdout_metrics},
        "feature_usage_report": feature_report,
        "feature_report": feature_report,
        "neutral_baseline_forecast": baseline_sim["daily"],
        "as_is_forecast": as_is_sim["daily"],
        "scenario_forecast": scenario_forecast,
        "delta_vs_as_is": delta_vs_as_is,
        "delta_vs_baseline": delta_vs_baseline,
        "warnings": warnings,
        "small_mode_info": small_mode_info,
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "elasticity_map": shrunk_random_effects,
        "current_price": float(base_ctx.get("price")),
        "scenario_price": None,
        "current_profit": float(as_is_sim.get("adjusted_profit", as_is_sim.get("total_profit", 0.0))),
        "cost_input_available": bool("cost" in txn.columns and pd.to_numeric(txn.get("cost"), errors="coerce").notna().any()),
        "excel_buffer": excel_buffer,
        "analysis_run_summary_json": json.dumps(run_summary, ensure_ascii=False, indent=2).encode("utf-8"),
        "holdout_predictions_csv": holdout_predictions.to_csv(index=False).encode("utf-8"),
        "holdout_weekly_diagnostics_csv": holdout_weekly_diagnostics.to_csv(index=False).encode("utf-8"),
        "uplift_debug_report_csv": uplift_debug_report.to_csv(index=False).encode("utf-8"),
        "uplift_holdout_trace_csv": uplift_holdout_trace.to_csv(index=False).encode("utf-8"),
        "analysis_baseline_vs_as_is_csv": analysis_baseline_vs_as_is.to_csv(index=False).encode("utf-8"),
        "manual_scenario_summary_json": None,
        "manual_scenario_daily_csv": None,
        "feature_report_csv": feature_report.to_csv(index=False).encode("utf-8"),
        "flag": {"auto_apply": False, "reasons": {"mode": "single-core-what-if"}},
        "_trained_bundle": {
            "baseline_bundle": baseline_bundle,
            "uplift_bundle": uplift_bundle,
            "uplift_bundle_attempted": uplift_bundle_attempted,
            "uplift_bundle_active": uplift_bundle,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "elasticity_map": shrunk_random_effects,
            "pooled_elasticity": fixed_log_price_coef,
            "ref_net_price": ref_net_price,
            "confidence": confidence,
            "small_mode": small_mode,
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
    current_ctx = dict(trained_bundle["base_ctx"])
    latest_row_current = dict(trained_bundle["latest_row"])

    scenario_overrides = dict(overrides or {})
    scenario_overrides.setdefault("freight_multiplier", float(freight_multiplier))
    scenario_overrides.setdefault("discount_multiplier", float(discount_multiplier))
    scenario_overrides.setdefault("cost_multiplier", float(cost_multiplier))
    if stock_cap:
        scenario_overrides["stock_cap"] = float(stock_cap)
    if float(demand_multiplier) != 1.0:
        scenario_overrides["manual_shock_multiplier"] = float(demand_multiplier)

    future_dates = trained_bundle["future_dates"]
    if horizon_days is not None:
        future_dates = forecast_future_dates(pd.Timestamp(base_history["date"].max()), n_days=int(horizon_days))

    baseline_price_ref = float(current_ctx.get("price", manual_price))
    baseline_overrides = {
        "promotion": float(current_ctx.get("promotion", 0.0)),
        "freight_multiplier": 1.0,
        "discount_multiplier": 1.0,
        "cost_multiplier": 1.0,
        "manual_shock_multiplier": 1.0,
    }
    baseline_sim = simulate_horizon_profit(
        latest_row_current,
        baseline_price_ref,
        future_dates,
        trained_bundle["baseline_bundle"],
        trained_bundle["uplift_bundle"],
        base_history,
        current_ctx,
        trained_bundle["elasticity_map"],
        trained_bundle["pooled_elasticity"],
        overrides=baseline_overrides,
    )
    daily = baseline_sim["daily"].copy()
    baseline_daily = baseline_sim["daily"].copy()
    shocks = list(scenario_overrides.get("shocks", [])) if isinstance(scenario_overrides.get("shocks", []), list) else []
    if float(demand_multiplier) != 1.0 and len(future_dates):
        dm = float(demand_multiplier)
        shocks.append(
            {
                "shock_name": "demand_multiplier",
                "shock_type": "percent",
                "shock_value": dm - 1.0,
                "start_date": str(pd.to_datetime(future_dates["date"]).min().date()),
                "end_date": str(pd.to_datetime(future_dates["date"]).max().date()),
            }
        )

    current_price_raw = float(current_ctx.get("price", manual_price))
    baseline_discount = float(np.clip(current_ctx.get("discount", 0.0), 0.0, 0.95))
    scenario_discount = float(scenario_overrides.get("discount", baseline_discount))
    scenario_discount *= float(scenario_overrides.get("discount_multiplier", 1.0))
    scenario_discount = float(np.clip(scenario_discount, 0.0, 0.95))
    requested_price = float(manual_price)
    train_min = float(pd.to_numeric(base_history.get("price", pd.Series([requested_price])), errors="coerce").dropna().min())
    train_max = float(pd.to_numeric(base_history.get("price", pd.Series([requested_price])), errors="coerce").dropna().max())
    if not np.isfinite(train_min) or not np.isfinite(train_max):
        train_min = requested_price
        train_max = requested_price
    clip = compute_scenario_price_inputs(requested_price=requested_price, train_min=train_min, train_max=train_max)
    model_price = float(clip["model_price"])
    base_net_price_ref = float(max(0.01, current_price_raw * (1.0 - baseline_discount)))
    scenario_net_price = float(max(0.01, model_price * (1.0 - scenario_discount)))

    baseline_cost = float(current_ctx.get("cost", baseline_price_ref * CONFIG["COST_PROXY_RATIO"]))
    scenario_cost = float(scenario_overrides.get("cost", baseline_cost))
    scenario_cost *= float(scenario_overrides.get("cost_multiplier", 1.0))
    scenario_cost = max(0.0, scenario_cost)

    baseline_freight = float(current_ctx.get("freight_value", 0.0))
    scenario_freight = float(scenario_overrides.get("freight_value", baseline_freight))
    scenario_freight *= float(scenario_overrides.get("freight_multiplier", 1.0))
    scenario_freight = max(0.0, scenario_freight)

    stock_series = pd.to_numeric(
        baseline_daily.get("stock", daily.get("stock", pd.Series([np.inf] * len(daily)))),
        errors="coerce",
    ).fillna(np.inf).to_numpy(dtype=float)
    if float(scenario_overrides.get("stock_cap", 0.0)) > 0:
        stock_series = np.minimum(stock_series, float(scenario_overrides["stock_cap"]))

    effective_scenario = {
        "requested_price_gross": float(manual_price),
        "applied_price_gross": float(model_price),
        "current_price_gross": float(current_price_raw),
        "current_discount": float(baseline_discount),
        "applied_discount": float(scenario_discount),
        "current_price_net": float(base_net_price_ref),
        "applied_price_net": float(scenario_net_price),
        "promotion": float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
        "freight_value": float(scenario_freight),
        "cost": float(scenario_cost),
        "demand_multiplier": float(scenario_overrides.get("manual_shock_multiplier", 1.0)),
        "price_clipped": bool(clip.get("price_clipped", False)),
        "clip_reason": clip.get("clip_reason"),
        "lower_clip": float(train_min) if np.isfinite(train_min) else None,
        "upper_clip": float(train_max) if np.isfinite(train_max) else None,
    }

    scenario_inputs = {
        "demand_price_baseline": float(base_net_price_ref),
        "demand_price_scenario": float(scenario_net_price),
        "gross_price_baseline": float(current_price_raw),
        "gross_price_scenario": float(model_price),
        "base_price": float(base_net_price_ref),
        "scenario_price": float(scenario_net_price),
        "baseline_net_price": float(base_net_price_ref),
        "scenario_net_price": float(scenario_net_price),
        "price_elasticity": float(trained_bundle.get("pooled_elasticity", CONFIG["PRIOR_ELASTICITY"])),
        "price_elasticity_prior": float(CONFIG["PRIOR_ELASTICITY"]),
        "price_cap": 0.35,
        "promo_baseline": float(current_ctx.get("promotion", 0.0)),
        "promo_scenario": float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
        "promo_flag_baseline": float(1.0 if float(current_ctx.get("promotion", 0.0)) > 0 else 0.0),
        "promo_flag_scenario": float(1.0 if float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))) > 0 else 0.0),
        "promo_intensity_baseline": float(current_ctx.get("promotion", 0.0)),
        "promo_intensity_scenario": float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
        "freight_baseline": float(current_ctx.get("freight_value", 0.0)),
        "freight_ref": float(current_ctx.get("freight_value", 0.0)),
        "freight_scenario": scenario_freight,
        "baseline_freight_value": float(current_ctx.get("freight_value", 0.0)),
        "freight_value": scenario_freight,
        "baseline_unit_cost": float(current_ctx.get("cost", baseline_price_ref * CONFIG["COST_PROXY_RATIO"])),
        "unit_cost": scenario_cost,
        "available_stock": stock_series,
    }
    promo_base = float(current_ctx.get("promotion", 0.0))
    promo_scenario = float(scenario_overrides.get("promotion", promo_base))
    manual_shock_multiplier = float(scenario_overrides.get("manual_shock_multiplier", 1.0))
    has_explicit_shocks = bool(len(shocks))
    scenario_changed = bool(
        abs(effective_scenario["applied_price_gross"] - effective_scenario["current_price_gross"]) > 1e-9
        or abs(scenario_discount - baseline_discount) > 1e-9
        or abs(scenario_freight - baseline_freight) > 1e-9
        or abs(scenario_cost - baseline_cost) > 1e-9
        or abs(promo_scenario - promo_base) > 1e-9
        or abs(manual_shock_multiplier - 1.0) > 1e-9
        or has_explicit_shocks
        or abs(float(scenario_overrides.get("stock_cap", 0.0))) > 1e-9
    )
    scenario_status = "computed" if scenario_changed else "as_is"
    baseline_for_scenario = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["date"]) if "date" in daily.columns else pd.Series(dtype="datetime64[ns]"),
            "baseline_units": pd.to_numeric(baseline_sim["daily"].get("base_pred_sales", pd.Series(np.zeros(len(daily)))), errors="coerce").fillna(0.0).values,
        }
    )
    scenario_meta = dict(trained_bundle.get("small_mode_info", {}))
    scenario_result = run_scenario(
        baseline_output=baseline_for_scenario,
        scenario_inputs=scenario_inputs,
        shocks=shocks if isinstance(shocks, list) else [],
        metadata=scenario_meta,
    )
    if len(daily):
        daily["base_pred_sales"] = np.asarray(scenario_result["baseline_units"], dtype=float)
        daily["pred_sales"] = np.asarray(scenario_result["final_units"], dtype=float)
        daily["price"] = float(model_price)
        daily["discount"] = float(scenario_discount)
        daily["cost"] = scenario_cost
        daily["freight_value"] = scenario_freight
        daily["net_unit_price"] = scenario_net_price
        daily["promotion"] = float(promo_scenario)
        daily["stock"] = stock_series
        daily["unconstrained_demand"] = daily["pred_sales"].clip(lower=0.0)
        stock_lookup = pd.to_numeric(daily.get("stock", pd.Series([np.inf] * len(daily))), errors="coerce").fillna(np.inf).values
        daily["actual_sales"] = np.minimum(daily["unconstrained_demand"].values, stock_lookup)
        daily["lost_sales"] = (daily["unconstrained_demand"] - daily["actual_sales"]).clip(lower=0.0)
        daily["revenue"] = daily["net_unit_price"] * daily["actual_sales"]
        daily["profit"] = (daily["net_unit_price"] - daily["cost"] - daily["freight_value"]) * daily["actual_sales"]
    demand_total = float(daily["actual_sales"].sum()) if "actual_sales" in daily.columns else 0.0
    profit_total_raw = float(daily["profit"].sum()) if "profit" in daily.columns else 0.0
    revenue_total = float(daily["revenue"].sum()) if "revenue" in daily.columns else 0.0
    lost_sales_total = float(daily["lost_sales"].sum()) if "lost_sales" in daily.columns else 0.0
    base_confidence = float(trained_bundle.get("confidence", 0.6))
    price_plausibility = 1.0 if is_price_plausible(effective_scenario["applied_price_gross"], train_min, train_max, margin=0.25) else 0.0
    ood_penalty = 0.20 if not bool(price_plausibility) else 0.0
    horizon_penalty = max(0.0, (len(future_dates) - 30) / 120.0)
    extreme_penalty = max(0.0, abs(float(freight_multiplier) - 1.0) + abs(float(discount_multiplier) - 1.0) + abs(float(cost_multiplier) - 1.0) - 0.25) * 0.08
    history_daily = base_history.copy()
    hist_discount = pd.to_numeric(history_daily.get("discount", pd.Series(np.zeros(len(history_daily)))), errors="coerce").fillna(0.0)
    hist_discount = hist_discount.clip(0.0, 0.95)
    history_daily["net_price_hist"] = pd.to_numeric(history_daily.get("price", pd.Series(np.zeros(len(history_daily)))), errors="coerce").fillna(0.0) * (1.0 - hist_discount)
    net_support = evaluate_net_price_support(
        history_daily["net_price_hist"] if "net_price_hist" in history_daily.columns else history_daily.get("price"),
        float(effective_scenario["applied_price_net"]),
    )
    support_weight = float(scenario_result.get("confidence", {}).get("price", {}).get("score", base_confidence))
    trend_confidence = float(np.clip(1.0 - horizon_penalty, 0.0, 1.0))
    data_confidence = float(np.clip(base_confidence, 0.0, 1.0))
    net_price_penalty = 0.0 if bool(net_support.get("net_price_supported", True)) else 0.18
    discount_depth_penalty = min(0.15, max(0.0, scenario_discount - baseline_discount) * 0.5)
    confidence_scenario = max(
        0.05,
        min(
            0.98,
            0.60 * price_plausibility
            + 0.20 * data_confidence
            + 0.10 * support_weight
            + 0.10 * trend_confidence
            - ood_penalty
            - horizon_penalty
            - extreme_penalty
            - net_price_penalty
            - discount_depth_penalty,
        ),
    )
    confidence_label = "Высокая" if confidence_scenario >= 0.75 else ("Средняя" if confidence_scenario >= 0.45 else "Низкая")
    baseline_bundle_meta = trained_bundle.get("baseline_bundle", {})
    active_path = resolve_final_active_path(
        selected_forecaster=str(baseline_bundle_meta.get("selected_forecaster", "weekly_model")),
        production_selected_candidate=str(baseline_bundle_meta.get("selected_candidate", "legacy_baseline")),
    )
    model_backend = str(trained_bundle.get("baseline_bundle", {}).get("model_backend", "deterministic_fallback"))
    backend_reason = str(trained_bundle.get("baseline_bundle", {}).get("backend_reason", "unknown"))
    backend_warning_text = build_backend_warning(model_backend, backend_reason)
    backend_warning = [backend_warning_text] if backend_warning_text else []
    runtime_meta = {
        "fallback_multiplier_used": bool(baseline_sim.get("fallback_multiplier_used", False)),
        "fallback_reason": str(baseline_sim.get("fallback_reason", "")),
        "learned_uplift_active": False,
        "scenario_driver_mode": "weekly_ml_exogenous_recompute",
        "weekly_driver_mode": str(baseline_sim.get("weekly_driver_mode", "weekly_ml_core_only")),
        "baseline_has_exogenous_driver": bool(baseline_sim.get("baseline_has_exogenous_driver", True)),
        "active_path_contract": active_path,
        "model_backend": model_backend,
        "backend_reason": backend_reason,
    }
    scenario_engine_meta = {
        "engine": "scenario_engine_v1",
        "legacy_simulation_used_upstream": False,
        "price_elasticity_local": float(scenario_inputs["price_elasticity"]),
        "price_elasticity_prior": float(scenario_inputs["price_elasticity_prior"]),
        "price_confidence_score": float(scenario_result.get("confidence", {}).get("price", {}).get("score", float("nan"))),
        "price_confidence_label": str(scenario_result.get("confidence", {}).get("price", {}).get("label", "unknown")),
    }
    return {
        "daily": daily,
        "demand_total": demand_total,
        "profit_total": profit_total_raw,
        "profit_total_raw": profit_total_raw,
        "profit_total_adjusted": float(profit_total_raw),
        "uncertainty_penalty": 0.0,
        "disagreement_penalty": 0.0,
        "revenue_total": revenue_total,
        "lost_sales_total": lost_sales_total,
        "confidence": confidence_scenario,
        "confidence_base": base_confidence,
        "confidence_scenario": confidence_scenario,
        "confidence_label": confidence_label,
        "uncertainty": 1.0 - confidence_scenario,
        "ood_flag": bool(not is_price_plausible(effective_scenario["applied_price_gross"], train_min, train_max, margin=0.25)),
        "requested_price": requested_price,
        "model_price": model_price,
        "price_for_model": model_price,
        "current_price_raw": float(current_price_raw),
        "price_clipped": bool(clip["price_clipped"]),
        "clip_applied": bool(clip["price_clipped"]),
        "clip_reason": str(clip["clip_reason"]),
        "net_price_supported": bool(net_support.get("net_price_supported", True)),
        "net_price_support": net_support,
        "scenario_price_effect_source": "scenario_engine_recompute_from_baseline",
        "fallback_multiplier_used": runtime_meta["fallback_multiplier_used"],
        "fallback_reason": runtime_meta["fallback_reason"],
        "learned_uplift_active": runtime_meta["learned_uplift_active"],
        "uplift_mode": "diagnostic_only",
        "uplift_used_in_production": False,
        "scenario_status": scenario_status,
        "scenario_driver_mode": runtime_meta["scenario_driver_mode"],
        "weekly_driver_mode": runtime_meta["weekly_driver_mode"],
        "baseline_has_exogenous_driver": runtime_meta["baseline_has_exogenous_driver"],
        "legacy_simulation_used": False,
        "legacy_baseline_meta": runtime_meta,
        "active_path_contract": runtime_meta["active_path_contract"],
        "model_backend": model_backend,
        "backend_reason": backend_reason,
        "scenario_engine_meta": scenario_engine_meta,
        "applied_overrides": scenario_overrides,
        "effects": {
            "price_effect": float(scenario_result["price_effect"]),
            "promo_effect": float(scenario_result["promo_effect"]),
            "freight_effect": float(scenario_result["freight_effect"]),
            "stock_effect": float(scenario_result["stock_effect"]),
            "shock_multiplier_mean": float(np.mean(scenario_result["shock_multiplier"])) if len(scenario_result["shock_multiplier"]) else 1.0,
        },
        "scenario_inputs_contract": {
            "base_price": float(baseline_price_ref),
            "scenario_price": float(model_price),
            "requested_price": float(requested_price),
            "model_price": float(model_price),
            "base_promo": float(current_ctx.get("promotion", 0.0)),
            "scenario_promo": float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
            "base_freight": float(current_ctx.get("freight_value", 0.0)),
            "scenario_freight": float(scenario_freight),
            "shock_multiplier": float(demand_multiplier),
        },
        "effective_scenario": effective_scenario,
        "confidence_factors": scenario_result.get("confidence", {}),
        "warnings": scenario_result.get("warnings", []) + backend_warning + ([str(net_support.get("net_price_warning", ""))] if net_support.get("net_price_warning") else []),
    }


def choose_uplift_features(df: pd.DataFrame, small_mode: bool, candidates: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    info = {
        "promotion_positive_share": float((pd.to_numeric(df.get("promotion", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0) > 0).mean()) if len(df) else 0.0,
        "freight_nunique": int(pd.to_numeric(df.get("freight_value", pd.Series(np.zeros(len(df)))), errors="coerce").dropna().nunique()) if len(df) else 0,
        "uplift_target_std": float(pd.to_numeric(df.get("uplift_target_log", pd.Series(np.zeros(len(df)))), errors="coerce").std(ddof=0)) if len(df) else 0.0,
    }
    features = select_eligible_features(df, candidates)
    weak_signal = small_mode or info["promotion_positive_share"] < 0.08 or info["freight_nunique"] <= 2 or info["uplift_target_std"] < 0.03
    if weak_signal:
        features = [f for f in features if f in {"promotion", "freight_value", "baseline_log_feature"}]
    if not features:
        features = ["baseline_log_feature"] if "baseline_log_feature" in df.columns else []
    info["weak_signal_mode"] = bool(weak_signal)
    info["selected_features"] = list(features)
    return features, info


def build_weekly_weak_signal_view(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    wk = df.copy()
    wk["date"] = pd.to_datetime(wk["date"], errors="coerce")
    wk["week_start"] = wk["date"] - pd.to_timedelta(wk["date"].dt.dayofweek, unit="D")
    agg = (
        wk.groupby("week_start", as_index=False)
        .agg(
            sales=("sales", "sum"),
            revenue=("revenue", "sum"),
            freight_value=("freight_value", "mean"),
            discount=("discount", "mean"),
            promotion=("promotion", "mean"),
            cost=("cost", "mean"),
            stock=("stock", "mean"),
            review_score=("review_score", "mean"),
            reviews_count=("reviews_count", "sum"),
            price=("price", "mean"),
        )
        .rename(columns={"week_start": "date"})
    )
    if "net_unit_price" in wk.columns:
        nup = wk.groupby("week_start")["net_unit_price"].mean().reset_index().rename(columns={"week_start": "date"})
        agg = agg.merge(nup, on="date", how="left")
    else:
        agg["net_unit_price"] = agg["price"] * (1.0 - agg["discount"].clip(0.0, 0.95))
    return build_feature_matrix(agg).dropna(subset=["sales", "price", "log_sales", "log_price"]).reset_index(drop=True)


def apply_weekly_fallback_projection(daily_df: pd.DataFrame, history_df: pd.DataFrame, lookback_days: int = 84) -> pd.DataFrame:
    if len(daily_df) == 0:
        return daily_df.copy()
    out = daily_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    hist = history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.sort_values("date")
    if len(hist) > lookback_days:
        hist = hist.iloc[-lookback_days:]
    hist_sales = pd.to_numeric(hist.get("sales", pd.Series(np.zeros(len(hist)))), errors="coerce").fillna(0.0).clip(lower=0.0)
    hist_dow = hist["date"].dt.dayofweek
    dow_share = hist_sales.groupby(hist_dow).sum().reindex(range(7), fill_value=0.0)
    if float(dow_share.sum()) <= 1e-12:
        dow_share = pd.Series(np.repeat(1.0 / 7.0, 7), index=range(7))
    else:
        dow_share = dow_share / float(dow_share.sum())

    out["_dow"] = out["date"].dt.dayofweek
    out["_week_start"] = out["date"] - pd.to_timedelta(out["_dow"], unit="D")
    metrics = ["actual_sales", "revenue", "profit", "lost_sales"]
    for wk_start, idx in out.groupby("_week_start").groups.items():
        idx_list = list(idx)
        week_dows = out.loc[idx_list, "_dow"].astype(int).tolist()
        week_weights = dow_share.reindex(week_dows).astype(float).values
        if float(week_weights.sum()) <= 1e-12:
            week_weights = np.repeat(1.0 / max(len(idx_list), 1), len(idx_list))
        else:
            week_weights = week_weights / float(week_weights.sum())
        for col in metrics:
            if col not in out.columns:
                continue
            total = float(out.loc[idx_list, col].sum())
            out.loc[idx_list, col] = total * week_weights
    out = out.drop(columns=["_dow", "_week_start"])
    return out


def build_manual_scenario_artifacts(result_dict: Dict[str, Any], what_if_result: Dict[str, Any]) -> Tuple[bytes, bytes]:
    as_is = result_dict["as_is_forecast"].copy()
    baseline = result_dict["neutral_baseline_forecast"].copy()
    scenario = what_if_result["daily"].copy()
    merged = (
        as_is[["date", "actual_sales", "revenue", "profit"]]
        .rename(columns={"actual_sales": "as_is_demand", "revenue": "as_is_revenue", "profit": "as_is_profit"})
        .merge(
            scenario[
                [
                    "date",
                    "actual_sales",
                    "revenue",
                    "profit",
                    "price",
                    "discount",
                    "net_unit_price",
                    "promotion",
                    "freight_value",
                    "cost",
                    "lost_sales",
                ]
            ].rename(
                columns={
                    "actual_sales": "scenario_demand",
                    "revenue": "scenario_revenue",
                    "profit": "scenario_profit",
                    "price": "scenario_price_gross",
                    "discount": "scenario_discount",
                    "net_unit_price": "scenario_price_net",
                    "promotion": "scenario_promotion",
                    "freight_value": "scenario_freight_value",
                    "cost": "scenario_cost",
                }
            ),
            on="date",
            how="outer",
        )
        .merge(
            baseline[["date", "actual_sales", "revenue", "profit"]].rename(
                columns={"actual_sales": "baseline_demand", "revenue": "baseline_revenue", "profit": "baseline_profit"}
            ),
            on="date",
            how="left",
        )
        .sort_values("date")
    )
    merged["delta_demand"] = merged["scenario_demand"] - merged["as_is_demand"]
    merged["delta_revenue"] = merged["scenario_revenue"] - merged["as_is_revenue"]
    merged["delta_profit"] = merged["scenario_profit"] - merged["as_is_profit"]
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged["series_id"] = str(result_dict.get("_trained_bundle", {}).get("base_ctx", {}).get("product_id", "unknown"))
    manual_daily = merged[
        [
            "date",
            "series_id",
            "as_is_demand",
            "scenario_demand",
            "delta_demand",
            "as_is_revenue",
            "scenario_revenue",
            "delta_revenue",
            "as_is_profit",
            "scenario_profit",
            "delta_profit",
            "scenario_price_gross",
            "scenario_discount",
            "scenario_price_net",
            "scenario_promotion",
            "scenario_freight_value",
            "scenario_cost",
            "lost_sales",
        ]
    ].copy()
    summary = {
        "artifact_scope": "manual_scenario",
        "scenario_status": what_if_result.get("scenario_status", "as_is"),
        "requested_price": float(what_if_result.get("requested_price", np.nan)),
        "model_price": float(what_if_result.get("model_price", what_if_result.get("price_for_model", np.nan))),
        "applied_discount": float((what_if_result.get("effective_scenario", {}) or {}).get("applied_discount", np.nan)),
        "applied_price_net": float((what_if_result.get("effective_scenario", {}) or {}).get("applied_price_net", np.nan)),
        "modeled_price": float(what_if_result.get("model_price", what_if_result.get("price_for_model", np.nan))),
        "current_price_raw": float(what_if_result.get("current_price_raw", np.nan)),
        "clip_applied": bool(what_if_result.get("clip_applied", what_if_result.get("price_clipped", False))),
        "clip_reason": str(what_if_result.get("clip_reason", "")),
        "scenario_price_effect_source": str(what_if_result.get("scenario_price_effect_source", "")),
        "ood_flag": bool(what_if_result.get("ood_flag", False)),
        "horizon_days": int(len(scenario)),
        "scenario_demand_total": float(manual_daily["scenario_demand"].sum()),
        "scenario_revenue_total": float(manual_daily["scenario_revenue"].sum()),
        "scenario_profit_total": float(manual_daily["scenario_profit"].sum()),
        "scenario_vs_as_is_demand_pct": float((manual_daily["delta_demand"].sum() / max(float(manual_daily["as_is_demand"].sum()), 1e-9)) * 100.0),
        "scenario_vs_as_is_profit_pct": float((manual_daily["delta_profit"].sum() / max(float(manual_daily["as_is_profit"].sum()), 1e-9)) * 100.0),
        "delta_vs_as_is": {
            "demand_total": float(manual_daily["delta_demand"].sum()),
            "revenue_total": float(manual_daily["delta_revenue"].sum()),
            "profit_total": float(manual_daily["delta_profit"].sum()),
        },
        "delta_vs_neutral_baseline": {
            "demand_total": float(manual_daily["scenario_demand"].sum() - baseline["actual_sales"].sum()),
            "revenue_total": float(manual_daily["scenario_revenue"].sum() - baseline["revenue"].sum()),
            "profit_total": float(manual_daily["scenario_profit"].sum() - baseline["profit"].sum()),
        },
        "uncertainty_penalty": float(what_if_result.get("uncertainty_penalty", 0.0)),
        "confidence": float(what_if_result.get("confidence", np.nan)),
        "scenario_assumptions": what_if_result.get("applied_overrides", {}),
        "scenario_inputs_contract": what_if_result.get("scenario_inputs_contract", {}),
        "applied_overrides": what_if_result.get("applied_overrides", {}),
        "scenario_forecast": manual_daily.to_dict("records"),
        "active_path_contract": str(what_if_result.get("active_path_contract", "legacy_baseline+scenario_recompute")),
        "model_backend": str(what_if_result.get("model_backend", "unknown")),
        "backend_reason": str(what_if_result.get("backend_reason", "")),
        "learned_uplift_contract": "inactive_production_diagnostic_only",
        "scenario_driver_mode": str(what_if_result.get("scenario_driver_mode", "unknown")),
        "weekly_driver_mode": str(what_if_result.get("weekly_driver_mode", "unknown")),
        "fallback_multiplier_used": bool(what_if_result.get("fallback_multiplier_used", False)),
        "fallback_reason": str(what_if_result.get("fallback_reason", "")),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"), manual_daily.to_csv(index=False).encode("utf-8")


def build_trust_block(results: Dict[str, Any], what_if_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    history = results.get("history_daily", pd.DataFrame())
    sufficiency = build_data_sufficiency_status(history)
    active_path = resolve_final_active_path("weekly_model", "legacy_baseline")
    run_summary_cfg: Dict[str, Any] = {}
    scenario_output_summary: Dict[str, Any] = {}
    try:
        run_summary = json.loads(results.get("analysis_run_summary_json", b"{}").decode("utf-8"))
        run_summary_cfg = run_summary.get("config", {}) or {}
        scenario_output_summary = run_summary.get("scenario_output_summary", {}) or {}
        active_path = str(
            run_summary_cfg.get(
                "final_active_path",
                resolve_final_active_path(
                    selected_forecaster=str(run_summary_cfg.get("selected_forecaster", "weekly_model")),
                    production_selected_candidate=str(run_summary_cfg.get("selected_candidate", "legacy_baseline")),
                ),
            )
        )
    except Exception:
        pass
    effective_scenario = (what_if_result or {}).get("effective_scenario", {})
    price_min = float(pd.to_numeric(history.get("price", pd.Series([0.0])), errors="coerce").dropna().min()) if len(history) else 0.0
    price_max = float(pd.to_numeric(history.get("price", pd.Series([0.0])), errors="coerce").dropna().max()) if len(history) else 0.0
    freight_min = float(pd.to_numeric(history.get("freight_value", pd.Series([0.0])), errors="coerce").dropna().min()) if len(history) else 0.0
    freight_max = float(pd.to_numeric(history.get("freight_value", pd.Series([0.0])), errors="coerce").dropna().max()) if len(history) else 0.0
    promo_min = float(pd.to_numeric(history.get("promotion", pd.Series([0.0])), errors="coerce").dropna().min()) if len(history) else 0.0
    promo_max = float(pd.to_numeric(history.get("promotion", pd.Series([0.0])), errors="coerce").dropna().max()) if len(history) else 0.0
    hist_discount = pd.to_numeric(history.get("discount", pd.Series(np.zeros(len(history)))), errors="coerce").fillna(0.0).clip(0.0, 0.95)
    hist_net = pd.to_numeric(history.get("price", pd.Series(np.zeros(len(history)))), errors="coerce").fillna(0.0) * (1.0 - hist_discount)
    net_min = float(pd.to_numeric(hist_net, errors="coerce").dropna().min()) if len(history) else 0.0
    net_max = float(pd.to_numeric(hist_net, errors="coerce").dropna().max()) if len(history) else 0.0
    warnings_local: List[str] = []
    in_range = True
    scenario_range_status: Dict[str, str] = {"price": "ok", "net_price": "ok", "promo": "ok", "freight": "ok"}
    tol = 1e-9

    def _check_range(value: Optional[float], rng_min: float, rng_max: float, key: str, flat_msg: str, out_msg: str) -> None:
        nonlocal in_range
        if value is None:
            return
        if abs(rng_max - rng_min) <= tol:
            if abs(float(value) - rng_min) > tol:
                scenario_range_status[key] = "warn"
                warnings_local.append(flat_msg)
                in_range = False
        else:
            lower_clip = rng_min - 0.15 * abs(rng_max - rng_min)
            upper_clip = rng_max + 0.15 * abs(rng_max - rng_min)
            if not (lower_clip <= float(value) <= upper_clip):
                scenario_range_status[key] = "warn"
                warnings_local.append(out_msg)
                in_range = False

    if effective_scenario:
        _check_range(
            float(effective_scenario.get("applied_price_gross", 0.0)),
            price_min,
            price_max,
            "price",
            "Цена вышла за пределы исторически наблюдаемого уровня",
            "Изменение цены выходит далеко за пределы исторического диапазона",
        )
        _check_range(
            float(effective_scenario.get("applied_price_net", 0.0)),
            net_min,
            net_max,
            "net_price",
            "Итоговая цена для клиента после скидки вышла за фиксированный исторический уровень",
            "Итоговая цена для клиента после скидки выходит за историческую поддержку",
        )
        _check_range(
            float(effective_scenario.get("promotion", 0.0)),
            promo_min,
            promo_max,
            "promo",
            "Промо вышло за пределы исторически наблюдаемого уровня",
            "Промо выходит далеко за пределы исторического диапазона",
        )
        _check_range(
            float(effective_scenario.get("freight_value", 0.0)),
            freight_min,
            freight_max,
            "freight",
            "Логистика вышла за пределы исторически наблюдаемого уровня",
            "Логистика выходит далеко за пределы исторического диапазона",
        )
        net_warning = str((what_if_result or {}).get("net_price_support", {}).get("net_price_warning", ""))
        if net_warning:
            warnings_local.append(net_warning)
            scenario_range_status["net_price"] = "warn"
            in_range = False
    std_ratio_final = scenario_output_summary.get("std_ratio_final", run_summary_cfg.get("std_ratio_final", float("nan")))
    shape_quality_low = bool(run_summary_cfg.get("shape_quality_low", scenario_output_summary.get("shape_quality_low", False)))
    if shape_quality_low or (pd.notna(std_ratio_final) and float(std_ratio_final) < 0.5):
        warnings_local.append("Forecast shape is flatter than actual demand; use scenario results as directional, not exact.")
    return {
        "data_sufficiency": sufficiency["label"],
        "data_message": sufficiency["message"],
        "scenario_range_status": "in_range" if in_range else "out_of_range",
        "scenario_range_details": scenario_range_status if effective_scenario else {},
        "scenario_range_overall": "in_range" if in_range else "out_of_range",
        "active_calculation_mode": active_path,
        "fallback_used": bool((what_if_result or {}).get("fallback_multiplier_used", False)),
        "warnings": sufficiency.get("warnings", []) + warnings_local + list((what_if_result or {}).get("warnings", [])),
    }


def build_business_report_payload(
    results: Dict[str, Any],
    what_if_result: Optional[Dict[str, Any]] = None,
    saved_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    def _finite(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            return parsed if np.isfinite(parsed) else float(default)
        except Exception:
            return float(default)

    as_is = results.get("as_is_forecast", pd.DataFrame())
    baseline = results.get("neutral_baseline_forecast", pd.DataFrame())
    scenario_daily_raw = (what_if_result or {}).get("daily", None)
    requested_status = str((what_if_result or {}).get("scenario_status", "as_is"))
    has_valid_computed_payload = isinstance(scenario_daily_raw, pd.DataFrame) and len(scenario_daily_raw) > 0
    scenario_payload_missing = bool(requested_status == "computed" and not has_valid_computed_payload)
    if requested_status == "computed" and has_valid_computed_payload:
        scenario_daily = scenario_daily_raw
        scenario_status = "computed"
    else:
        scenario_daily = as_is
        scenario_status = "as_is"
    trust = build_trust_block(results, what_if_result)
    conf_score = _finite((what_if_result or {}).get("confidence_scenario", 0.0))
    conf_label = str((what_if_result or {}).get("confidence_label", "Низкая"))
    as_is_units = _finite(float(as_is["actual_sales"].sum()) if len(as_is) else 0.0)
    as_is_revenue = _finite(float(as_is["revenue"].sum()) if len(as_is) else 0.0)
    as_is_profit = _finite(float(as_is["profit"].sum()) if len(as_is) and "profit" in as_is.columns else 0.0)
    scenario_units = _finite(float(scenario_daily["actual_sales"].sum()) if len(scenario_daily) else 0.0)
    scenario_revenue = _finite(float(scenario_daily["revenue"].sum()) if len(scenario_daily) else 0.0)
    scenario_profit = _finite(float(scenario_daily["profit"].sum()) if len(scenario_daily) and "profit" in scenario_daily.columns else 0.0)
    margin_available = bool(results.get("cost_input_available", False))
    run_summary_cfg = {}
    try:
        run_summary_cfg = json.loads(results.get("analysis_run_summary_json", b"{}").decode("utf-8")).get("config", {})
    except Exception:
        run_summary_cfg = {}
    return {
        "timestamp_utc": str(pd.Timestamp.utcnow()),
        "input_summary": {
            "sku": str(results.get("_trained_bundle", {}).get("base_ctx", {}).get("product_id", "")),
            "category": str(results.get("_trained_bundle", {}).get("base_ctx", {}).get("category", "")),
            "history_rows": int(len(results.get("history_daily", pd.DataFrame()))),
            "history_start": str(pd.to_datetime(results.get("history_daily", pd.DataFrame()).get("date"), errors="coerce").min()),
            "history_end": str(pd.to_datetime(results.get("history_daily", pd.DataFrame()).get("date"), errors="coerce").max()),
        },
        "active_path_summary": {
            "active_path": trust["active_calculation_mode"],
            "fallback_used": trust["fallback_used"],
            "selected_candidate": str(run_summary_cfg.get("selected_candidate", "")),
            "production_selected_candidate": str(run_summary_cfg.get("production_selected_candidate", run_summary_cfg.get("selected_candidate", "legacy_baseline"))),
            "diagnostic_selected_candidate": str(run_summary_cfg.get("diagnostic_selected_candidate", "")),
            "selection_mode": str(run_summary_cfg.get("selection_mode", "diagnostic_comparison_runtime_frozen_to_legacy")),
            "production_selection_reason": str(run_summary_cfg.get("production_selection_reason", "v1_contract_runtime_frozen_to_legacy")),
            "selection_reason": str(run_summary_cfg.get("selection_reason", "")),
            "model_backend": str(run_summary_cfg.get("model_backend", "")),
            "backend_reason": str(run_summary_cfg.get("backend_reason", "")),
            "uplift_mode": "diagnostic_only",
            "uplift_used_in_production": False,
            "scenario_status": scenario_status,
            "price_clip": {
                "requested_price": _finite((what_if_result or {}).get("requested_price", results.get("current_price", 0.0))),
                "model_price": _finite((what_if_result or {}).get("model_price", results.get("current_price", 0.0))),
                "price_clipped": bool((what_if_result or {}).get("price_clipped", False)),
                "clip_reason": str((what_if_result or {}).get("clip_reason", "")),
            },
            "scenario_effective_summary": {
                "requested_price_gross": _finite((what_if_result or {}).get("effective_scenario", {}).get("requested_price_gross", (what_if_result or {}).get("requested_price", results.get("current_price", 0.0)))),
                "applied_price_gross": _finite((what_if_result or {}).get("effective_scenario", {}).get("applied_price_gross", (what_if_result or {}).get("model_price", results.get("current_price", 0.0)))),
                "applied_discount": _finite((what_if_result or {}).get("effective_scenario", {}).get("applied_discount", 0.0)),
                "applied_price_net": _finite((what_if_result or {}).get("effective_scenario", {}).get("applied_price_net", 0.0)),
                "clip_flag": bool((what_if_result or {}).get("effective_scenario", {}).get("price_clipped", (what_if_result or {}).get("price_clipped", False))),
                "clip_reason": str((what_if_result or {}).get("effective_scenario", {}).get("clip_reason", (what_if_result or {}).get("clip_reason", ""))),
            },
        },
        "margin_available": margin_available,
        "baseline_forecast": {
            "units": _finite(float(baseline["actual_sales"].sum()) if len(baseline) else 0.0),
            "revenue": _finite(float(baseline["revenue"].sum()) if len(baseline) else 0.0),
            "profit": _finite(float(baseline["profit"].sum())) if margin_available and len(baseline) and "profit" in baseline.columns else None,
        },
        "as_is_forecast": {"units": as_is_units, "revenue": as_is_revenue, "profit": as_is_profit if margin_available else None},
        "scenario_forecast": {"units": scenario_units, "revenue": scenario_revenue, "profit": scenario_profit if margin_available else None},
        "delta_vs_as_is": {
            "units": _finite(scenario_units - as_is_units),
            "revenue": _finite(scenario_revenue - as_is_revenue),
            "profit": _finite(scenario_profit - as_is_profit) if margin_available else None,
        },
        "scenario_status": scenario_status,
        "price_clip": {
            "requested_price": _finite((what_if_result or {}).get("requested_price", results.get("current_price", 0.0))),
            "model_price": _finite((what_if_result or {}).get("model_price", results.get("current_price", 0.0))),
            "price_clipped": bool((what_if_result or {}).get("price_clipped", False)),
            "clip_reason": str((what_if_result or {}).get("clip_reason", "")),
        },
        "warnings_reliability_notes": trust["warnings"] + (["scenario payload missing; fell back to as-is"] if scenario_payload_missing else []),
        "scenario_confidence": {"score": conf_score, "label": conf_label},
        "data_quality": {"data_quality_label": str(trust.get("data_sufficiency", "unknown")), "data_quality_score": _finite(results.get("_trained_bundle", {}).get("confidence", 0.0))},
        "saved_scenarios": saved_scenarios or {},
    }


def build_scenario_comparison_table(
    as_is_forecast: pd.DataFrame,
    saved_scenarios: Dict[str, Dict[str, Any]],
    show_margin: bool,
) -> pd.DataFrame:
    compare_rows = [
        {
            "scenario": "As-is",
            "units": float(as_is_forecast["actual_sales"].sum()),
            "revenue": float(as_is_forecast["revenue"].sum()),
            "profit": float(as_is_forecast["profit"].sum()),
            "delta_units": 0.0,
            "delta_revenue": 0.0,
            "delta_profit": 0.0,
        }
    ]
    for name in ["Scenario A", "Scenario B", "Scenario C"]:
        if name in saved_scenarios:
            compare_rows.append({"scenario": name, **saved_scenarios[name]})
    compare_df = pd.DataFrame(compare_rows)
    if not show_margin and "profit" in compare_df.columns:
        compare_df["profit"] = np.nan
        compare_df["delta_profit"] = np.nan
    return compare_df


def build_saved_scenario_metrics(
    as_is_forecast: pd.DataFrame,
    scenario_forecast: pd.DataFrame,
    what_if_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    as_is_units = float(as_is_forecast["actual_sales"].sum())
    as_is_revenue = float(as_is_forecast["revenue"].sum())
    as_is_profit = float(as_is_forecast["profit"].sum()) if "profit" in as_is_forecast.columns else float("nan")

    scenario_units = float(scenario_forecast["actual_sales"].sum())
    scenario_revenue = float(scenario_forecast["revenue"].sum())
    scenario_profit = float(scenario_forecast["profit"].sum()) if "profit" in scenario_forecast.columns else float("nan")

    effective = (what_if_result or {}).get("effective_scenario", {})
    return {
        "units": scenario_units,
        "revenue": scenario_revenue,
        "profit": scenario_profit,
        "delta_units": scenario_units - as_is_units,
        "delta_revenue": scenario_revenue - as_is_revenue,
        "delta_profit": scenario_profit - as_is_profit,
        "requested_price_gross": float(effective.get("requested_price_gross", (what_if_result or {}).get("requested_price", np.nan))),
        "applied_price_gross": float(effective.get("applied_price_gross", (what_if_result or {}).get("model_price", np.nan))),
        "applied_price_net": float(effective.get("applied_price_net", np.nan)),
        "applied_discount": float(effective.get("applied_discount", np.nan)),
        "promotion": float(effective.get("promotion", np.nan)),
        "freight_value": float(effective.get("freight_value", np.nan)),
        "price_clipped": bool(effective.get("price_clipped", (what_if_result or {}).get("price_clipped", False))),
        "clip_reason": str(effective.get("clip_reason", (what_if_result or {}).get("clip_reason", ""))),
        "confidence_label": str((what_if_result or {}).get("confidence_label", "Низкая")),
    }


def collect_current_form_values(manual_price: float, discount: float, promo_value: float, freight_mult: float, demand_mult: float, hdays: int) -> Dict[str, float]:
    return {
        "manual_price": float(manual_price),
        "discount": float(discount),
        "promo_value": float(promo_value),
        "freight_mult": float(freight_mult),
        "demand_mult": float(demand_mult),
        "hdays": int(hdays),
    }


def build_applied_scenario_snapshot(
    wr: Dict[str, Any],
    manual_price: float,
    discount: float,
    promo_value: float,
    freight_mult: float,
    demand_mult: float,
    hdays: int,
) -> Dict[str, Any]:
    effective = wr.get("effective_scenario", {}) if isinstance(wr, dict) else {}
    return {
        "manual_price_requested": float(manual_price),
        "manual_price_applied": float(effective.get("applied_price_gross", wr.get("model_price", wr.get("price_for_model", manual_price)))),
        "discount_requested": float(discount),
        "discount_applied": float(effective.get("applied_discount", discount)),
        "net_price_applied": float(effective.get("applied_price_net", wr.get("daily", pd.DataFrame()).get("net_unit_price", pd.Series([np.nan])).iloc[0] if isinstance(wr.get("daily", None), pd.DataFrame) and len(wr.get("daily")) else np.nan)),
        "promo_requested": float(promo_value),
        "promo_applied": float(effective.get("promotion", promo_value)),
        "freight_requested_multiplier": float(freight_mult),
        "freight_applied": float(effective.get("freight_value", np.nan)),
        "price_clipped": bool(effective.get("price_clipped", wr.get("price_clipped", False))),
        "clip_reason": str(effective.get("clip_reason", wr.get("clip_reason", ""))),
        "freight_mult": float(freight_mult),
        "demand_mult": float(demand_mult),
        "horizon_days": int(hdays),
        "scenario_status": str(wr.get("scenario_status", "as_is")),
    }


def get_user_scenario_status(
    current_form: Dict[str, Any],
    base_form: Dict[str, Any],
    applied_snapshot: Optional[Dict[str, Any]],
    current_status: str,
) -> str:
    tol = 1e-9

    def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        for k in a.keys():
            av, bv = a.get(k), b.get(k)
            if isinstance(av, (float, int)) or isinstance(bv, (float, int)):
                if not np.isclose(float(av), float(bv), atol=tol):
                    return False
            elif av != bv:
                return False
        return True

    if applied_snapshot is None:
        return "as_is" if _same(current_form, base_form) else "dirty"
    if str(applied_snapshot.get("scenario_status", "computed")) in {"as_is", "no_effect"}:
        return "as_is" if _same(current_form, base_form) else "dirty"
    applied_form = {
        "manual_price": float(applied_snapshot.get("manual_price_requested", base_form["manual_price"])),
        "discount": float(applied_snapshot.get("discount_requested", base_form["discount"])),
        "promo_value": float(applied_snapshot.get("promo_requested", base_form["promo_value"])),
        "freight_mult": float(applied_snapshot.get("freight_mult", base_form["freight_mult"])),
        "demand_mult": float(applied_snapshot.get("demand_mult", base_form["demand_mult"])),
        "hdays": int(applied_snapshot.get("horizon_days", base_form["hdays"])),
    }
    if _same(current_form, applied_form):
        return "saved" if current_status == "saved" else "applied"
    return "dirty"


def render_scenario_status_banner(status: str, snapshot: Optional[Dict[str, Any]] = None, last_saved_slot: Optional[str] = None) -> None:
    if status == "as_is":
        st.info("Сейчас показана база без изменений. Сценарий ещё не применён.")
    elif status == "dirty":
        st.warning("Вы изменили параметры, но ещё не применили сценарий. Цифры ниже относятся к последнему применённому состоянию.")
    elif status == "saved":
        slot = last_saved_slot or "Scenario A"
        st.success(f"Сценарий применён и сохранён в слот {slot}.")
    elif status == "applied":
        st.success("Сценарий применён. Ниже показан результат именно для этих параметров.")
    else:
        st.info("Состояние сценария не определено.")


def render_applied_scenario_block(snapshot: Optional[Dict[str, Any]]) -> None:
    if not snapshot:
        return
    open_surface("Применённый сценарий")
    c1, c2, c3 = st.columns(3)
    c1.metric("Цена (запрошенная)", f"{float(snapshot.get('manual_price_requested', 0.0)):.2f} ₽")
    c1.metric("Скидка (applied)", f"{float(snapshot.get('discount_applied', 0.0)):.0%}")
    c1.metric("Промо (applied)", f"{float(snapshot.get('promo_applied', 0.0)):.0%}")
    c2.metric("Цена (applied)", f"{float(snapshot.get('manual_price_applied', 0.0)):.2f} ₽")
    c2.metric("Цена net (applied)", f"{float(snapshot.get('net_price_applied', 0.0)):.2f} ₽")
    c2.metric("Логистика (applied)", f"{float(snapshot.get('freight_applied', 0.0)):.2f}")
    c2.metric("Доп. множитель спроса", f"{float(snapshot.get('demand_mult', 1.0)):.2f}x")
    clip_text = "Цена ограничена" if bool(snapshot.get("price_clipped", False)) else "Без ограничений"
    c3.metric("Ограничение цены", clip_text)
    c3.metric("Горизонт", f"{int(snapshot.get('horizon_days', 0))} дн.")
    if bool(snapshot.get("price_clipped", False)):
        st.warning(f"Введённая цена была скорректирована моделью: {snapshot.get('clip_reason', 'ограничение диапазона')}")
    close_surface()


def reset_scenario_ui_state_to_base(results: Dict[str, Any], clear_saved_slot: bool = True) -> Dict[str, Any]:
    results["scenario_forecast"] = None
    results["scenario_price_requested"] = float(results.get("current_price", 0.0))
    results["scenario_price_modeled"] = float(results.get("current_price", 0.0))
    results["scenario_price"] = float(results.get("current_price", 0.0))
    results["manual_scenario_summary_json"] = b"{}"
    results["manual_scenario_daily_csv"] = b""
    st.session_state.what_if_result = None
    st.session_state.applied_scenario_snapshot = None
    st.session_state.scenario_ui_status = "as_is"
    if clear_saved_slot:
        st.session_state.last_saved_slot = None
    base_ctx = results.get("_trained_bundle", {}).get("base_ctx", {})
    st.session_state["what_if_price"] = float(results.get("current_price", 0.0))
    st.session_state["what_if_discount"] = float(base_ctx.get("discount", 0.0))
    st.session_state["what_if_promo"] = float(np.clip(base_ctx.get("promotion", 0.0), 0.0, 1.0))
    st.session_state["what_if_freight_mult"] = 1.0
    st.session_state["what_if_demand_mult"] = 1.0
    st.session_state["what_if_hdays"] = int(CONFIG["HORIZON_DAYS_DEFAULT"])
    st.session_state.form_last_values = collect_current_form_values(
        st.session_state["what_if_price"],
        st.session_state["what_if_discount"],
        st.session_state["what_if_promo"],
        st.session_state["what_if_freight_mult"],
        st.session_state["what_if_demand_mult"],
        st.session_state["what_if_hdays"],
    )
    return results


def build_user_friendly_comparison_table(
    as_is_forecast: pd.DataFrame,
    current_scenario_forecast: Optional[pd.DataFrame],
    saved_scenarios: Dict[str, Dict[str, Any]],
    show_margin: bool = True,
) -> pd.DataFrame:
    as_is_units = float(as_is_forecast["actual_sales"].sum())
    as_is_revenue = float(as_is_forecast["revenue"].sum())
    as_is_profit = float(as_is_forecast["profit"].sum()) if "profit" in as_is_forecast.columns else float("nan")

    def _row(name: str, units: float, revenue: float, profit: float) -> Dict[str, Any]:
        margin = (profit / max(revenue, 1e-9)) * 100 if np.isfinite(profit) else np.nan
        return {
            "Сценарий": name,
            "Спрос": units,
            "Выручка": revenue,
            "Прибыль": profit,
            "Маржа": margin if show_margin else np.nan,
            "Δ спроса": units - as_is_units,
            "Δ выручки": revenue - as_is_revenue,
            "Δ прибыли": profit - as_is_profit if np.isfinite(profit) and np.isfinite(as_is_profit) else np.nan,
        }

    rows: List[Dict[str, Any]] = [_row("База (as-is)", as_is_units, as_is_revenue, as_is_profit)]
    if current_scenario_forecast is not None and len(current_scenario_forecast):
        rows.append(
            _row(
                "Текущий сценарий",
                float(current_scenario_forecast["actual_sales"].sum()),
                float(current_scenario_forecast["revenue"].sum()),
                float(current_scenario_forecast["profit"].sum()) if "profit" in current_scenario_forecast.columns else float("nan"),
            )
        )
    for slot in ["Scenario A", "Scenario B", "Scenario C"]:
        if slot in saved_scenarios:
            s = saved_scenarios[slot]
            row = _row(slot, float(s.get("units", 0.0)), float(s.get("revenue", 0.0)), float(s.get("profit", np.nan)))
            row.update(
                {
                    "Цена gross": float(s.get("applied_price_gross", np.nan)),
                    "Цена net": float(s.get("applied_price_net", np.nan)),
                    "Скидка": float(s.get("applied_discount", np.nan)),
                    "Промо": float(s.get("promotion", np.nan)),
                    "Логистика": float(s.get("freight_value", np.nan)),
                    "Confidence": str(s.get("confidence_label", "")),
                    "Clip": "Да" if bool(s.get("price_clipped", False)) else "Нет",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_user_recommendation(scenario_applied: bool, profit_delta: float, warnings_count: int, confidence_label: str) -> str:
    if not scenario_applied:
        return "Сценарий ещё не применён"
    if profit_delta < 0:
        return "Не рекомендуется"
    if warnings_count > 0 or confidence_label in ("Средняя", "Низкая"):
        return "Требует дополнительной проверки"
    return "Рекомендуется к пилоту"


if __name__ == "__main__":
    st.set_page_config(page_title="What-if Cloud", layout="wide", page_icon="📊")

    if "results" not in st.session_state:
        st.session_state.results = None
    if "what_if_result" not in st.session_state:
        st.session_state.what_if_result = None
    if "app_stage" not in st.session_state:
        st.session_state.app_stage = "upload"
    if "selected_category_for_results" not in st.session_state:
        st.session_state.selected_category_for_results = None
    if "selected_sku_for_results" not in st.session_state:
        st.session_state.selected_sku_for_results = None
    if "scenario_table" not in st.session_state:
        st.session_state.scenario_table = None
    if "sensitivity_df" not in st.session_state:
        st.session_state.sensitivity_df = None
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    if "compare_slot" not in st.session_state:
        st.session_state.compare_slot = "Scenario A"
    if "active_workspace_tab" not in st.session_state:
        st.session_state.active_workspace_tab = "Дашборд"
    if "scenario_ui_status" not in st.session_state:
        st.session_state.scenario_ui_status = "as_is"
    if "applied_scenario_snapshot" not in st.session_state:
        st.session_state.applied_scenario_snapshot = None
    if "form_last_values" not in st.session_state:
        st.session_state.form_last_values = None
    if "last_saved_slot" not in st.session_state:
        st.session_state.last_saved_slot = None

    apply_theme()

    query_page = st.query_params.get("page", "landing")
    if isinstance(query_page, list):
        query_page = query_page[0]

    if query_page == "landing":
        from ui.components import (
            render_landing_nav,
            render_landing_hero_v2,
            render_landing_proof_strip,
            render_landing_controls_and_outputs,
            render_landing_pipeline,
            render_landing_trust_v2,
            render_landing_cta_v2,
            render_landing_footer,
        )

        render_landing_nav()
        render_landing_hero_v2()
        render_landing_proof_strip()
        render_landing_controls_and_outputs()
        render_landing_pipeline()
        render_landing_trust_v2()
        render_landing_cta_v2()
        render_landing_footer()
        if st.button("Перейти в приложение", type="primary", use_container_width=True, key="to_app"):
            st.query_params["page"] = "app"
            st.rerun()
        st.stop()

    from ui.components import (
        render_top_header,
        render_object_header,
        render_action_row,
        render_tabs,
        render_chart_card,
        render_metric_summary_card,
        render_insight_card,
        render_compare_card,
        render_report_card,
        render_warning_card,
        render_empty_state,
        render_debug_expander,
        open_surface,
        close_surface,
    )

    render_top_header()


    def _plot_layout(title: str) -> dict:
        return dict(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F4F7FB"),
            margin=dict(l=20, r=20, t=44, b=20),
            legend=dict(orientation="h", y=1.05, x=0),
        )

    def _render_chart_legend_help(metric_name: str, unit_hint: str) -> None:
        st.caption(
            f"Как читать: зеленая линия — базовый прогноз (без изменений), фиолетовая — сценарий (с вашими параметрами). "
            f"Δ показывает разницу сценарий − база по метрике «{metric_name}» ({unit_hint})."
        )

    def _render_mechanics_explainer(results: Dict[str, Any], what_if_result: Optional[Dict[str, Any]]) -> None:
        wr = what_if_result or {}
        effective = wr.get("effective_scenario", {}) if isinstance(wr, dict) else {}
        model_price = effective.get("applied_price_gross", wr.get("model_price", wr.get("price_for_model", results.get("current_price", "n/a"))))
        requested_price = effective.get("requested_price_gross", wr.get("requested_price", results.get("current_price", "n/a")))
        profit_raw = wr.get("profit_total_raw", "n/a")
        profit_adjusted = wr.get("profit_total_adjusted", "n/a")
        confidence = wr.get("confidence_label", "n/a")

        st.markdown("#### Как работает механизм расчёта")
        st.markdown(
            "1) Пользователь задаёт **gross price**.  \n"
            "2) Система применяет **clip** по историческому диапазону (если нужно).  \n"
            "3) Затем применяется **discount** и формируется **effective net price**.  \n"
            "4) Реакция спроса считается именно от **effective net price**.  \n"
            "5) Выручка/прибыль считаются из applied gross/discount/cost/freight."
        )
        st.markdown("#### Что именно система сравнивает")
        st.markdown(
            f"- Цена: requested `{requested_price}` → modeled `{model_price}`.  \n"
            f"- Discount (applied): `{effective.get('applied_discount', wr.get('scenario_inputs_contract', {}).get('scenario_discount', 'n/a'))}`.  \n"
            f"- Net price (applied): `{effective.get('applied_price_net', wr.get('scenario_inputs_contract', {}).get('scenario_net_price', 'n/a'))}`.  \n"
            f"- Промо: `{effective.get('promotion', 'n/a')}`.  \n"
            f"- Freight: `{effective.get('freight_value', 'n/a')}`.  \n"
            f"- Clip reason: `{effective.get('clip_reason', wr.get('clip_reason', ''))}`.  \n"
            f"- Confidence: `{confidence}`.  \n"
            f"- Прибыль: raw `{profit_raw}` и adjusted `{profit_adjusted}`."
        )

    def _fmt_money(v: float) -> str:
        return f"₽ {v:,.0f}"

    def _fmt_units(v: float) -> str:
        return f"{v:,.1f} шт."


    def render_upload_screen() -> dict[str, Any]:
        st.markdown(
            '<div class="section-title">Мастер запуска анализа</div>'
            '<div class="section-subtitle">3 шага: загрузите файл → проверьте сопоставление колонок → выберите SKU и запустите расчёт</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.2, 1], gap="large")

        with left:
            open_surface("Шаг 1. Загрузка файла")
            st.caption("Поддерживаются форматы CSV и XLSX. Загрузите выгрузку с продажами, ценой и датой.")
            universal_file = st.file_uploader("Файл транзакций", type=["csv", "xlsx"], key="universal_file")
            close_surface()

            open_surface("Шаг 2. Проверка структуры данных")
            universal_txn = None
            schema_valid = False
            schema_errors: list[str] = []
            universal_quality: dict[str, Any] = {}
            raw_for_select = None
            universal_mapping: dict[str, Optional[str]] = {}

            if universal_file is not None:
                preview = read_uploaded_table_safely(universal_file)
                auto_map = build_auto_mapping(list(preview.columns))
                required_fields = [f.name for f in CANONICAL_FIELDS if f.required]
                optional_fields = [f.name for f in CANONICAL_FIELDS if not f.required]
                st.markdown("**Обязательные поля:** " + ", ".join(required_fields))
                st.markdown("**Опциональные поля:** " + ", ".join(optional_fields[:6]) + ("..." if len(optional_fields) > 6 else ""))
                with st.expander("Настроить сопоставление колонок", expanded=True):
                    for f in CANONICAL_FIELDS:
                        choices = ["<не использовать>"] + list(preview.columns)
                        guessed = auto_map.get(f.name)
                        idx = choices.index(guessed) if guessed in choices else 0
                        selected = st.selectbox(
                            f"{f.name} {'*' if f.required else ''}",
                            choices,
                            index=idx,
                            key=f"map_universal_{f.name}",
                            help="* — обязательное поле для запуска расчёта.",
                        )
                        universal_mapping[f.name] = None if selected == "<не использовать>" else selected
                missing_required = validate_mapping_required_columns(universal_mapping)
                if missing_required:
                    schema_errors.append(f"Отсутствуют обязательные поля: {', '.join(missing_required)}")
                else:
                    universal_txn, universal_quality = normalize_transactions(preview, universal_mapping)
                    if universal_quality.get("errors"):
                        schema_errors.extend(list(universal_quality["errors"]))
                    else:
                        schema_valid = True
                        raw_for_select = universal_txn.copy().rename(columns={"category": "product_category_name", "product_id": "product_id"})
                        st.success(f"✅ Данные готовы к расчёту: {len(universal_txn):,} строк после нормализации.")
                        for w in universal_quality.get("warnings", []):
                            st.warning(w)
                        st.caption("Предпросмотр нормализованных данных:")
                        st.dataframe(universal_txn.head(10), use_container_width=True, height=240)
            for err in schema_errors:
                st.error(err)
            close_surface()

        with right:
            open_surface("Шаг 3. Выбор объекта и запуск")
            st.caption("Выберите категорию и SKU, для которого хотите построить сценарий.")
            target_category, target_sku = None, None
            if 'raw_for_select' in locals() and raw_for_select is not None and len(raw_for_select) > 0:
                category_col = "product_category_name" if "product_category_name" in raw_for_select.columns else "category"
                categories = sorted(raw_for_select[category_col].dropna().astype(str).unique())
                target_category = st.selectbox("Категория", categories, key="input_target_category") if categories else None
                if target_category is not None:
                    sku_col = "product_id"
                    skus = sorted(raw_for_select[raw_for_select[category_col].astype(str) == str(target_category)][sku_col].astype(str).dropna().unique())
                    target_sku = st.selectbox("SKU", skus, key="input_target_sku") if skus else None
            horizon_days = st.slider("Горизонт прогноза, дней", 7, 90, int(CONFIG["HORIZON_DAYS_DEFAULT"]), 1)
            if target_category and target_sku:
                st.info(f"Будет рассчитан SKU: **{target_sku}** в категории **{target_category}**.")
            run_requested = st.button("Запустить анализ", type="primary", use_container_width=True)
            close_surface()

        return {
            "universal_txn": universal_txn if 'universal_txn' in locals() else None,
            "schema_valid": schema_valid if 'schema_valid' in locals() else False,
            "target_category": target_category,
            "target_sku": target_sku,
            "run_requested": run_requested,
            "horizon_days": horizon_days,
        }


    ctx = {}
    if st.session_state.results is None:
        ctx = render_upload_screen()
        if ctx.get("run_requested"):
            if ctx.get("universal_txn") is None or not ctx.get("schema_valid"):
                st.error("Схема невалидна. Исправьте ошибки и повторите запуск.")
            elif ctx.get("target_category") is None or ctx.get("target_sku") is None:
                st.error("Выберите категорию и SKU для анализа.")
            else:
                with st.spinner("Выполняется расчет..."):
                    results = run_full_pricing_analysis_universal(
                        ctx["universal_txn"],
                        ctx["target_category"],
                        ctx["target_sku"],
                    )
                    st.session_state.results = copy.deepcopy(results)
                    st.session_state.selected_category_for_results = ctx["target_category"]
                    st.session_state.selected_sku_for_results = ctx["target_sku"]
                    st.session_state.app_stage = "dashboard"
                    st.rerun()
        st.stop()

    r = st.session_state.results
    history_daily = r["history_daily"]
    current_forecast = r["as_is_forecast"]
    baseline_forecast = r["neutral_baseline_forecast"]
    has_applied_scenario = r.get("scenario_forecast") is not None and st.session_state.what_if_result is not None
    scenario_forecast = r["scenario_forecast"] if has_applied_scenario else current_forecast
    last_update = pd.Timestamp.utcnow().strftime("%d.%m.%Y")
    ui_status = st.session_state.get("scenario_ui_status", "as_is")
    status_map = {
        "as_is": ("База без изменений", "#8FA3B8"),
        "dirty": ("Есть неприменённые изменения", "#F6C177"),
        "applied": ("Сценарий применён", "#7AD0A9"),
        "saved": (f"Сценарий сохранён в {st.session_state.get('last_saved_slot', 'слот')}", "#7AD0A9"),
    }
    status_text, status_color = status_map.get(ui_status, ("База без изменений", "#8FA3B8"))

    render_object_header(
        object_title=str(st.session_state.get("selected_sku_for_results", "SKU")),
        status_text=status_text,
        horizon_text=f"{len(current_forecast)} дней",
        last_update=last_update,
        status_color=status_color,
    )
    action_click = render_action_row()
    if action_click == "new":
        r = reset_scenario_ui_state_to_base(r, clear_saved_slot=True)
        st.session_state.results = r
        st.toast("Создан новый сценарий", icon="✚")
        st.rerun()
    elif action_click == "reset_form":
        base_ctx = r.get("_trained_bundle", {}).get("base_ctx", {})
        st.session_state["what_if_price"] = float(r.get("current_price", 0.0))
        st.session_state["what_if_discount"] = float(base_ctx.get("discount", 0.0))
        st.session_state["what_if_promo"] = float(np.clip(base_ctx.get("promotion", 0.0), 0.0, 1.0))
        st.session_state["what_if_freight_mult"] = 1.0
        st.session_state["what_if_demand_mult"] = 1.0
        st.session_state["what_if_hdays"] = int(CONFIG["HORIZON_DAYS_DEFAULT"])
        st.toast("Форма сброшена к базовым значениям", icon="⟲")
        st.rerun()
    elif action_click == "cancel_active":
        r = reset_scenario_ui_state_to_base(r, clear_saved_slot=False)
        st.session_state.results = r
        st.toast("Активный сценарий отменён", icon="⊘")
        st.rerun()
    elif action_click == "compare":
        st.session_state.active_workspace_tab = "Сценарий"
        st.session_state["workspace_tab_radio"] = "Сценарий"
        st.rerun()
    elif action_click == "export":
        st.session_state.active_workspace_tab = "Отчет"
        st.session_state["workspace_tab_radio"] = "Отчет"
        st.rerun()

    tabs = ["Дашборд", "Сценарий", "Факторы", "Диагностика", "Отчет"]
    if "workspace_tab_radio" not in st.session_state or st.session_state["workspace_tab_radio"] not in tabs:
        st.session_state["workspace_tab_radio"] = st.session_state.active_workspace_tab
    active_tab = render_tabs(st.session_state.active_workspace_tab, tabs, key="workspace_tab_radio")
    st.session_state.active_workspace_tab = active_tab

    base_units = float(current_forecast["actual_sales"].sum())
    base_revenue = float(current_forecast["revenue"].sum())
    base_profit = float(current_forecast["profit"].sum()) if "profit" in current_forecast.columns else float("nan")
    sc_units = float(scenario_forecast["actual_sales"].sum())
    sc_revenue = float(scenario_forecast["revenue"].sum())
    sc_profit = float(scenario_forecast["profit"].sum()) if "profit" in scenario_forecast.columns else float("nan")

    if active_tab == "Дашборд":
        st.info("На всех графиках сравнение всегда одинаковое: **База (зелёный)** против **Сценария (фиолетовый)**.")
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=baseline_forecast["date"], y=baseline_forecast["actual_sales"], name="База", line=dict(color="#9DCC84", width=2)))
        fig_d.add_trace(go.Scatter(x=scenario_forecast["date"], y=scenario_forecast["actual_sales"], name="Сценарий", line=dict(color="#6F70FF", width=2.4)))
        fig_d.update_layout(**_plot_layout("Спрос"))
        render_chart_card("Спрос", "Горизонт · шт", fig_d, [("База", f"{base_units:,.0f}"), ("Сценарий", f"{sc_units:,.0f}"), ("Δ", f"{sc_units-base_units:+,.0f}")])
        _render_chart_legend_help("Спрос", "шт")

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=baseline_forecast["date"], y=baseline_forecast["revenue"], name="База", line=dict(color="#9DCC84", width=2)))
        fig_r.add_trace(go.Scatter(x=scenario_forecast["date"], y=scenario_forecast["revenue"], name="Сценарий", line=dict(color="#6F70FF", width=2.4)))
        fig_r.update_layout(**_plot_layout("Выручка"))
        render_chart_card("Выручка", "Горизонт · ₽", fig_r, [("База", f"₽ {base_revenue:,.0f}"), ("Сценарий", f"₽ {sc_revenue:,.0f}"), ("Δ", f"₽ {sc_revenue-base_revenue:+,.0f}")])
        _render_chart_legend_help("Выручка", "₽")

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=baseline_forecast["date"], y=baseline_forecast["profit"], name="База", line=dict(color="#9DCC84", width=2), fill="tozeroy"))
        fig_p.add_trace(go.Scatter(x=scenario_forecast["date"], y=scenario_forecast["profit"], name="Сценарий", line=dict(color="#6F70FF", width=2.4)))
        fig_p.update_layout(**_plot_layout("Прибыль"))
        render_chart_card("Прибыль", "Горизонт · ₽", fig_p, [("База", f"₽ {base_profit:,.0f}"), ("Сценарий", f"₽ {sc_profit:,.0f}"), ("Δ", f"₽ {sc_profit-base_profit:+,.0f}")])
        _render_chart_legend_help("Прибыль", "₽")

        effect_pct = ((sc_units - base_units) / base_units * 100) if base_units else 0.0
        render_metric_summary_card(
            "Итог сценария",
            f"{effect_pct:+.1f}%",
            "Изменение спроса к базе",
            [("База", f"{base_units:,.0f}"), ("Сценарий", f"{sc_units:,.0f}"), ("Δ", f"{sc_units-base_units:+,.0f}")],
        )

        recommendation = "применять" if sc_profit >= base_profit else "не применять"
        render_insight_card([
            f"Спрос: {effect_pct:+.1f}%",
            f"Выручка: ₽ {sc_revenue-base_revenue:+,.0f}",
            f"Прибыль: ₽ {sc_profit-base_profit:+,.0f}",
            f"Рекомендация: {recommendation}",
        ])

    elif active_tab == "Сценарий":
        base_ctx = r["_trained_bundle"]["base_ctx"]
        base_form = collect_current_form_values(
            float(r.get("current_price", 0.0)),
            float(base_ctx.get("discount", 0.0)),
            float(np.clip(base_ctx.get("promotion", 0.0), 0.0, 1.0)),
            1.0,
            1.0,
            int(CONFIG["HORIZON_DAYS_DEFAULT"]),
        )
        for key, val in {
            "what_if_price": base_form["manual_price"],
            "what_if_discount": base_form["discount"],
            "what_if_promo": base_form["promo_value"],
            "what_if_freight_mult": base_form["freight_mult"],
            "what_if_demand_mult": base_form["demand_mult"],
            "what_if_hdays": base_form["hdays"],
        }.items():
            if key not in st.session_state:
                st.session_state[key] = val

        current_form = collect_current_form_values(
            st.session_state["what_if_price"],
            st.session_state["what_if_discount"],
            st.session_state["what_if_promo"],
            st.session_state["what_if_freight_mult"],
            st.session_state["what_if_demand_mult"],
            st.session_state["what_if_hdays"],
        )
        st.session_state.form_last_values = current_form
        st.session_state.scenario_ui_status = get_user_scenario_status(
            current_form,
            base_form,
            st.session_state.applied_scenario_snapshot,
            st.session_state.get("scenario_ui_status", "as_is"),
        )
        render_scenario_status_banner(
            st.session_state.scenario_ui_status,
            st.session_state.applied_scenario_snapshot,
            st.session_state.last_saved_slot,
        )

        open_surface("Форма сценария")
        st.markdown("#### Блок 1. Цена и промо")
        c1, c2, c3 = st.columns(3)
        with c1:
            manual_price = st.number_input("Цена, ₽", min_value=0.01, step=1.0, key="what_if_price")
        with c2:
            discount = st.slider("Скидка", 0.0, 0.95, key="what_if_discount", step=0.01)
        with c3:
            promo_value = st.slider("Промо", 0.0, 1.0, key="what_if_promo", step=0.05)
        st.markdown("#### Блок 2. Внешние условия")
        c4, c5 = st.columns(2)
        with c4:
            freight_mult = st.slider("Логистика, множитель", 0.5, 1.5, key="what_if_freight_mult", step=0.05)
        with c5:
            demand_mult = st.slider("Ручной множитель спроса", 0.7, 1.3, key="what_if_demand_mult", step=0.05)
        st.markdown("#### Блок 3. Горизонт")
        hdays = st.slider("Горизонт, дней", 7, 90, key="what_if_hdays", step=1)

        p1, p2 = st.columns(2)
        p1.markdown(f"- Цена: **{manual_price:.2f} ₽**\n- Скидка: **{discount:.0%}**\n- Промо: **{promo_value:.0%}**")
        p2.markdown(f"- Логистика: **{freight_mult:.2f}x**\n- Доп. множитель спроса: **{demand_mult:.2f}x**\n- Горизонт: **{hdays} дней**")
        if abs(float(manual_price) - float(r["current_price"])) / max(float(r["current_price"]), 1e-9) > 0.30:
            st.warning("Цена отличается от текущей более чем на 30%. Результат может быть менее стабильным для интерпретации.")
        apply_btn = st.button("Применить сценарий", type="primary", use_container_width=True)
        close_surface()

        if apply_btn:
            st.session_state.what_if_result = run_what_if_projection(
                r["_trained_bundle"],
                manual_price=float(manual_price),
                freight_multiplier=float(freight_mult),
                demand_multiplier=float(demand_mult),
                horizon_days=int(hdays),
                overrides={"promotion": float(promo_value), "discount": float(discount)},
            )
            wr = st.session_state.what_if_result
            r["scenario_forecast"] = wr["daily"].copy()
            r["scenario_price_requested"] = float(wr.get("requested_price", manual_price))
            r["scenario_price_modeled"] = float(wr.get("model_price", wr.get("price_for_model", manual_price)))
            r["scenario_price"] = r["scenario_price_modeled"]
            r["manual_scenario_summary_json"], r["manual_scenario_daily_csv"] = build_manual_scenario_artifacts(r, wr)
            st.session_state.applied_scenario_snapshot = build_applied_scenario_snapshot(
                wr,
                manual_price,
                discount,
                promo_value,
                freight_mult,
                demand_mult,
                hdays,
            )
            st.session_state.scenario_ui_status = "applied"
            st.session_state.results = r
            st.rerun()

        open_surface("Слоты сравнения")
        selected_slot = st.selectbox(
            "Куда сохранить текущий сценарий",
            options=["Scenario A", "Scenario B", "Scenario C"],
            key="compare_slot",
        )
        save_to_slot_btn = st.button("Сохранить в слот", use_container_width=True, disabled=r.get("scenario_forecast") is None)
        if save_to_slot_btn and r.get("scenario_forecast") is not None:
            st.session_state.saved_scenarios[selected_slot] = build_saved_scenario_metrics(current_forecast, r["scenario_forecast"], st.session_state.what_if_result)
            st.session_state.last_saved_slot = selected_slot
            st.session_state.scenario_ui_status = "saved"
            st.toast(f"Сценарий сохранён в слот {selected_slot}", icon="✓")
            st.rerun()
        saved_keys = ", ".join(sorted(st.session_state.saved_scenarios.keys())) if st.session_state.saved_scenarios else "пока нет"
        st.caption(f"Сохраненные слоты: {saved_keys}.")
        close_surface()

        render_applied_scenario_block(st.session_state.applied_scenario_snapshot)

        wr = st.session_state.what_if_result
        if wr is None:
            st.info("Сценарий ещё не применён. Чтобы увидеть эффект, задайте параметры и нажмите «Применить сценарий».")
            demand_delta = revenue_delta = profit_delta = margin_delta = 0.0
        else:
            demand_delta = float(wr.get("demand_total", sc_units)) - base_units
            revenue_delta = float(wr.get("revenue_total", sc_revenue)) - base_revenue
            profit_delta = float(wr.get("profit_total_adjusted", sc_profit)) - base_profit
            scenario_margin = (float(wr.get("profit_total_adjusted", sc_profit)) / max(float(wr.get("revenue_total", sc_revenue)), 1e-9)) * 100
            base_margin = (base_profit / max(base_revenue, 1e-9)) * 100
            margin_delta = scenario_margin - base_margin
        render_metric_summary_card(
            "Что изменится относительно текущего плана",
            f"{demand_delta:+,.1f} шт.",
            "Изменение спроса",
            [("Выручка", f"₽ {revenue_delta:+,.0f}"), ("Прибыль", f"₽ {profit_delta:+,.0f}"), ("Маржа", f"{margin_delta:+.2f} п.п.")],
        )
        if wr is not None:
            render_report_card("Эффект сценария простыми словами", [
                f"При текущих настройках сценарий меняет спрос на {demand_delta:+,.1f} шт. относительно текущего плана.",
                f"Ожидаемая выручка меняется на ₽ {revenue_delta:+,.0f}.",
                f"Ожидаемая прибыль меняется на ₽ {profit_delta:+,.0f}.",
                "Это базовая оценка, её нужно интерпретировать вместе с уровнем надёжности.",
            ])

        compare_df = build_user_friendly_comparison_table(current_forecast, r.get("scenario_forecast"), st.session_state.saved_scenarios, True)
        render_compare_card(compare_df)
        st.caption("В таблице всегда есть база и текущий сценарий (если он применён), даже без сохранения в слот.")

        open_surface("Чувствительность")
        sens = build_sensitivity_grid(r["_trained_bundle"], base_price=float(r["current_price"]), runner=run_what_if_projection).head(5)
        fig_s = px.bar(sens, x="price", y="profit_adjusted", color_discrete_sequence=["#6F70FF"])
        fig_s.update_layout(**_plot_layout("Чувствительность"))
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
        st.caption("Этот график показывает, как меняется прибыль при нескольких соседних ценах вокруг текущей.")
        close_surface()
        with st.expander("Как работает механизм сценария", expanded=False):
            _render_mechanics_explainer(r, st.session_state.what_if_result)

    elif active_tab == "Факторы":
        factor_report = r.get("feature_report", pd.DataFrame())
        used = factor_report[factor_report.get("used_in_active_model", False) == True]["feature"].tolist() if len(factor_report) else []
        changed = []
        if st.session_state.what_if_result is not None:
            c = st.session_state.what_if_result.get("scenario_inputs_contract", {})
            for k in ["base_price", "scenario_price", "base_promo", "scenario_promo", "base_freight", "scenario_freight"]:
                if k in c:
                    changed.append((k, c[k]))

        render_report_card("Используемые факторы", used[:12] if used else ["Факторы будут показаны после расчета"]) 
        render_report_card("Управляемые факторы", ["цена — целевая цена", "промо — интенсивность промо", "управляемые входы — ручные множители"]) 
        render_report_card("Внешние факторы", ["сезонность — сезонность", "внешний рынок — внешний спрос", "погода/праздники — календарные эффекты (если есть)"]) 
        render_report_card("Изменено в сценарии", [f"{k}: {v}" for k, v in changed] if changed else ["Изменений пока нет"]) 

    elif active_tab == "Диагностика":
        holdout = r.get("holdout_metrics", pd.DataFrame())
        c1, c2, c3, c4 = st.columns(4)
        wape = float(holdout.get("wape", pd.Series([np.nan])).iloc[0]) if len(holdout) else np.nan
        mape = float(holdout.get("mape", pd.Series([np.nan])).iloc[0]) if len(holdout) else np.nan
        with c1:
            st.metric("WAPE", f"{wape:.3f}" if np.isfinite(wape) else "n/a")
        with c2:
            st.metric("MAPE", f"{mape:.3f}" if np.isfinite(mape) else "n/a")
        with c3:
            st.metric("holdout", "ok" if len(holdout) else "n/a")
        with c4:
            st.metric("Качество", str(build_trust_block(r, st.session_state.what_if_result).get("data_sufficiency", "n/a")))

        render_warning_card(r.get("warnings", []))

        open_surface("Диагностический график")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=history_daily["date"], y=history_daily["sales"], name="Факт", line=dict(color="#9DCC84", width=2)))
        fig_h.add_trace(go.Scatter(x=scenario_forecast["date"], y=scenario_forecast["actual_sales"], name="Прогноз", line=dict(color="#6F70FF", width=2.4)))
        fig_h.update_layout(**_plot_layout("Факт vs прогноз"))
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        close_surface()

        run_summary_ui = json.loads(r.get("analysis_run_summary_json", b"{}").decode("utf-8")) if r.get("analysis_run_summary_json") else {}
        render_debug_expander(run_summary_ui)

    elif active_tab == "Отчет":
        st.markdown("### Отчёт в 1 экране")
        wr = st.session_state.what_if_result or {}
        scenario_applied = wr != {} and r.get("scenario_forecast") is not None
        trust = build_trust_block(r, st.session_state.what_if_result)
        warnings_count = len(trust.get("warnings", []))
        confidence_label = str(wr.get("confidence_label", "Низкая"))
        demand_delta = sc_units - base_units if scenario_applied else np.nan
        revenue_delta = sc_revenue - base_revenue if scenario_applied else np.nan
        profit_delta = sc_profit - base_profit if scenario_applied else np.nan
        recommendation = build_user_recommendation(scenario_applied, float(profit_delta) if np.isfinite(profit_delta) else 0.0, warnings_count, confidence_label)

        if not scenario_applied:
            st.info("Сценарий ещё не применён. Сейчас отчёт показывает только текущий базовый план.")
        elif recommendation == "Рекомендуется к пилоту":
            st.success("Рекомендуется к пилоту: сценарий повышает прибыль и не имеет критичных сигналов риска.")
        elif recommendation == "Не рекомендуется":
            st.error("Не рекомендуется: сценарий снижает прибыль относительно текущего плана.")
        else:
            st.warning("Требует дополнительной проверки: эффект не отрицательный, но есть ограничения по надёжности/предупреждениям.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Решение", recommendation if scenario_applied else "—")
        k2.metric("Δ спроса", _fmt_units(demand_delta) if scenario_applied else "—")
        k3.metric("Δ выручки", _fmt_money(revenue_delta) if scenario_applied else "—")
        k4.metric("Δ прибыли", _fmt_money(profit_delta) if scenario_applied else "—")

        snapshot = st.session_state.applied_scenario_snapshot
        open_surface("Что именно мы проверили")
        if snapshot:
            st.markdown(
                f"- Текущая цена → проверяемая: **{float(r.get('current_price', 0.0)):.2f} ₽ → {float(snapshot.get('manual_price_applied', 0.0)):.2f} ₽**\n"
                f"- Скидка (applied): **{float(snapshot.get('discount_applied', 0.0)):.0%}**\n"
                f"- Промо (applied): **{float(snapshot.get('promo_applied', 0.0)):.0%}**\n"
                f"- Логистика (applied): **{float(snapshot.get('freight_applied', 0.0)):.2f}**\n"
                f"- Множитель спроса: **{float(snapshot.get('demand_mult', 1.0)):.2f}x**\n"
                f"- Горизонт: **{int(snapshot.get('horizon_days', 0))} дней**"
            )
            if bool(snapshot.get("price_clipped", False)):
                st.warning("Введённая цена была скорректирована моделью.")
        else:
            st.info("Пока нет применённого сценария.")
        close_surface()

        if scenario_applied:
            base_margin = (base_profit / max(base_revenue, 1e-9)) * 100
            scenario_margin = (sc_profit / max(sc_revenue, 1e-9)) * 100
            quick_compare = pd.DataFrame(
                [
                    {"Сценарий": "Текущий план", "Спрос": _fmt_units(base_units), "Выручка": _fmt_money(base_revenue), "Прибыль": _fmt_money(base_profit), "Маржа": f"{base_margin:.2f}%"},
                    {"Сценарий": "Ваш сценарий", "Спрос": _fmt_units(sc_units), "Выручка": _fmt_money(sc_revenue), "Прибыль": _fmt_money(sc_profit), "Маржа": f"{scenario_margin:.2f}%"},
                    {"Сценарий": "Разница", "Спрос": _fmt_units(demand_delta), "Выручка": _fmt_money(revenue_delta), "Прибыль": _fmt_money(profit_delta), "Маржа": f"{(scenario_margin - base_margin):+.2f} п.п."},
                ]
            )
            open_surface("Что изменится относительно текущего плана")
            st.dataframe(quick_compare, use_container_width=True, hide_index=True)
            close_surface()
        else:
            st.info("Ваш сценарий пока не применён — блок сравнения появится после Apply.")

        effects = wr.get("effects", {}) if isinstance(wr, dict) else {}
        reasons: List[str] = []
        if scenario_applied:
            pe = float(effects.get("price_effect", 1.0))
            pp = float(effects.get("promo_effect", 1.0))
            fe = float(effects.get("freight_effect", 1.0))
            se = float(effects.get("shock_multiplier_mean", 1.0))
            reasons.append("Изменение цены снижает ожидаемый спрос." if pe < 1 else "Изменение цены повышает ожидаемый спрос.")
            if pp > 1:
                reasons.append("Промо поддерживает рост продаж.")
            if fe < 1:
                reasons.append("Условия логистики сдерживают спрос.")
            reasons.append("Дополнительный множитель спроса усиливает прогноз." if se > 1 else "Дополнительный множитель спроса снижает прогноз.")
            reasons.append(f"Эффект цены: {pe:.2f}x; промо: {pp:.2f}x; логистика: {fe:.2f}x; доп. множитель: {se:.2f}x.")
        else:
            reasons.append("Чтобы получить объяснение причин, сначала примените сценарий.")
        render_report_card("Почему получился такой результат", reasons)

        reliability_text = {
            "Высокая": "Модель видит достаточно опоры в данных, серьёзных предупреждений нет.",
            "Средняя": "Результат полезен как ориентир, но его лучше дополнительно проверить.",
            "Низкая": "В данных мало опоры для такого сценария. Используйте результат осторожно.",
        }
        render_report_card("Надёжность оценки", [f"Уровень: {confidence_label}", reliability_text[confidence_label]] + [str(w) for w in trust.get("warnings", [])[:3]])

        next_steps = [
            "Сохраните сценарий в слот сравнения.",
            "Сравните его с альтернативным сценарием.",
            "Скачайте Excel/CSV и передайте результат команде.",
        ] if scenario_applied and recommendation != "Не рекомендуется" else [
            "Проверьте другой диапазон цены.",
            "Снизьте масштаб изменений сценария.",
            "Пересмотрите настройки промо и логистики.",
        ]
        render_report_card("Что делать дальше", next_steps)

        with st.expander("Пояснение терминов", expanded=False):
            st.markdown("- Текущий план — прогноз без ручных изменений.\n- Ваш сценарий — прогноз после нажатия «Применить сценарий».")

        open_surface("Экспорт")
        has_results = st.session_state.results is not None
        st.download_button(
            "Excel",
            data=st.session_state.results["excel_buffer"] if has_results else b"",
            file_name=f"pricing_report_{st.session_state.get('selected_sku_for_results', 'report')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not has_results,
            use_container_width=True,
        )
        st.download_button(
            "CSV",
            data=st.session_state.results.get("manual_scenario_daily_csv", b"") if has_results else b"",
            file_name="scenario_daily.csv",
            mime="text/csv",
            disabled=(not has_results) or (not scenario_applied),
            use_container_width=True,
        )
        if not scenario_applied:
            st.caption("CSV станет доступен после применения сценария.")
        st.download_button(
            "Сводка",
            data=st.session_state.results.get("manual_scenario_summary_json", b"{}") if has_results else b"",
            file_name="summary.json",
            mime="application/json",
            disabled=(not has_results) or (not scenario_applied),
            use_container_width=True,
        )
        close_surface()

    st.caption("What-if Cloud")
