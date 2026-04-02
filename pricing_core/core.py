from __future__ import annotations
# LEGACY HYBRID ENGINE, NOT USED BY V1 UNIVERSAL PATH

import gc
import logging
import os
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_adapter import (
    build_auto_mapping,
    build_daily_from_transactions,
    normalize_transactions,
    objective_to_weights,
)
from calc_engine import compute_daily_unit_economics, sanitize_discount, sanitize_non_negative
from data_schema import CANONICAL_FIELDS
from recommendation import build_business_recommendation
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
    "FORCE_ENHANCED_MODE": False,
    "SMALL_CAT_L2": 12.0,
    "SMALL_CAT_DEPTH": 4,
    "SMALL_CAT_ITER": 350,
    "SMALL_HGB_L2": 15.0,
    "SMALL_RF_MAX_FEATURES": 0.55,
    "AUGMENT_N": 8,
    "AUGMENT_PRICE_NOISE": 0.05,
    "MAX_TRAIN_ROWS_PER_MODEL": 6000,
    "LARGE_DATA_ROWS": 5000,
    "XL_DATA_ROWS": 20000,
    "LARGE_CAT_ITER": 450,
    "XL_CAT_ITER": 300,
    "LARGE_HGB_MAX_ITER": 280,
    "XL_HGB_MAX_ITER": 200,
    "LARGE_RF_TREES": 180,
    "XL_RF_TREES": 120,
}

USER_FACTOR_PREFIX = "user_factor__"
OBJECTIVE_LABEL_TO_MODE = {
    "Максимум прибыли": "maximize_profit",
    "Максимум выручки": "maximize_revenue",
    "Сохранить объём": "protect_volume",
    "Сбалансированный режим": "balanced_mode",
}
OBJECTIVE_HINTS = {
    "Максимум прибыли": "Максимум прибыли — выбираем вариант с наибольшим ожидаемым денежным результатом.",
    "Максимум выручки": "Максимум выручки — приоритет отдаётся росту оборота.",
    "Сохранить объём": "Сохранить объём — уменьшаем риск падения продаж.",
    "Сбалансированный режим": "Сбалансированный режим — компромисс между прибылью, выручкой и объёмом.",
}

FEATURE_INPUT_GROUPS: Dict[str, str] = {
    "price": "manual",
    "discount": "manual",
    "promotion": "manual",
    "freight_value": "manual",
    "stock": "manual",
    "rating": "manual",
    "review_score": "manual",
    "reviews_count": "manual",
    "cost": "manual",
    "importance": "diagnostic",
    "shap_abs": "diagnostic",
    "feature impact": "diagnostic",
}

FEATURE_HUMAN_MAP: Dict[str, Dict[str, Any]] = {
    "log_freight": {
        "label": "Логистика (сглаженное значение)",
        "meaning": "Логистические расходы после сглаживания, чтобы убрать резкие скачки.",
        "how_to_fill": "Не заполняется вручную. Рассчитывается системой из freight_value.",
        "unit": "безразмерный индекс",
        "tooltip": "Показывает тренд логистики без шума по отдельным дням.",
        "is_derived": True,
    },
    "freight_value": {
        "label": "Логистические расходы",
        "meaning": "Средние затраты на доставку единицы товара.",
        "how_to_fill": "Введите фактические расходы на доставку за единицу товара.",
        "unit": "₽/шт.",
        "tooltip": "Если расход не известен, используйте среднее по последним продажам.",
        "is_derived": False,
    },
    "price_vs_ma90": {"label": "Цена относительно 90-дневной средней", "meaning": "Отношение текущей цены к средней цене за 90 дней.", "how_to_fill": "Авторасчёт по истории цен.", "unit": "коэффициент", "tooltip": "1.00 = цена на уровне среднего; >1 = выше среднего.", "is_derived": True},
    "price_vs_ma28": {"label": "Цена относительно 28-дневной средней", "meaning": "Показывает, насколько текущая цена выше/ниже средней за 28 дней.", "how_to_fill": "Авторасчёт по истории цен.", "unit": "коэффициент", "tooltip": "Используется для оценки среднесрочного отклонения цены.", "is_derived": True},
    "price_vs_ma7": {"label": "Цена относительно 7-дневной средней", "meaning": "Показывает текущее отклонение цены от последней недельной средней.", "how_to_fill": "Авторасчёт по последним 7 дням.", "unit": "коэффициент", "tooltip": "Полезно для коротких ценовых изменений.", "is_derived": True},
    "price_vs_cat_median": {"label": "Цена относительно медианы по категории", "meaning": "Сравнение цены SKU с типичной ценой в категории.", "how_to_fill": "Авторасчёт по категории.", "unit": "коэффициент", "tooltip": ">1 означает цену выше медианы по категории.", "is_derived": True},
    "sales_momentum_7_28": {"label": "Ускорение спроса: 7 дней против 28 дней", "meaning": "Изменение краткосрочного спроса относительно месячного тренда.", "how_to_fill": "Авторасчёт по истории продаж.", "unit": "коэффициент", "tooltip": "Помогает увидеть ускорение/замедление спроса.", "is_derived": True},
    "sales_momentum_28_90": {"label": "Ускорение спроса: 28 дней против 90 дней", "meaning": "Изменение среднесрочного спроса относительно долгосрочного тренда.", "how_to_fill": "Авторасчёт по истории продаж.", "unit": "коэффициент", "tooltip": "Показывает устойчивость тренда спроса.", "is_derived": True},
    "availability_ratio": {"label": "Доступность товара", "meaning": "Доля дней, когда товар был доступен для продажи.", "how_to_fill": "Авторасчёт из stock и истории продаж.", "unit": "%", "tooltip": "Низкая доступность может занижать прогноз спроса.", "is_derived": True},
    "review_score": {"label": "Рейтинг товара", "meaning": "Средняя оценка покупателей (обычно от 1 до 5).", "how_to_fill": "Введите актуальный средний рейтинг карточки товара.", "unit": "баллы (0–5)", "tooltip": "Рейтинг выше — обычно лучше конверсия.", "is_derived": False},
    "rating": {"label": "Рейтинг товара", "meaning": "Средняя оценка покупателей (обычно от 1 до 5).", "how_to_fill": "Введите актуальный средний рейтинг карточки товара.", "unit": "баллы (0–5)", "tooltip": "Используйте значение из витрины маркетплейса.", "is_derived": False},
    "reviews_count": {"label": "Количество отзывов", "meaning": "Сколько отзывов оставили покупатели.", "how_to_fill": "Введите текущее количество отзывов на карточке.", "unit": "шт.", "tooltip": "Большое число отзывов повышает устойчивость сигнала рейтинга.", "is_derived": False},
    "log_reviews": {"label": "Сглаженное количество отзывов", "meaning": "Логарифм количества отзывов для более стабильной оценки.", "how_to_fill": "Рассчитывается автоматически из reviews_count.", "unit": "безразмерный индекс", "tooltip": "Нужен модели для стабильности при очень больших значениях отзывов.", "is_derived": True},
    "trust_signal": {"label": "Сигнал доверия к оценке", "meaning": "Комбинация рейтинга и объёма отзывов в одном индикаторе доверия.", "how_to_fill": "Авторасчёт по рейтингу и отзывам.", "unit": "индекс доверия", "tooltip": "Чем выше, тем надёжнее пользовательская оценка товара.", "is_derived": True},
    "rating_x_reviews": {"label": "Сила рейтинга с учётом числа отзывов", "meaning": "Рейтинг, усиленный количеством отзывов.", "how_to_fill": "Авторасчёт по rating и reviews_count.", "unit": "условные единицы", "tooltip": "Высокий рейтинг при малом числе отзывов влияет слабее.", "is_derived": True},
}


def get_feature_human_description(feature_name: str) -> Dict[str, Any]:
    key = str(feature_name or "").strip()
    if key.startswith(USER_FACTOR_PREFIX):
        source_name = key.replace(USER_FACTOR_PREFIX, "", 1)
        return {
            "label": f"Пользовательский фактор: {source_name}",
            "meaning": "Дополнительный фактор из загруженного файла пользователя.",
            "how_to_fill": "Заполняется в исходном CSV/Excel. В what-if сейчас не редактируется вручную.",
            "is_derived": False,
            "unit": "как в исходных данных",
            "tooltip": "Фактор загружен пользователем и учтён в обучении модели.",
            "input_type": "пользовательский ввод",
        }
    meta = FEATURE_HUMAN_MAP.get(key, {}).copy()
    if not meta:
        is_diag = key.lower() in {"importance", "shap_abs", "feature impact"}
        return {
            "label": key,
            "meaning": "Технический признак модели.",
            "how_to_fill": "Обычно рассчитывается системой автоматически.",
            "is_derived": not FEATURE_INPUT_GROUPS.get(key) == "manual",
            "unit": "—",
            "tooltip": "Технический признак",
            "input_type": "диагностический" if is_diag else "технический признак",
        }
    input_group = FEATURE_INPUT_GROUPS.get(key, "derived" if meta.get("is_derived", True) else "manual")
    input_type = "ручной ввод" if input_group == "manual" else ("диагностический" if input_group == "diagnostic" else "авторасчёт")
    meta["input_type"] = input_type
    return meta


def _impact_level(share_pct: float) -> str:
    if share_pct >= 25:
        return "высокое"
    if share_pct >= 10:
        return "среднее"
    return "низкое"


def _impact_comment(level: str, impact_value: float) -> str:
    direction = "повышает" if impact_value >= 0 else "снижает"
    if level == "высокое":
        return f"Сильное влияние: фактор заметно {direction} прогноз."
    if level == "среднее":
        return f"Умеренное влияние: фактор {direction} прогноз, но не доминирует."
    return f"Слабое влияние: фактор {direction} прогноз, но вклад небольшой."


def _iter_user_factor_columns(columns: List[str]) -> List[str]:
    return [c for c in columns if str(c).startswith(USER_FACTOR_PREFIX)]


def resolve_objective_weights(mode: str) -> Tuple[Dict[str, float], str, Optional[str]]:
    try:
        return objective_to_weights(mode), mode, None
    except ValueError:
        fallback_mode = "balanced_mode"
        return (
            objective_to_weights(fallback_mode),
            fallback_mode,
            "Неизвестный режим оптимизации. Применён безопасный режим: «Сбалансированный режим».",
        )


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


def _quantize_sales_units(value: Any) -> float:
    value_num = sanitize_non_negative(value, fallback=0.0)
    if not np.isfinite(value_num):
        return 0.0
    return float(np.rint(value_num))


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


def fit_feature_stats(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for c in features:
        if c not in df.columns or len(df) == 0:
            stats[c] = 0.0
            continue
        if str(df[c].dtype) in ("object", "category"):
            mode = df[c].astype(str).mode()
            stats[c] = str(mode.iloc[0]) if len(mode) else "unknown"
        else:
            stats[c] = safe_median(df[c], 0.0)
    return stats


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
        daily_agg["freight_value"] = ("price", "count")
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
        daily[col] = daily[col].ffill()

    daily["price"] = daily["price"].fillna(safe_median(daily["price"], 1.0)).clip(lower=0.01)
    daily["price_median"] = daily["price_median"].fillna(daily["price"]).clip(lower=0.01)
    if "freight_value" in sku_df.columns:
        daily["freight_value"] = daily["freight_value"].fillna(safe_median(daily["freight_value"], 0.0)).clip(lower=0.0)
    else:
        daily["freight_value"] = 0.0
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
    if "discount" in out.columns:
        out["discount"] = pd.to_numeric(out["discount"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "stock" in out.columns:
        out["stock"] = pd.to_numeric(out["stock"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "promotion" in out.columns:
        out["promotion"] = pd.to_numeric(out["promotion"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "reviews_count" in out.columns:
        out["reviews_count"] = pd.to_numeric(out["reviews_count"], errors="coerce").fillna(0.0).clip(lower=0.0)

    out["log_sales"] = np.log1p(out["sales"])
    out["log_price"] = np.log(out["price"].clip(lower=0.01))
    out["log_freight"] = np.log1p(out["freight_value"].clip(lower=0.0))

    out["price_lag1"] = out["price"].shift(1)
    out["price_lag7"] = out["price"].shift(7)
    out["price_lag28"] = out["price"].shift(28)

    out["sales_lag1"] = out["sales"].shift(1)
    out["sales_lag7"] = out["sales"].shift(7)
    out["sales_lag14"] = out["sales"].shift(14)
    out["sales_lag28"] = out["sales"].shift(28)

    out["freight_lag1"] = out["freight_value"].shift(1)
    out["freight_lag7"] = out["freight_value"].shift(7)
    out["freight_lag28"] = out["freight_value"].shift(28)
    if "discount" in out.columns:
        out["discount_lag1"] = out["discount"].shift(1)
        out["discount_lag7"] = out["discount"].shift(7)
    if "stock" in out.columns:
        out["stock_lag1"] = out["stock"].shift(1)
        out["stock_lag7"] = out["stock"].shift(7)
    if "review_score" in out.columns:
        out["rating_lag1"] = out["review_score"].shift(1)
        out["rating_lag7"] = out["review_score"].shift(7)

    for win in [7, 28, 90]:
        minp = max(3, min(7, win))
        out[f"sales_ma{win}"] = out["sales"].shift(1).rolling(win, min_periods=minp).mean()
        out[f"price_ma{win}"] = out["price"].shift(1).rolling(win, min_periods=minp).mean()
        out[f"freight_ma{win}"] = out["freight_value"].shift(1).rolling(win, min_periods=minp).mean()

    out["sales_std28"] = out["sales"].shift(1).rolling(28, min_periods=7).std()
    out["price_change_1d"] = out["price"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["price_vs_ma7"] = out["price"] / out["price_ma7"].replace(0, np.nan)
    out["price_vs_ma28"] = out["price"] / out["price_ma28"].replace(0, np.nan)
    out["price_vs_ma90"] = out["price"] / out["price_ma90"].replace(0, np.nan)
    out["price_vs_cat_median"] = out["price"] / out["price_median"].replace(0, np.nan)
    out["freight_to_price"] = out["freight_value"] / out["price"].replace(0, np.nan)
    out["sales_momentum_7_28"] = out["sales_ma7"] / out["sales_ma28"].replace(0, np.nan)
    out["sales_momentum_28_90"] = out["sales_ma28"] / out["sales_ma90"].replace(0, np.nan)
    out["effective_price"] = out["price"] * (1.0 - out.get("discount", 0.0).clip(lower=0.0))
    out["discount_depth"] = out.get("discount", 0.0)
    out["promo_pressure"] = out.get("promotion", 0.0) * out.get("discount", 0.0)
    out["delivery_burden"] = out["freight_value"] / out["price"].replace(0, np.nan)
    out["price_x_discount"] = out["price"] * out.get("discount", 0.0)
    out["stock_x_promotion"] = out.get("stock", 0.0) * out.get("promotion", 0.0)
    out["log_reviews"] = np.log1p(out.get("reviews_count", 0.0))
    out["trust_signal"] = out.get("review_score", 4.5) * out["log_reviews"]
    out["rating_x_reviews"] = out.get("review_score", 4.5) * out.get("reviews_count", 0.0)
    out["freight_x_price"] = out["freight_value"] * out["price"]
    out["promo_x_seasonality"] = out.get("promotion", 0.0) * out["sin_doy"]
    out["rolling_demand"] = out["sales_ma28"].replace(0, np.nan)
    out["availability_ratio"] = out.get("stock", 0.0) / out["rolling_demand"]

    for c in ["price_vs_ma7", "price_vs_ma28", "price_vs_ma90", "price_vs_cat_median", "freight_to_price", "sales_momentum_7_28", "sales_momentum_28_90", "delivery_burden", "availability_ratio"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        med = float(out[c].median()) if not out[c].isna().all() else 0.0
        out[c] = out[c].fillna(med)

    return out


def build_feature_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    return add_leak_free_lag_features(add_time_features(daily))


def derive_feature_spec(df: pd.DataFrame) -> Dict[str, Any]:
    blocked = {"date", "sales", "quantity", "revenue", "log_sales", "pred_sales"}
    cat_cols = [c for c in df.columns if c not in blocked and str(df[c].dtype) in ("object", "category")]
    num_cols = [c for c in df.columns if c not in blocked and c not in cat_cols and c != "product_id"]
    direct = [c for c in num_cols + cat_cols if c in df.columns]
    baseline = [c for c in direct if c not in {"price", "log_price", "price_change_1d", "effective_price", "price_x_discount"}]
    return {
        "direct_features": direct,
        "baseline_features": baseline,
        "cat_features_direct": [c for c in cat_cols if c in direct],
        "cat_features_baseline": [c for c in cat_cols if c in baseline],
    }


DIRECT_FEATURES: List[str] = [
    "log_price", "price", "price_change_1d", "price_lag1", "price_lag7", "price_lag28",
    "price_ma7", "price_ma28", "price_ma90", "price_vs_ma7", "price_vs_ma28",
    "price_vs_ma90", "price_vs_cat_median", "sales_lag1", "sales_lag7", "sales_lag14",
    "sales_lag28", "sales_ma7", "sales_ma28", "sales_ma90", "sales_std28",
    "sales_momentum_7_28", "sales_momentum_28_90", "freight_value", "log_freight",
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
    stats = feature_stats or {}
    for c in features:
        if c not in out.columns:
            out[c] = stats.get(c, 0.0) if isinstance(stats, dict) else ""
        if str(out[c].dtype) in ("object", "category"):
            fallback_cat = str(stats.get(c, "unknown")) if isinstance(stats, dict) else "unknown"
            out[c] = out[c].astype(str).replace({"nan": fallback_cat}).fillna(fallback_cat)
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        fallback = float(stats.get(c, 0.0)) if isinstance(stats, dict) else 0.0
        med = float(out[c].median()) if not out[c].isna().all() and np.isfinite(out[c].median()) else fallback
        if isinstance(stats, dict) and c in stats and np.isfinite(stats[c]):
            med = float(stats[c])
        out[c] = out[c].fillna(med)
        out[c] = out[c].replace([np.inf, -np.inf], med)
    return out


def _numeric_for_non_catboost(X: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_features:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(float)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
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


def build_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_models: int = CONFIG["ENSEMBLE_SIZE"],
    kind: str = "direct",
    small_mode: bool = False,
    cat_features: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None,
) -> List[Any]:
    ensemble: List[Any] = []
    if len(X) == 0:
        raise ValueError("Пустая обучающая выборка.")
    n_rows = int(len(X))
    max_train_rows = int(CONFIG["MAX_TRAIN_ROWS_PER_MODEL"])
    effective_n_models = int(n_models)
    if n_rows >= CONFIG["XL_DATA_ROWS"]:
        effective_n_models = min(effective_n_models, 2)
    elif n_rows >= CONFIG["LARGE_DATA_ROWS"]:
        effective_n_models = min(effective_n_models, 3)

    def _sample_indices(seed_offset: int) -> np.ndarray:
        sample_size = min(n_rows, max_train_rows)
        replace = sample_size >= n_rows
        rng = np.random.default_rng(42 + seed_offset)
        return rng.choice(n_rows, size=sample_size, replace=replace)

    monotone = _make_direct_monotone_constraints(feature_names) if kind == "direct" else [0] * len(feature_names)
    cat_depth = CONFIG["SMALL_CAT_DEPTH"] if small_mode else CONFIG["CAT_DEPTH"]
    cat_iter = CONFIG["SMALL_CAT_ITER"] if small_mode else CONFIG["CAT_ITER"]
    if n_rows >= CONFIG["XL_DATA_ROWS"]:
        cat_iter = min(cat_iter, int(CONFIG["XL_CAT_ITER"]))
        cat_depth = min(cat_depth, 5)
    elif n_rows >= CONFIG["LARGE_DATA_ROWS"]:
        cat_iter = min(cat_iter, int(CONFIG["LARGE_CAT_ITER"]))
    cat_l2 = CONFIG["SMALL_CAT_L2"] if small_mode else 3.0
    cat_features = cat_features or []
    cat_idx = [feature_names.index(c) for c in cat_features if c in feature_names]

    if USE_CATBOOST and CatBoostRegressor is not None:
        for i in range(effective_n_models):
            model = CatBoostRegressor(
                iterations=cat_iter, learning_rate=CONFIG["CAT_LR"], depth=cat_depth,
                loss_function="RMSE", verbose=0, random_seed=42 + i,
                monotone_constraints=monotone, allow_writing_files=False,
                od_type="Iter", od_wait=50, l2_leaf_reg=cat_l2, thread_count=1
            )
            idx = _sample_indices(i)
            fit_x = X.iloc[idx][feature_names].copy()
            for c in cat_features:
                if c in fit_x.columns:
                    fit_x[c] = fit_x[c].astype(str)
            fit_weight = None
            if sample_weight is not None:
                fit_weight = pd.to_numeric(sample_weight.iloc[idx], errors="coerce").fillna(1.0).values
            model.fit(fit_x, y.iloc[idx], cat_features=cat_idx if cat_idx else None, sample_weight=fit_weight)
            ensemble.append(model)
            gc.collect()
        return ensemble

    if USE_HGB and HistGradientBoostingRegressor is not None:
        hgb_max_iter = int(CONFIG["HGB_MAX_ITER"])
        if n_rows >= CONFIG["XL_DATA_ROWS"]:
            hgb_max_iter = min(hgb_max_iter, int(CONFIG["XL_HGB_MAX_ITER"]))
        elif n_rows >= CONFIG["LARGE_DATA_ROWS"]:
            hgb_max_iter = min(hgb_max_iter, int(CONFIG["LARGE_HGB_MAX_ITER"]))
        for i in range(effective_n_models):
            try:
                kwargs = dict(
                    loss="squared_error", learning_rate=CONFIG["HGB_LR"], max_iter=hgb_max_iter,
                    max_depth=CONFIG["HGB_DEPTH"], min_samples_leaf=CONFIG["HGB_MIN_LEAF"],
                    l2_regularization=CONFIG["SMALL_HGB_L2"] if small_mode else 0.1, random_state=42 + i
                )
                model = HistGradientBoostingRegressor(monotonic_cst=monotone, **kwargs)
                idx = _sample_indices(i)
                fit_weight = None
                if sample_weight is not None:
                    fit_weight = pd.to_numeric(sample_weight.iloc[idx], errors="coerce").fillna(1.0).values
                model.fit(_numeric_for_non_catboost(X.iloc[idx][feature_names], cat_features), y.iloc[idx], sample_weight=fit_weight)
                ensemble.append(model)
                gc.collect()
            except Exception:
                pass
        if len(ensemble) > 0:
            return ensemble

    rf_trees = int(CONFIG["RF_TREES"])
    if n_rows >= CONFIG["XL_DATA_ROWS"]:
        rf_trees = min(rf_trees, int(CONFIG["XL_RF_TREES"]))
    elif n_rows >= CONFIG["LARGE_DATA_ROWS"]:
        rf_trees = min(rf_trees, int(CONFIG["LARGE_RF_TREES"]))
    for i in range(effective_n_models):
        max_feat = CONFIG["SMALL_RF_MAX_FEATURES"] if small_mode else "sqrt"
        model = RandomForestRegressor(
            n_estimators=rf_trees, max_depth=CONFIG["RF_DEPTH"], max_features=max_feat,
            random_state=42 + i, n_jobs=1
        )
        idx = _sample_indices(i)
        fit_weight = None
        if sample_weight is not None:
            fit_weight = pd.to_numeric(sample_weight.iloc[idx], errors="coerce").fillna(1.0).values
        model.fit(_numeric_for_non_catboost(X.iloc[idx][feature_names], cat_features), y.iloc[idx], sample_weight=fit_weight)
        ensemble.append(model)
        gc.collect()
    return ensemble


def ensemble_predict(models_local: List[Any], X_local: pd.DataFrame, feature_names: Optional[List[str]] = None, cat_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if len(models_local) == 0:
        raise ValueError("No models in ensemble")
    feature_names = feature_names or list(X_local.columns)
    cat_features = cat_features or []
    preds = []
    for m in models_local:
        if "catboost" in m.__class__.__module__.lower():
            x = X_local[feature_names].copy()
            for c in cat_features:
                if c in x.columns:
                    x[c] = x[c].astype(str)
            preds.append(m.predict(x))
        else:
            preds.append(m.predict(_numeric_for_non_catboost(X_local[feature_names], cat_features)))
    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0, ddof=0)


def predict_direct_log(frame: pd.DataFrame, models_local: List[Any], feature_spec: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    spec = feature_spec or {}
    feature_names = spec.get("direct_features", DIRECT_FEATURES)
    cat_features = spec.get("cat_features_direct", [])
    return ensemble_predict(models_local, frame[feature_names], feature_names=feature_names, cat_features=cat_features)


def predict_baseline_log(frame: pd.DataFrame, models_local: List[Any], feature_spec: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    spec = feature_spec or {}
    feature_names = spec.get("baseline_features", BASELINE_FEATURES)
    cat_features = spec.get("cat_features_baseline", [])
    return ensemble_predict(models_local, frame[feature_names], feature_names=feature_names, cat_features=cat_features)


def structural_predict_log(frame: pd.DataFrame, baseline_models: List[Any], elasticity_map: Dict[str, float], pooled_prior: float, feature_spec: Optional[Dict[str, Any]] = None, price_ref_col: str = "price_ma28") -> Tuple[np.ndarray, np.ndarray]:
    base_log, base_std = predict_baseline_log(frame, baseline_models, feature_spec=feature_spec)
    price = frame["price"].astype(float).values
    ref = frame[price_ref_col].astype(float).values if price_ref_col in frame.columns else frame["price"].astype(float).values
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


def blended_predict_log(frame: pd.DataFrame, direct_models: List[Any], baseline_models: List[Any], elasticity_map: Dict[str, float], pooled_elasticity: float, w_direct: float, feature_spec: Optional[Dict[str, Any]] = None) -> np.ndarray:
    direct_log, _ = predict_direct_log(frame, direct_models, feature_spec=feature_spec)
    struct_log, _ = structural_predict_log(frame, baseline_models, elasticity_map, pooled_elasticity, feature_spec=feature_spec)
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
        "product_id": str(base_ctx.get("product_id", "unknown")),
        "elasticity_bucket": str(dt.to_period("M")),
    }
    for c in _iter_user_factor_columns(list(base_ctx.keys())):
        row[c] = sanitize_non_negative(base_ctx.get(c, 0.0), fallback=0.0)
    return pd.DataFrame([row])


def build_one_step_direct_features(history_df: pd.DataFrame, current_date: pd.Timestamp, base_ctx: Dict[str, Any], history_span_days: int, price_value: float) -> pd.DataFrame:
    h = history_df.sort_values("date").reset_index(drop=True)
    sales_series = h["sales"].astype(float).values if "sales" in h.columns else np.zeros(len(h), dtype=float)
    price_series = h["price"].astype(float).values if "price" in h.columns else np.full(len(h), float(price_value), dtype=float)
    freight_series = h["freight_value"].astype(float).values if "freight_value" in h.columns else np.zeros(len(h), dtype=float)
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
    last_price = float(price_series[-1]) if len(price_series) else current_price
    last_sales = float(sales_series[-1]) if len(sales_series) else 0.0
    last_freight = float(freight_series[-1]) if len(freight_series) else float(base_ctx.get("freight_value", 0.0))
    last_review = float(review_series[-1]) if len(review_series) else float(base_ctx.get("review_score", 4.5))
    history_norm_den = max(1.0, float(history_span_days - 1))
    time_index = float(len(history_df))
    time_index_norm = min(1.5, time_index / history_norm_den)

    row = {
        "date": dt,
        "log_price": float(np.log(current_price)),
        "price": float(current_price),
        "price_change_1d": float(current_price / max(last_price, 1e-9) - 1.0) if len(price_series) >= 1 else 0.0,
        "price_lag1": float(price_series[-1]) if len(price_series) >= 1 else current_price,
        "price_lag7": float(price_series[-7]) if len(price_series) >= 7 else current_price,
        "price_lag28": float(price_series[-28]) if len(price_series) >= 28 else current_price,
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
        "freight_to_price": float(last_freight / max(current_price, 1e-9)),
        "price_ma7": _tail_mean(price_series, 7, current_price),
        "price_ma28": _tail_mean(price_series, 28, current_price),
        "price_ma90": _tail_mean(price_series, 90, current_price),
        "price_vs_ma7": float(current_price / max(_tail_mean(price_series, 7, current_price), 1e-9)),
        "price_vs_ma28": float(current_price / max(_tail_mean(price_series, 28, current_price), 1e-9)),
        "price_vs_ma90": float(current_price / max(_tail_mean(price_series, 90, current_price), 1e-9)),
        "price_vs_cat_median": float(current_price / max(safe_median(price_series, current_price), 1e-9)),
        "review_score": float(last_review),
        "discount": float(base_ctx.get("discount", 0.0)),
        "promotion": float(base_ctx.get("promotion", 0.0)),
        "stock": float(base_ctx.get("stock", 0.0)),
        "reviews_count": float(base_ctx.get("reviews_count", 0.0)),
        "effective_price": float(current_price * (1.0 - max(0.0, float(base_ctx.get("discount", 0.0))))),
        "promo_pressure": float(base_ctx.get("promotion", 0.0)) * float(base_ctx.get("discount", 0.0)),
        "log_reviews": float(np.log1p(max(0.0, float(base_ctx.get("reviews_count", 0.0))))),
        "trust_signal": float(last_review) * float(np.log1p(max(0.0, float(base_ctx.get("reviews_count", 0.0))))),
        "delivery_burden": float(last_freight / max(current_price, 1e-9)),
        "dow": dow, "month": month, "weekofyear": weekofyear, "is_weekend": is_weekend,
        "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "time_index": time_index, "time_index_norm": time_index_norm,
        "category": base_ctx.get("category", "unknown"),
        "product_id": str(base_ctx.get("product_id", "unknown")),
        "elasticity_bucket": str(dt.to_period("M")),
    }
    for c in _iter_user_factor_columns(list(base_ctx.keys())):
        row[c] = sanitize_non_negative(base_ctx.get(c, 0.0), fallback=0.0)
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


def recursive_baseline_forecast(base_history: pd.DataFrame, horizon_df: pd.DataFrame, baseline_models: List[Any], base_ctx: Dict[str, Any], feature_spec: Optional[Dict[str, Any]] = None, feature_stats: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    history = base_history.copy()
    if "freight_value" not in history.columns:
        history["freight_value"] = float(base_ctx.get("freight_value", 0.0))
    if "review_score" not in history.columns:
        history["review_score"] = float(base_ctx.get("review_score", 4.5))
    history_span_days = max(len(base_history), 2)
    outputs = []
    for _, fr in horizon_df.iterrows():
        current_date = pd.Timestamp(fr["date"])
        feat = build_one_step_baseline_features(history, current_date, base_ctx, history_span_days)
        baseline_features = (feature_spec or {}).get("baseline_features", BASELINE_FEATURES)
        cat_features = (feature_spec or {}).get("cat_features_baseline", [])
        feat = clean_feature_frame(feat, baseline_features, feature_stats=feature_stats)
        X = feat[baseline_features]
        pred_log_mean, pred_log_std = ensemble_predict(baseline_models, X, feature_names=baseline_features, cat_features=cat_features)
        pred_log_mean = float(pred_log_mean[0]); pred_log_std = float(pred_log_std[0])
        pred_sales = _quantize_sales_units(np.expm1(pred_log_mean))
        outputs.append(pd.DataFrame({"date": [current_date], "base_pred_log_sales": [pred_log_mean], "base_pred_sales": [pred_sales], "base_pred_std_log": [pred_log_std], "year_month": [str(current_date.to_period("M"))]}))
        append_row = {"date": [current_date], "sales": [pred_sales], "freight_value": [float(base_ctx.get("freight_value", 0.0))], "review_score": [float(base_ctx.get("review_score", 4.5))]}
        for c in _iter_user_factor_columns(list(base_ctx.keys())):
            append_row[c] = [sanitize_non_negative(base_ctx.get(c, 0.0), fallback=0.0)]
        history = pd.concat([history, pd.DataFrame(append_row)], ignore_index=True)
    return pd.concat(outputs, ignore_index=True)


def recursive_direct_forecast(base_history: pd.DataFrame, horizon_df: pd.DataFrame, direct_models: List[Any], base_ctx: Dict[str, Any], price_value: float, feature_spec: Optional[Dict[str, Any]] = None, feature_stats: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    history = base_history.copy()
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
        direct_features = (feature_spec or {}).get("direct_features", DIRECT_FEATURES)
        cat_features = (feature_spec or {}).get("cat_features_direct", [])
        feat = clean_feature_frame(feat, direct_features, feature_stats=feature_stats)
        X = feat[direct_features]
        pred_log_mean, pred_log_std = ensemble_predict(direct_models, X, feature_names=direct_features, cat_features=cat_features)
        pred_log_mean = float(pred_log_mean[0]); pred_log_std = float(pred_log_std[0])
        pred_sales = _quantize_sales_units(np.expm1(pred_log_mean))
        outputs.append(pd.DataFrame({"date": [current_date], "direct_pred_log_sales": [pred_log_mean], "direct_pred_sales": [pred_sales], "direct_pred_std_log": [pred_log_std], "year_month": [str(current_date.to_period("M"))], "price": [price_value]}))
        append_row = {"date": [current_date], "sales": [pred_sales], "price": [price_value], "freight_value": [float(base_ctx.get("freight_value", 0.0))], "review_score": [float(base_ctx.get("review_score", 4.5))]}
        for c in _iter_user_factor_columns(list(base_ctx.keys())):
            append_row[c] = [sanitize_non_negative(base_ctx.get(c, 0.0), fallback=0.0)]
        history = pd.concat([history, pd.DataFrame(append_row)], ignore_index=True)
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


def choose_blend_weight(val_df: pd.DataFrame, direct_models: List[Any], baseline_models: List[Any], elasticity_map: Dict[str, float], pooled_elasticity: float, feature_spec: Optional[Dict[str, Any]] = None) -> Tuple[float, pd.DataFrame]:
    if len(val_df) == 0:
        return 0.5, pd.DataFrame(columns=["w_direct", "rmse", "smape", "wape"])
    direct_log, _ = predict_direct_log(val_df, direct_models, feature_spec=feature_spec)
    struct_log, _ = structural_predict_log(val_df, baseline_models, elasticity_map, pooled_elasticity, feature_spec=feature_spec)
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


def collect_feature_diagnostics(models: List[Any], train_frame: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    empty = {"importance": pd.DataFrame(columns=["feature", "importance"]), "top_prediction_factors": pd.DataFrame(columns=["feature", "shap_abs"]), "has_shap": False}
    if len(models) == 0 or len(feature_names) == 0:
        empty["explainability_note"] = "Детальное объяснение факторов недоступно: нет обученной модели или признаков."
        empty["fallback_used"] = "none"
        empty["factor_table"] = pd.DataFrame()
        return empty
    m0 = models[0]
    if "catboost" not in m0.__class__.__module__.lower() or not hasattr(m0, "get_feature_importance"):
        empty["explainability_note"] = "SHAP недоступен для текущего типа модели. Показана базовая важность признаков, где возможно."
        empty["fallback_used"] = "importance"
        empty["factor_table"] = pd.DataFrame()
        return empty

    sample = train_frame[feature_names].tail(min(200, len(train_frame))).copy()
    try:
        imp = m0.get_feature_importance()
        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        imp_df = pd.DataFrame(columns=["feature", "importance"])

    try:
        shap_vals = m0.get_feature_importance(type="ShapValues", data=sample)
        if shap_vals is not None and len(shap_vals) > 0:
            shap_abs = np.abs(np.asarray(shap_vals)[:, :-1]).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "shap_abs": shap_abs}).sort_values("shap_abs", ascending=False).reset_index(drop=True)
        else:
            shap_df = pd.DataFrame(columns=["feature", "shap_abs"])
    except Exception:
        shap_df = pd.DataFrame(columns=["feature", "shap_abs"])

    merged = imp_df.merge(shap_df, on="feature", how="outer")
    merged["importance"] = pd.to_numeric(merged.get("importance"), errors="coerce").fillna(0.0)
    merged["shap_abs"] = pd.to_numeric(merged.get("shap_abs"), errors="coerce").fillna(0.0)
    has_shap = len(shap_df) > 0
    impact_col = "shap_abs" if has_shap else "importance"
    impact_source = "SHAP" if has_shap else "importance"

    imp_sum = float(merged["importance"].sum()) if len(merged) else 0.0
    impact_sum = float(merged[impact_col].sum()) if len(merged) else 0.0
    merged["importance_share_pct"] = (merged["importance"] / imp_sum * 100.0) if imp_sum > 0 else 0.0
    merged["impact_raw"] = merged[impact_col]
    merged["impact_share_pct"] = (merged[impact_col] / impact_sum * 100.0) if impact_sum > 0 else 0.0
    merged["influence_level"] = merged["impact_share_pct"].map(_impact_level)
    merged["impact_comment"] = [
        _impact_comment(level=str(lv), impact_value=float(val)) for lv, val in zip(merged["influence_level"], merged["impact_raw"])
    ]

    merged["human"] = merged["feature"].map(get_feature_human_description)
    merged["label"] = merged["human"].map(lambda d: d.get("label", ""))
    merged["meaning"] = merged["human"].map(lambda d: d.get("meaning", ""))
    merged["how_to_fill"] = merged["human"].map(lambda d: d.get("how_to_fill", ""))
    merged["unit"] = merged["human"].map(lambda d: d.get("unit", "—"))
    merged["input_type"] = merged["human"].map(lambda d: d.get("input_type", "авторасчёт"))
    merged["is_derived"] = merged["human"].map(lambda d: bool(d.get("is_derived", True)))
    merged["calc_note"] = np.where(merged["is_derived"], "рассчитывается автоматически; не вводится вручную", "ручной ввод")
    merged = merged.drop(columns=["human"])
    merged = merged.sort_values("impact_share_pct", ascending=False).reset_index(drop=True)

    explainability_note = (
        "Показаны SHAP и importance: это технические меры влияния модели, а не деньги и не процент прибыли."
        if has_shap
        else "SHAP недоступен. Показана важность признаков (importance) как fallback. Это техническая мера влияния модели."
    )

    return {
        "importance": imp_df,
        "top_prediction_factors": shap_df.head(15),
        "has_shap": has_shap,
        "fallback_used": impact_source.lower(),
        "impact_source": impact_source,
        "explainability_note": explainability_note,
        "factor_table": merged,
    }


def _monotone_price_multiplier(price_candidate: float, current_price: float, elasticity: float) -> float:
    elasticity = float(np.clip(elasticity, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"]))
    ratio = max(float(price_candidate), 1e-9) / max(float(current_price), 1e-9)
    ratio = float(np.clip(ratio, 0.20, 5.0))
    mult = np.exp(elasticity * np.log(ratio))
    return float(np.clip(mult, 0.15, 3.0))


def _safe_price_delta_pct(current_price: float, recommended_price: float) -> float:
    current_price = float(current_price)
    recommended_price = float(recommended_price)
    if current_price <= 0 or (not np.isfinite(current_price)):
        return 0.0
    return float(((recommended_price - current_price) / current_price) * 100.0)


def _resolve_recommended_price(result_bundle: Dict[str, Any], current_price: float) -> float:
    for key in ("recommended_price", "best_price", "target_price", "optimal_price", "final_price"):
        val = result_bundle.get(key, None)
        try:
            val_num = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val_num) and val_num > 0:
            return float(val_num)
    return float(current_price)


def _resolve_unit_cost(cost_value: Any, current_price: float) -> float:
    raw_cost = sanitize_non_negative(cost_value, fallback=float(current_price) * CONFIG["COST_PROXY_RATIO"])
    current_price = max(0.01, sanitize_non_negative(current_price, fallback=1.0))
    if raw_cost > current_price * 3.0:
        return float(current_price * CONFIG["COST_PROXY_RATIO"])
    return float(raw_cost)


def simulate_horizon_profit(
    base_row: Dict[str, Any],
    price_candidate: float,
    future_dates_df: pd.DataFrame,
    direct_models: List[Any],
    baseline_models: List[Any],
    base_history: pd.DataFrame,
    base_ctx: Dict[str, Any],
    elasticity_map: Dict[str, float],
    pooled_elasticity: float,
    w_direct: float,
    feature_spec: Optional[Dict[str, Any]] = None,
    feature_stats: Optional[Dict[str, float]] = None,
    allow_extrapolate: bool = False,
    baseline_daily_precomputed: Optional[pd.DataFrame] = None,
    direct_current_daily_precomputed: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    train_min = float(base_history["price"].min())
    train_max = float(base_history["price"].max())
    requested_price = float(price_candidate)
    current_price_raw = float(base_row.get("price", base_ctx.get("price", requested_price)))
    if not np.isfinite(current_price_raw) or current_price_raw <= 0:
        current_price_raw = float(base_ctx.get("price", max(train_min, 1.0)))
    current_price_model = current_price_raw if allow_extrapolate else float(np.clip(current_price_raw, train_min, train_max))
    price_for_model = requested_price if allow_extrapolate else float(np.clip(requested_price, train_min, train_max))
    ood = bool(not is_price_plausible(requested_price, train_min, train_max, margin=0.25))
    unit_cost = _resolve_unit_cost(base_row.get("cost", base_ctx.get("cost", current_price_raw * CONFIG["COST_PROXY_RATIO"])), current_price_raw)
    discount_rate = sanitize_discount(base_row.get("discount", base_ctx.get("discount", 0.0)))

    if baseline_daily_precomputed is None:
        baseline_daily = recursive_baseline_forecast(base_history, future_dates_df, baseline_models, base_ctx, feature_spec=feature_spec, feature_stats=feature_stats)
    else:
        baseline_daily = baseline_daily_precomputed.copy()
    if direct_current_daily_precomputed is None:
        direct_current_daily = recursive_direct_forecast(base_history, future_dates_df, direct_models, base_ctx, current_price_model, feature_spec=feature_spec, feature_stats=feature_stats)
    else:
        direct_current_daily = direct_current_daily_precomputed.copy()

    direct_level_ratio = np.array(direct_current_daily["direct_pred_sales"].values / np.maximum(baseline_daily["base_pred_sales"].values, 1e-9), dtype=float)
    direct_level_ratio = np.clip(direct_level_ratio, 0.50, 1.50)
    direct_level_factor = np.power(direct_level_ratio, min(max(w_direct, CONFIG["PRICING_DIRECT_WEIGHT_FLOOR"]), CONFIG["PRICING_DIRECT_WEIGHT_CAP"]))

    future_months = [str(pd.Timestamp(d).to_period("M")) for d in future_dates_df["date"]]
    elasticities = np.array([elasticity_map.get(m, pooled_elasticity) for m in future_months], dtype=float)
    elasticities = np.clip(elasticities, CONFIG["ELASTICITY_FLOOR"], CONFIG["ELASTICITY_CEILING"])
    price_multiplier = np.array([_monotone_price_multiplier(price_for_model, current_price_model, e) for e in elasticities], dtype=float)

    pred_sales = np.rint(np.maximum(0.0, baseline_daily["base_pred_sales"].values * direct_level_factor * price_multiplier))
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
        "cost": float(unit_cost),
        "discount": float(discount_rate),
    })
    daily, econ_checks = compute_daily_unit_economics(daily, unit_price_col="price", unit_cost_col="cost", quantity_col="pred_sales", discount_col="discount")
    daily["pred_sales"] = daily["pred_quantity"]

    std_mean = float(np.nanmean(daily["pred_std_log"].values)) if len(daily) else 0.0
    total_demand = float(np.nansum(daily["pred_sales"].values)) if len(daily) else 0.0
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
        "price_clipped": bool(abs(requested_price - price_for_model) > 1e-9),
        "daily": daily,
        "daily_prices": daily["price"].values if len(daily) else np.array([]),
        "daily_demands": daily["pred_sales"].values if len(daily) else np.array([]),
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
        "sanity_warnings": econ_checks.get("sanity_warnings", []),
    }


def recommend_price_horizon(base_row: Dict[str, Any], direct_models: List[Any], baseline_models: List[Any], base_history: pd.DataFrame, base_ctx: Dict[str, Any], elasticity_map: Dict[str, float], pooled_elasticity: float, w_direct: float, feature_spec: Optional[Dict[str, Any]] = None, feature_stats: Optional[Dict[str, float]] = None, objective_weights: Optional[Dict[str, float]] = None, min_rel: float = CONFIG["MIN_REL_STEP"], max_rel: float = CONFIG["MAX_REL_STEP"], n_grid: int = 120, n_days: int = None, risk_lambda: float = 0.7, price_change_penalty: float = None, search_margin: float = None, allow_extrapolate: bool = False) -> Dict[str, Any]:
    if n_days is None:
        n_days = CONFIG["HORIZON_DAYS_DEFAULT"]
    if price_change_penalty is None:
        price_change_penalty = CONFIG["PRICE_CHANGE_PENALTY_SCALE"]
    if search_margin is None:
        search_margin = CONFIG["SEARCH_MARGIN"]
    if objective_weights is None:
        objective_weights = objective_to_weights("maximize_profit")
    future_dates_df = forecast_future_dates(pd.Timestamp(base_history["date"].max()), n_days=n_days)
    train_min = float(base_history["price"].min())
    train_max = float(base_history["price"].max())
    train_span = max(1e-9, train_max - train_min)
    current_price_raw = float(base_row.get("price", base_history["price"].median()))
    if not np.isfinite(current_price_raw) or current_price_raw <= 0:
        current_price_raw = float(base_history["price"].median())
    current_price_model = current_price_raw if allow_extrapolate else float(np.clip(current_price_raw, train_min, train_max))
    unit_cost = _resolve_unit_cost(base_row.get("cost", base_ctx.get("cost", current_price_raw * CONFIG["COST_PROXY_RATIO"])), current_price_raw)

    window_min = max(unit_cost * 1.01, current_price_model * (1.0 + min_rel), train_min - search_margin * train_span)
    window_max = min(current_price_model * (1.0 + max_rel), train_max + search_margin * train_span)
    if not allow_extrapolate:
        window_min = max(window_min, train_min)
        window_max = min(window_max, train_max)
    if not np.isfinite(window_min) or not np.isfinite(window_max) or window_min >= window_max:
        window_min = max(unit_cost * 1.01, train_min)
        window_max = max(window_min + 1e-6, train_max)

    prices = np.linspace(window_min, window_max, max(2, int(n_grid)))
    results = []
    baseline_daily_precomputed = recursive_baseline_forecast(base_history, future_dates_df, baseline_models, base_ctx, feature_spec=feature_spec, feature_stats=feature_stats)
    direct_current_daily_precomputed = recursive_direct_forecast(base_history, future_dates_df, direct_models, base_ctx, current_price_model, feature_spec=feature_spec, feature_stats=feature_stats)

    current_sim = simulate_horizon_profit(
        base_row,
        current_price_raw,
        future_dates_df,
        direct_models,
        baseline_models,
        base_history,
        base_ctx,
        elasticity_map,
        pooled_elasticity,
        w_direct,
        allow_extrapolate=allow_extrapolate,
        baseline_daily_precomputed=baseline_daily_precomputed,
        direct_current_daily_precomputed=direct_current_daily_precomputed,
    )
    current_profit = float(np.nansum(current_sim["daily_profits"]))
    current_adjusted = float(current_sim["adjusted_profit"])
    current_demand = float(np.nansum(current_sim["daily_demands"]))
    current_revenue = float(np.nansum(current_sim["daily"]["price"].values * current_sim["daily"]["pred_sales"].values)) if len(current_sim["daily"]) else 0.0
    current_margin = current_profit / max(current_revenue, 1e-9)
    hist_daily_revenue = pd.to_numeric(base_history.get("revenue", pd.Series(dtype=float)), errors="coerce")
    hist_daily_sales = pd.to_numeric(base_history.get("sales", pd.Series(dtype=float)), errors="coerce")
    revenue_scale = float(max(abs(current_revenue), abs(hist_daily_revenue.mean()) * max(int(n_days), 1), 1.0)) if len(hist_daily_revenue) else float(max(abs(current_revenue), 1.0))
    volume_scale = float(max(abs(current_demand), abs(hist_daily_sales.mean()) * max(int(n_days), 1), 1.0)) if len(hist_daily_sales) else float(max(abs(current_demand), 1.0))
    profit_scale = float(max(abs(current_adjusted), abs(current_profit), 1.0))
    risk_scale = float(max(abs(current_adjusted), abs(current_profit), revenue_scale * 0.1, 1.0))

    for p in prices:
        sim = simulate_horizon_profit(
            base_row,
            float(p),
            future_dates_df,
            direct_models,
            baseline_models,
            base_history,
            base_ctx,
            elasticity_map,
            pooled_elasticity,
            w_direct,
            feature_spec=feature_spec,
            feature_stats=feature_stats,
            allow_extrapolate=allow_extrapolate,
            baseline_daily_precomputed=baseline_daily_precomputed,
            direct_current_daily_precomputed=direct_current_daily_precomputed,
        )
        demand_sum = float(np.nansum(sim["daily_demands"]))
        revenue_sum = float(np.nansum(sim["daily"]["price"].values * sim["daily"]["pred_sales"].values)) if len(sim["daily"]) else 0.0
        margin_ratio = float(sim["total_profit"] / max(revenue_sum, 1e-9))
        demand_drop_rel = (demand_sum - current_demand) / max(current_demand, 1e-9)
        mean_std = float(np.nanmean(sim["std_log"])) if len(sim["std_log"]) else 0.0
        margin = max(float(sim["price_for_model"]) - unit_cost, 0.0)
        scale = max(margin * max(current_demand, 1.0), 1.0)
        demand_penalty = risk_lambda * max(0.0, -demand_drop_rel) * scale
        jump_penalty = price_change_penalty * abs((float(p) - current_price_model) / max(current_price_model, 1e-9)) * scale
        quad_penalty = CONFIG["QUAD_PENALTY_COEF"] * ((float(p) - current_price_model) ** 2) * scale
        adjusted = float(sim["adjusted_profit"] - demand_penalty - jump_penalty - quad_penalty)
        risk_raw = (
            max(0.0, -demand_drop_rel)
            + float(mean_std)
            + float(sim["uncertainty_penalty"])
            + float(sim["disagreement_penalty"])
            + float(demand_penalty + jump_penalty + quad_penalty)
        )
        results.append(
            {
                "price": float(p),
                "raw_profit": float(sim["total_profit"]),
                "adjusted_profit": float(adjusted),
                "demand_sum": demand_sum,
                "revenue_sum": revenue_sum,
                "margin_ratio": margin_ratio,
                "risk_raw": float(risk_raw),
                "objective_score": 0.0,
                "mean_std": mean_std,
                "daily": sim["daily"],
                "ood_flag": bool(sim["ood_flag"]),
                "uncertainty_penalty": float(sim["uncertainty_penalty"]),
                "disagreement_penalty": float(sim["disagreement_penalty"]),
                "demand_penalty": float(demand_penalty),
                "jump_penalty": float(jump_penalty),
                "quad_penalty": float(quad_penalty),
            }
        )

    if len(results) == 0:
        return {"prices": np.array([]), "results": [], "best_idx": None, "best_price": None, "best_profit_adjusted": None, "best_profit_raw": None, "best_daily": None, "ml_best_price": None, "ml_best_profit_raw": None, "ml_best_daily": None, "current_profit_raw": current_profit, "current_profit_adjusted": current_adjusted, "explain": "No valid candidate prices found within search window."}

    for r in results:
        p_score = float(r["adjusted_profit"] / profit_scale)
        rv_score = float(r["revenue_sum"] / revenue_scale)
        v_score = float(r["demand_sum"] / volume_scale)
        m_score = float(np.clip(r["margin_ratio"], -1.0, 1.0))
        rk_score = float(r["risk_raw"] / risk_scale)
        objective_score = (
            float(objective_weights.get("profit", 0.0)) * p_score
            + float(objective_weights.get("revenue", 0.0)) * rv_score
            + float(objective_weights.get("volume", 0.0)) * v_score
            + float(objective_weights.get("margin", 0.0)) * m_score
            - float(objective_weights.get("risk", 0.0)) * rk_score
        )
        r["score_components"] = {
            "profit_score": float(p_score),
            "revenue_score": float(rv_score),
            "volume_score": float(v_score),
            "margin_score": float(m_score),
            "risk_penalty": float(rk_score),
        }
        r["objective_score"] = float(objective_score)

    best_idx = int(np.nanargmax([r["objective_score"] for r in results]))
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
        "explain": {"search_min": float(window_min), "search_max": float(window_max), "n_candidates": len(results), "train_range": (float(train_min), float(train_max)), "current_price": float(current_price_raw), "pricing_w_direct": float(min(max(w_direct, CONFIG["PRICING_DIRECT_WEIGHT_FLOOR"]), CONFIG["PRICING_DIRECT_WEIGHT_CAP"])), "objective_weights": dict(objective_weights), "scaling_reference": {"profit_scale": float(profit_scale), "revenue_scale": float(revenue_scale), "volume_scale": float(volume_scale), "risk_scale": float(risk_scale)}},
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


def run_full_pricing_analysis(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    target_category: str,
    target_sku: str,
    horizon_days: int = CONFIG["HORIZON_DAYS_DEFAULT"],
    risk_lambda: float = 0.7,
):
    raw_df = build_raw_frame(orders, order_items, products, reviews)
    if len(raw_df) == 0:
        raise ValueError("После объединения данных не осталось строк.")
    sku_df = raw_df[(raw_df["product_category_name"] == target_category) & (raw_df["product_id"].astype(str) == str(target_sku))].copy()
    if len(sku_df) == 0:
        raise ValueError("Для выбранной категории и SKU нет данных.")

    total_sales = len(sku_df)
    small_data_mode = total_sales < CONFIG["MIN_ML_SALES"] or CONFIG.get("FORCE_ENHANCED_MODE", False)

    raw_missing_share = float(sku_df.isna().mean().mean()) if len(sku_df) else 1.0
    daily_base = build_daily_sku_frame(sku_df, target_sku)
    if small_data_mode:
        daily_base = robust_clean_dirty_data(daily_base)
    daily_base = build_feature_matrix(daily_base).dropna(subset=["sales", "price", "log_sales", "log_price"]).reset_index(drop=True)
    if len(daily_base) < 5:
        raise ValueError("Слишком мало дневных наблюдений после агрегации для обучения модели.")

    feature_spec = derive_feature_spec(daily_base)
    direct_features = feature_spec["direct_features"]
    baseline_features = feature_spec["baseline_features"]
    all_feats = list(dict.fromkeys(direct_features + baseline_features))
    n = len(daily_base)
    train_end, val_end = _safe_split_sizes(n)
    train_raw = daily_base.iloc[:train_end].copy().reset_index(drop=True)
    val_raw = daily_base.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_raw = daily_base.iloc[val_end:].copy().reset_index(drop=True)

    feature_stats = fit_feature_stats(train_raw, all_feats)

    train_df = clean_feature_frame(train_raw, all_feats, feature_stats)
    val_df = clean_feature_frame(val_raw, all_feats, feature_stats)
    test_df = clean_feature_frame(test_raw, all_feats, feature_stats)

    if small_data_mode and len(train_df) < 300:
        train_df = clean_feature_frame(augment_price_variations(train_df), all_feats, feature_stats)

    X_train_direct = train_df[direct_features].copy()
    y_train = train_df["log_sales"].astype(float).copy()
    X_train_base = train_df[baseline_features].copy()

    base_ctx = current_price_context(daily_base)
    base_ctx["category"] = target_category
    base_ctx["product_id"] = target_sku

    fixed_log_price_coef = estimate_pooled_elasticity(train_df, small_mode=small_data_mode)
    shrunk_random_effects, _ = compute_monthly_group_elasticities(train_df, fixed_log_price_coef, small_mode=small_data_mode)

    daily_base["elasticity_group"] = daily_base["date"].dt.to_period("M").astype(str).map(shrunk_random_effects).fillna(fixed_log_price_coef)
    train_df = clean_feature_frame(train_df, all_feats, feature_stats)
    val_df = clean_feature_frame(val_df, all_feats, feature_stats)
    test_df = clean_feature_frame(test_df, all_feats, feature_stats)

    direct_models = build_models(X_train_direct, y_train, direct_features, n_models=CONFIG["ENSEMBLE_SIZE"], kind="direct", small_mode=small_data_mode, cat_features=feature_spec["cat_features_direct"])
    baseline_models = build_models(X_train_base, y_train, baseline_features, n_models=CONFIG["ENSEMBLE_SIZE"], kind="baseline", small_mode=small_data_mode, cat_features=feature_spec["cat_features_baseline"])

    w_direct, _ = choose_blend_weight(val_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, feature_spec=feature_spec)

    holdout_metrics = eval_prediction_frame(test_df, blended_predict_log(test_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec), label="holdout") if len(test_df) > 0 else {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "smape": float("nan"), "wape": float("nan"), "sigma_log": float("nan")}
    feature_diagnostics = collect_feature_diagnostics(direct_models, train_df, direct_features)
    logging.info("Feature inventory | direct=%s | baseline=%s | categorical_direct=%s", direct_features, baseline_features, feature_spec["cat_features_direct"])
    history_days = int((daily_base["date"].max() - daily_base["date"].min()).days + 1)
    missing_share = float(daily_base.isna().mean().mean()) if len(daily_base) else 1.0
    data_quality = assess_data_quality(history_days, len(daily_base), missing_share, float(holdout_metrics.get("wape", float("nan"))))
    data_quality["raw_missing_share"] = raw_missing_share
    data_quality["post_clean_missing_share"] = missing_share

    latest_row = dict(base_ctx)
    latest_row["requested_price"] = float(base_ctx.get("price", train_df["price"].median()))
    latest_row["_ood_flag"] = bool(not is_price_plausible(float(latest_row["requested_price"]), float(train_df["price"].min()), float(train_df["price"].max()), margin=0.25))

    obj_w = objective_to_weights("maximize_profit")
    rec = recommend_price_horizon(
        latest_row,
        direct_models,
        baseline_models,
        daily_base,
        base_ctx,
        shrunk_random_effects,
        fixed_log_price_coef,
        w_direct,
        feature_spec=feature_spec,
        feature_stats=feature_stats,
        objective_weights=obj_w,
        n_days=int(horizon_days),
        risk_lambda=float(risk_lambda),
    )
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()), n_days=int(horizon_days))

    current_sim = simulate_horizon_profit(latest_row, float(base_ctx.get("price")), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec, feature_stats=feature_stats)
    optimal_sim = simulate_horizon_profit(latest_row, float(rec["best_price"]), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec, feature_stats=feature_stats)

    profit_curve_df = pd.DataFrame([{"price": p, "adjusted_profit": simulate_horizon_profit(latest_row, p, future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec, feature_stats=feature_stats)["adjusted_profit"]} for p in np.linspace(float(base_ctx.get("price")) * 0.8, float(base_ctx.get("price")) * 1.2, 50)])

    best_row_meta = rec["results"][int(rec["best_idx"])] if rec.get("results") and rec.get("best_idx") is not None else {}
    reason_hints = {
        "key_driver_positive": "Улучшение маржи и итоговой прибыли",
        "key_driver_negative": "Неопределённость спроса и риск просадки объёма",
        "reason_why_this_scenario_wins": "Сценарий цены лидирует по скорингу objective с учётом risk penalties.",
    }
    if best_row_meta:
        if float(best_row_meta.get("demand_penalty", 0.0)) > 0:
            reason_hints["key_driver_negative"] = "Штраф за риск снижения объёма"
        elif float(best_row_meta.get("disagreement_penalty", 0.0)) > 0:
            reason_hints["key_driver_negative"] = "Расхождение прогнозов моделей (model disagreement)"

    confidence_legacy = float(max(0.0, min(1.0, (1.0 / (1.0 + max(0.0, holdout_metrics.get("wape", 100.0) / 100.0))) * data_quality["confidence_cap"])))
    biz_rec = build_business_recommendation(
        current_price=float(base_ctx.get("price")),
        recommended_price=float(rec.get("best_price")),
        current_profit=float(current_sim.get("adjusted_profit", 0.0)),
        recommended_profit=float(optimal_sim.get("adjusted_profit", 0.0)),
        confidence=confidence_legacy,
        elasticity=float(fixed_log_price_coef),
        history_days=history_days,
        current_revenue=float(current_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in current_sim else None,
        recommended_revenue=float(optimal_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else None,
        current_volume=float(current_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in current_sim else None,
        recommended_volume=float(optimal_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else None,
        data_quality=data_quality,
        base_ctx=base_ctx,
        reason_hints=reason_hints,
    )

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
        "current_revenue": float(current_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in current_sim else 0.0,
        "best_revenue": float(optimal_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else 0.0,
        "current_volume": float(current_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in current_sim else 0.0,
        "best_volume": float(optimal_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else 0.0,
        "profit_lift_pct": profit_lift_pct,
        "data_quality": data_quality,
        "feature_diagnostics": feature_diagnostics,
        "excel_buffer": excel_buffer,
        "flag": decision_flag(latest_row, rec),
        "business_recommendation": biz_rec,
        "_trained_bundle": {
            "direct_models": direct_models,
            "baseline_models": baseline_models,
            "feature_spec": feature_spec,
            "feature_stats": feature_stats,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "elasticity_map": shrunk_random_effects,
            "pooled_elasticity": fixed_log_price_coef,
            "w_direct": w_direct,
            "objective_weights": obj_w,
            "confidence": confidence_legacy,
            "data_quality": data_quality,
            "feature_diagnostics": feature_diagnostics,
        },
    }


def run_full_pricing_analysis_universal(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    objective_mode: str = "maximize_profit",
    objective_weights_override: Optional[Dict[str, float]] = None,
    horizon_days: int = CONFIG["HORIZON_DAYS_DEFAULT"],
    risk_lambda: float = 0.7,
):
    txn = normalized_txn.copy()
    if len(txn) == 0:
        raise ValueError("Пустой датасет после нормализации.")
    sku_df = txn[(txn["category"].astype(str) == str(target_category)) & (txn["product_id"].astype(str) == str(target_sku))].copy()
    if len(sku_df) == 0:
        raise ValueError("Для выбранной категории и SKU нет данных.")

    daily_base = build_daily_from_transactions(txn, target_sku)
    daily_base = robust_clean_dirty_data(daily_base)
    daily_base = build_feature_matrix(daily_base).dropna(subset=["sales", "price", "log_sales", "log_price"]).reset_index(drop=True)
    if len(daily_base) < 5:
        raise ValueError("Слишком мало дневных наблюдений после агрегации.")

    feature_spec = derive_feature_spec(daily_base)
    direct_features = feature_spec["direct_features"]
    baseline_features = feature_spec["baseline_features"]
    all_feats = list(dict.fromkeys(direct_features + baseline_features))
    n = len(daily_base)
    train_end, val_end = _safe_split_sizes(n)
    train_raw = daily_base.iloc[:train_end].copy().reset_index(drop=True)
    val_raw = daily_base.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_raw = daily_base.iloc[val_end:].copy().reset_index(drop=True)

    feature_stats = fit_feature_stats(train_raw, all_feats)
    train_df = clean_feature_frame(train_raw, all_feats, feature_stats)
    val_df = clean_feature_frame(val_raw, all_feats, feature_stats)
    test_df = clean_feature_frame(test_raw, all_feats, feature_stats)

    X_train_direct = train_df[direct_features].copy()
    y_train = train_df["log_sales"].astype(float).copy()
    X_train_base = train_df[baseline_features].copy()

    base_ctx = current_price_context(daily_base)
    base_ctx["category"] = target_category
    base_ctx["product_id"] = target_sku
    fixed_log_price_coef = estimate_pooled_elasticity(train_df, small_mode=True)
    shrunk_random_effects, _ = compute_monthly_group_elasticities(train_df, fixed_log_price_coef, small_mode=True)
    direct_models = build_models(X_train_direct, y_train, direct_features, kind="direct", small_mode=True, cat_features=feature_spec["cat_features_direct"])
    baseline_models = build_models(X_train_base, y_train, baseline_features, kind="baseline", small_mode=True, cat_features=feature_spec["cat_features_baseline"])
    w_direct, _ = choose_blend_weight(val_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, feature_spec=feature_spec)
    holdout_metrics = eval_prediction_frame(test_df, blended_predict_log(test_df, direct_models, baseline_models, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec), label="holdout") if len(test_df) > 0 else {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "smape": float("nan"), "wape": float("nan"), "sigma_log": float("nan")}
    feature_diagnostics = collect_feature_diagnostics(direct_models, train_df, direct_features)
    logging.info("Feature inventory | direct=%s | baseline=%s | categorical_direct=%s", direct_features, baseline_features, feature_spec["cat_features_direct"])

    latest_row = dict(base_ctx)
    latest_row["requested_price"] = float(base_ctx.get("price", train_df["price"].median()))
    if objective_weights_override is not None:
        obj_w = objective_weights_override
    else:
        obj_w, objective_mode, _ = resolve_objective_weights(objective_mode)
    rec = recommend_price_horizon(
        latest_row,
        direct_models,
        baseline_models,
        daily_base,
        base_ctx,
        shrunk_random_effects,
        fixed_log_price_coef,
        w_direct,
        feature_spec=feature_spec,
        feature_stats=feature_stats,
        objective_weights=obj_w,
        n_days=int(horizon_days),
        risk_lambda=float(risk_lambda),
    )
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()), n_days=int(horizon_days))
    current_sim = simulate_horizon_profit(latest_row, float(base_ctx.get("price")), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec, feature_stats=feature_stats)
    optimal_sim = simulate_horizon_profit(latest_row, float(rec["best_price"]), future_dates, direct_models, baseline_models, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef, w_direct, feature_spec=feature_spec, feature_stats=feature_stats)
    history_days = int((daily_base["date"].max() - daily_base["date"].min()).days + 1)
    missing_share = float(txn.isna().mean().mean()) if len(txn) else 1.0
    post_clean_missing_share = float(daily_base.isna().mean().mean()) if len(daily_base) else 1.0
    data_quality = assess_data_quality(history_days, len(daily_base), missing_share, float(holdout_metrics.get("wape", float("nan"))))
    data_quality["raw_missing_share"] = missing_share
    data_quality["post_clean_missing_share"] = post_clean_missing_share
    raw_confidence = float(1.0 / (1.0 + max(0.0, holdout_metrics.get("wape", 100.0) / 100.0)))
    confidence = float(max(0.0, min(1.0, raw_confidence * data_quality["confidence_cap"])))
    best_row_meta = rec["results"][int(rec["best_idx"])] if rec.get("results") and rec.get("best_idx") is not None else {}
    reason_hints = {
        "key_driver_positive": "Улучшение маржи и итоговой прибыли",
        "key_driver_negative": "Неопределённость спроса и риск просадки объёма",
        "reason_why_this_scenario_wins": "Сценарий цены лидирует по скорингу objective с учётом risk penalties.",
    }
    if best_row_meta:
        if float(best_row_meta.get("demand_penalty", 0.0)) > 0:
            reason_hints["key_driver_negative"] = "Штраф за риск снижения объёма"
        elif float(best_row_meta.get("disagreement_penalty", 0.0)) > 0:
            reason_hints["key_driver_negative"] = "Расхождение прогнозов моделей (model disagreement)"

    biz_rec = build_business_recommendation(
        current_price=float(base_ctx.get("price")),
        recommended_price=float(rec.get("best_price")),
        current_profit=float(current_sim.get("adjusted_profit", 0.0)),
        recommended_profit=float(optimal_sim.get("adjusted_profit", 0.0)),
        confidence=confidence,
        elasticity=float(fixed_log_price_coef),
        history_days=history_days,
        current_revenue=float(current_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in current_sim else None,
        recommended_revenue=float(optimal_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else None,
        current_volume=float(current_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in current_sim else None,
        recommended_volume=float(optimal_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else None,
        data_quality=data_quality,
        base_ctx=base_ctx,
        reason_hints=reason_hints,
    )
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        daily_base.to_excel(writer, sheet_name="history", index=False)
        current_sim["daily"].to_excel(writer, sheet_name="baseline", index=False)
        optimal_sim["daily"].to_excel(writer, sheet_name="optimal", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame([biz_rec]).to_excel(writer, sheet_name="recommendation", index=False)
    excel_buffer.seek(0)
    return {
        "daily": daily_base,
        "recommendation": rec,
        "forecast_current": current_sim["daily"],
        "forecast_optimal": optimal_sim["daily"],
        "profit_curve": pd.DataFrame([{"price": x["price"], "adjusted_profit": x["adjusted_profit"]} for x in rec["results"]]),
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "elasticity_map": shrunk_random_effects,
        "current_price": float(base_ctx.get("price")),
        "best_price": float(rec.get("best_price")),
        "current_profit": float(current_sim.get("adjusted_profit", current_sim.get("total_profit", 0.0))),
        "best_profit": float(optimal_sim.get("adjusted_profit", optimal_sim.get("total_profit", 0.0))),
        "current_revenue": float(current_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in current_sim else 0.0,
        "best_revenue": float(optimal_sim["daily"].get("total_revenue", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else 0.0,
        "current_volume": float(current_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in current_sim else 0.0,
        "best_volume": float(optimal_sim["daily"].get("pred_sales", pd.Series(dtype=float)).sum()) if "daily" in optimal_sim else 0.0,
        "profit_lift_pct": ((optimal_sim["adjusted_profit"] - current_sim["adjusted_profit"]) / max(current_sim["adjusted_profit"], 1) * 100) if current_sim["adjusted_profit"] > 0 else 0.0,
        "data_quality": data_quality,
        "excel_buffer": excel_buffer,
        "flag": decision_flag(latest_row, rec),
        "business_recommendation": biz_rec,
        "objective_weights": obj_w,
        "feature_diagnostics": feature_diagnostics,
        "_trained_bundle": {
            "direct_models": direct_models,
            "baseline_models": baseline_models,
            "feature_spec": feature_spec,
            "feature_stats": feature_stats,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "elasticity_map": shrunk_random_effects,
            "pooled_elasticity": fixed_log_price_coef,
            "w_direct": w_direct,
            "confidence": confidence,
            "data_quality": data_quality,
            "objective_weights": obj_w,
            "feature_diagnostics": feature_diagnostics,
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
    scenario: Optional[Dict[str, Any]] = None,
    include_sensitivity: bool = True,
) -> Dict[str, Any]:
    base_history = trained_bundle["daily_base"].copy()
    base_ctx = dict(trained_bundle["base_ctx"])
    latest_row = dict(trained_bundle["latest_row"])
    scenario = dict(scenario or {})

    scenario_factors = dict(scenario.get("factors", {}))
    scenario_horizon = int(scenario.get("horizon_days", horizon_days if horizon_days is not None else len(trained_bundle.get("future_dates", [])) or CONFIG["HORIZON_DAYS_DEFAULT"]))
    scenario_name = str(scenario.get("name", "manual_scenario"))
    scenario_mode = str(scenario.get("mode", "manual"))
    scenario_warnings: List[str] = []

    for c in ["freight_value"]:
        if c in base_history.columns:
            base_history[c] = pd.to_numeric(base_history[c], errors="coerce").fillna(0.0) * float(freight_multiplier)
    if "discount" in base_history.columns:
        base_history["discount"] = pd.to_numeric(base_history["discount"], errors="coerce").fillna(0.0) * float(discount_multiplier)
    if "cost" in base_history.columns:
        base_history["cost"] = pd.to_numeric(base_history["cost"], errors="coerce").fillna(0.0) * float(cost_multiplier)
    for c in ["sales", "sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28", "sales_ma7", "sales_ma28", "sales_ma90"]:
        if c in base_history.columns:
            base_history[c] = pd.to_numeric(base_history[c], errors="coerce").fillna(0.0) * float(demand_multiplier)

    base_ctx["price"] = sanitize_non_negative(scenario_factors.get("price", manual_price), fallback=float(base_ctx.get("price", 0.0)))
    base_ctx["discount"] = sanitize_discount(scenario_factors.get("discount", float(base_ctx.get("discount", 0.0)) * float(discount_multiplier)))
    base_ctx["promotion"] = sanitize_non_negative(scenario_factors.get("promotion", base_ctx.get("promotion", 0.0)))
    base_ctx["stock"] = sanitize_non_negative(scenario_factors.get("stock", base_ctx.get("stock", 0.0)))
    base_ctx["rating"] = float(np.clip(float(scenario_factors.get("rating", base_ctx.get("rating", base_ctx.get("review_score", 4.5)))), 0.0, 5.0))
    base_ctx["reviews_count"] = sanitize_non_negative(scenario_factors.get("reviews_count", base_ctx.get("reviews_count", 0.0)))
    raw_cost = scenario_factors.get("cost", float(base_ctx.get("cost", base_ctx["price"] * CONFIG["COST_PROXY_RATIO"])) * float(cost_multiplier))
    base_ctx["cost"] = _resolve_unit_cost(raw_cost, base_ctx["price"])
    if sanitize_non_negative(raw_cost, fallback=0.0) > max(0.01, float(base_ctx["price"])) * 3.0:
        scenario_warnings.append("cost_unit_normalized")
    if base_ctx["discount"] >= 0.95:
        scenario_warnings.append("discount_capped")
    if "freight_value" in base_ctx:
        base_ctx["freight_value"] = sanitize_non_negative(scenario_factors.get("freight_value", float(base_ctx["freight_value"]) * float(freight_multiplier)))
    for c in ["sales", "sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28", "sales_ma7", "sales_ma28", "sales_ma90"]:
        if c in base_ctx:
            base_ctx[c] = float(base_ctx[c]) * float(demand_multiplier)

    latest_row.update(base_ctx)
    latest_row["requested_price"] = float(base_ctx["price"])
    latest_row["cost"] = float(base_ctx.get("cost", base_ctx["price"] * CONFIG["COST_PROXY_RATIO"]))
    latest_row["discount"] = float(base_ctx.get("discount", 0.0))
    latest_row["promotion"] = float(base_ctx.get("promotion", 0.0))
    latest_row["stock"] = float(base_ctx.get("stock", 0.0))
    latest_row["reviews_count"] = float(base_ctx.get("reviews_count", 0.0))
    latest_row["review_score"] = float(base_ctx.get("rating", base_ctx.get("review_score", 4.5)))

    future_dates = trained_bundle["future_dates"]
    if scenario_horizon is not None:
        future_dates = forecast_future_dates(pd.Timestamp(base_history["date"].max()), n_days=int(scenario_horizon))

    sim = simulate_horizon_profit(
        latest_row,
        float(base_ctx["price"]),
        future_dates,
        trained_bundle["direct_models"],
        trained_bundle["baseline_models"],
        base_history,
        base_ctx,
        trained_bundle["elasticity_map"],
        trained_bundle["pooled_elasticity"],
        trained_bundle["w_direct"],
        feature_spec=trained_bundle.get("feature_spec"),
        feature_stats=trained_bundle.get("feature_stats"),
    )
    daily = sim["daily"].copy()
    daily["cost"] = sanitize_non_negative(base_ctx.get("cost", 0.0))
    daily["discount"] = sanitize_discount(base_ctx.get("discount", 0.0))
    daily, econ_checks = compute_daily_unit_economics(
        daily,
        unit_price_col="price",
        unit_cost_col="cost",
        quantity_col="pred_sales",
        discount_col="discount",
        stock_cap=sanitize_non_negative(stock_cap),
    )
    daily["pred_sales"] = daily["pred_quantity"]
    daily["effective_price"] = daily["effective_unit_price"]
    daily["pred_demand"] = daily["pred_quantity"]
    stock_limit = sanitize_non_negative(stock_cap if stock_cap else base_ctx.get("stock", 0.0))
    if stock_limit > 0:
        daily["actual_sales"] = np.minimum(daily["pred_demand"], stock_limit)
    else:
        daily["actual_sales"] = daily["pred_demand"]
    daily["available_sales"] = daily["actual_sales"]
    daily["lost_sales"] = np.maximum(daily["pred_demand"] - daily["actual_sales"], 0.0)
    daily["pred_sales"] = daily["actual_sales"]
    daily["total_revenue"] = daily["effective_unit_price"] * daily["actual_sales"]
    unit_var_cost = daily["unit_variable_cost"] if "unit_variable_cost" in daily.columns else daily["unit_cost"]
    daily["total_cost"] = unit_var_cost * daily["actual_sales"]
    daily["profit"] = daily["total_revenue"] - daily["total_cost"]
    daily["margin"] = np.where(daily["total_revenue"] > 0, daily["profit"] / daily["total_revenue"], 0.0)
    demand_total = float(daily["pred_quantity"].sum()) if "pred_quantity" in daily.columns else 0.0
    actual_sales_total = float(daily["actual_sales"].sum()) if "actual_sales" in daily.columns else demand_total
    lost_sales_total = float(daily["lost_sales"].sum()) if "lost_sales" in daily.columns else 0.0
    profit_total = float(daily["profit"].sum()) if "profit" in daily.columns else 0.0
    revenue_total = float(daily["total_revenue"].sum()) if "total_revenue" in daily.columns else 0.0
    confidence = float(trained_bundle.get("confidence", 0.6))
    result = {
        "daily": daily,
        "demand_total": demand_total,
        "actual_sales_total": actual_sales_total,
        "lost_sales_total": lost_sales_total,
        "profit_total": profit_total,
        "revenue_total": revenue_total,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "sanity_warnings": econ_checks.get("sanity_warnings", []) + scenario_warnings + sim.get("sanity_warnings", []),
        "scenario_meta": {
            "name": scenario_name,
            "mode": scenario_mode,
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "model_version": str(trained_bundle.get("model_version", "v1")),
            "factors": scenario_factors,
        },
    }
    if include_sensitivity:
        sensitivity = {}
        for factor, step in {"price": 0.01, "discount": 0.02, "promotion": 0.1, "freight_value": 0.05, "stock": 0.1, "rating": 0.1}.items():
            if factor not in {"price", "discount", "promotion", "freight_value", "stock", "rating"}:
                continue
            local = dict(scenario_factors)
            base_val = float(base_ctx.get(factor, 0.0))
            bump = base_val * (1.0 + step) if abs(base_val) > 1e-9 else step
            local[factor] = bump
            plus = run_what_if_projection(
                trained_bundle,
                manual_price=float(local.get("price", base_ctx["price"])),
                horizon_days=scenario_horizon,
                stock_cap=stock_cap,
                scenario={"name": f"{scenario_name}_{factor}_plus", "mode": scenario_mode, "horizon_days": scenario_horizon, "factors": local},
                include_sensitivity=False,
            )
            sensitivity[factor] = {
                "delta_demand": float(plus.get("demand_total", 0.0) - demand_total),
                "delta_profit": float(plus.get("profit_total", 0.0) - profit_total),
            }
        result["local_sensitivity"] = sensitivity
    return result


def assess_data_quality(
    history_days: int,
    n_points: int,
    missing_share: float,
    holdout_wape: float,
) -> Dict[str, Any]:
    issues: List[str] = []
    if history_days < 60:
        issues.append("История данных короткая (меньше 60 дней).")
    if n_points < 45:
        issues.append("Слишком мало наблюдений для устойчивого прогноза.")
    if missing_share > 0.2:
        issues.append("Высокая доля пропусков в данных.")
    if np.isfinite(holdout_wape) and holdout_wape > 35:
        issues.append("Ошибка прогноза выше нормы, результат может быть неточным.")

    level = "good"
    confidence_cap = 1.0
    if history_days < 30 or n_points < 20:
        level = "unavailable"
        confidence_cap = 0.2
    elif issues:
        if len(issues) >= 3 or (np.isfinite(holdout_wape) and holdout_wape > 50):
            level = "poor"
            confidence_cap = 0.45
        else:
            level = "medium"
            confidence_cap = 0.75

    level_to_text = {
        "good": "Можно использовать рекомендацию",
        "medium": "Рекомендация предварительная",
        "poor": "Качество данных низкое",
        "unavailable": "Недостаточно данных для надёжной рекомендации",
    }
    return {
        "level": level,
        "label": level_to_text[level],
        "issues": issues,
        "confidence_cap": float(confidence_cap),
        "can_recommend": level not in {"unavailable"},
    }


def generate_explanation(
    result_bundle: Dict[str, Any],
    data_quality: Optional[Dict[str, Any]] = None,
    scenario_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    current_price = float(result_bundle.get("current_price", 0.0))
    best_price = _resolve_recommended_price(result_bundle, current_price)
    current_profit = float(result_bundle.get("current_profit", 0.0))
    best_profit = float(result_bundle.get("best_profit", current_profit))
    current_sales = float(result_bundle.get("forecast_current", pd.DataFrame()).get("pred_sales", pd.Series(dtype=float)).sum()) if "forecast_current" in result_bundle else 0.0
    best_sales = float(result_bundle.get("forecast_optimal", pd.DataFrame()).get("pred_sales", pd.Series(dtype=float)).sum()) if "forecast_optimal" in result_bundle else 0.0

    price_change_pct = _safe_price_delta_pct(current_price, best_price)
    profit_change_pct = ((best_profit - current_profit) / max(abs(current_profit), 1e-9)) * 100.0 if np.isfinite(current_profit) else 0.0
    sales_change_pct = ((best_sales - current_sales) / max(abs(current_sales), 1e-9)) * 100.0 if np.isfinite(current_sales) else 0.0

    elasticity_values = list(result_bundle.get("elasticity_map", {}).values())
    mean_sensitivity = float(np.nanmean(elasticity_values)) if elasticity_values else float("nan")
    quality_level = (data_quality or {}).get("level", "good")

    pros: List[str] = []
    cons: List[str] = []
    if profit_change_pct >= 1:
        pros.append(f"Ожидается рост прибыли примерно на {profit_change_pct:+.1f}%.")
    elif profit_change_pct <= -1:
        cons.append(f"Прибыль может снизиться примерно на {profit_change_pct:.1f}%.")
    else:
        pros.append("Прибыль остаётся близкой к текущему уровню без резких изменений.")

    if sales_change_pct < -2:
        cons.append(f"Продажи могут снизиться примерно на {abs(sales_change_pct):.1f}%.")
    elif sales_change_pct > 2:
        pros.append(f"Продажи могут вырасти примерно на {sales_change_pct:.1f}%.")
    else:
        pros.append("Ожидаемые продажи остаются в стабильном диапазоне.")

    if np.isfinite(mean_sensitivity):
        if mean_sensitivity <= -1.2:
            cons.append("Спрос чувствителен к цене, поэтому лучше менять цену постепенно.")
        elif mean_sensitivity >= -0.6:
            pros.append("Спрос реагирует на цену умеренно, риск резкого проседания ниже.")
        else:
            pros.append("Спрос реагирует на цену умеренно, изменение выглядит контролируемым.")

    if quality_level in {"poor", "unavailable"}:
        cons.append("Данные ограничены, поэтому рекомендация носит предварительный характер.")
    elif quality_level == "medium":
        cons.append("Качество данных среднее: результат лучше подтвердить дополнительной проверкой.")

    if abs(price_change_pct) >= 8:
        cons.append("Шаг по цене заметный — стоит внедрять поэтапно и наблюдать за спросом.")
    else:
        pros.append("Изменение цены умеренное, его проще внедрить без резкого эффекта.")

    if scenario_metrics:
        scenario_delta_profit = float(scenario_metrics.get("delta_profit", 0.0))
        scenario_delta_sales = float(scenario_metrics.get("delta_sales", 0.0))
        if scenario_delta_profit < 0:
            cons.append("Сценарная проверка показывает риск снижения прибыли при стресс-факторах.")
        elif scenario_delta_profit > 0:
            pros.append("Сценарный анализ подтверждает потенциал роста прибыли.")
        if scenario_delta_sales < 0:
            cons.append("В стресс-сценариях возможна потеря части объёма продаж.")

    summary = f"Цена {price_change_pct:+.1f}% может изменить прибыль на {profit_change_pct:+.1f}% и объём на {sales_change_pct:+.1f}%."
    return {
        "summary": summary,
        "pros": pros,
        "cons": cons,
        "price_change_pct": float(price_change_pct),
        "profit_change_pct": float(profit_change_pct),
        "sales_change_pct": float(sales_change_pct),
        "mean_elasticity": float(mean_sensitivity) if np.isfinite(mean_sensitivity) else None,
    }
