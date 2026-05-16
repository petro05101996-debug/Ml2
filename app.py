from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import copy
import gc
import hashlib
import json
import logging
import subprocess
import warnings
from html import escape as html_escape
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import importlib.util

if importlib.util.find_spec("streamlit") is not None:
    import streamlit as st
else:
    class _StreamlitUnavailable:
        """Minimal import-time stub; running the UI still requires streamlit."""
        session_state = {}
        def __getattr__(self, name):
            def _missing(*args, **kwargs):
                raise RuntimeError("Streamlit is required to render the UI. Install requirements or use core modules.")
            return _missing
    st = _StreamlitUnavailable()
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
from scenario_engine_enhanced import run_enhanced_scenario
from catboost_full_factor_engine import (
    CATBOOST_FULL_FACTOR_MODE,
    predict_catboost_full_factor_projection,
    train_catboost_full_factor_bundle,
)
from v1_runtime_helpers import (
    build_backend_warning,
    compute_scenario_price_inputs,
    evaluate_net_price_support,
    VALID_ELASTICITY_MAX,
    VALID_ELASTICITY_MIN,
    get_model_backend_status,
    select_weekly_baseline_candidate,
)
from what_if import build_sensitivity_grid, run_scenario_set
from price_optimizer import analyze_price_optimization, build_price_optimizer_signature
from decision_candidate_engine import generate_decision_candidates, evaluate_decision_candidates
from decision_optimizer import rank_decision_candidates
from decision_passport import build_decision_passport
from decision_analysis_engine import DecisionAnalysisInput, analyze_decision
from decision_math import safe_pct_delta
from recommendation_auditor import audit_and_improve_recommendation
from recommendation_gate import resolve_recommendation_gate
from production_contract import resolve_data_quality_gate
from scenario_audit import build_scenario_reproducibility_id
from model_quality_gate import evaluate_model_quality_gate
from production_monitoring import build_run_metadata
from ui.theme import apply_theme
from ui.copy import (
    PAGE_OVERVIEW, PAGE_DECISION, PAGE_WHAT_IF, PAGE_PRICE,
    PAGE_COMPARE, PAGE_REPORT, PAGE_DIAGNOSTICS, PRODUCT_PROMISE,
    PRODUCT_SUBTITLE, WORKSPACE_PAGES, normalize_workspace_page,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ACTIONABLE_PRICE_OPT_STATUSES = {"price_increase_recommended", "price_decrease_recommended"}
PRODUCTION_STABLE_ROLLING_VERDICTS = {"stable", "moderately_stable"}
TEST_ONLY_ROLLING_VERDICTS = {"test_only_unstable", "unstable_test_only"}
EXPERIMENTAL_ROLLING_VERDICTS = {"experimental_unstable", "unstable_experimental_only"}

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

PLOT_ACCENT = "#6F70FF"
PLOT_MUTED = "#8FA3B8"
PLOT_SUCCESS = "#9DCC84"
PLOT_TEXT = "#F4F7FB"
PLOT_ACCENT_FILL = "rgba(111,112,255,0.12)"

SCENARIO_CALC_MODES = {
    "legacy_current": "Базовый режим: текущий план + простые сценарные множители",
    "enhanced_local_factors": "Расширенный сценарный режим: дневной пересчёт факторов",
    CATBOOST_FULL_FACTOR_MODE: "Факторная модель: прогноз спроса по факторам",
}
DEFAULT_SCENARIO_CALC_MODE = "enhanced_local_factors"
PRICE_GUARDRAIL_SAFE_CLIP = "safe_clip"
PRICE_GUARDRAIL_EXTRAPOLATE = "economic_extrapolation"
DEFAULT_PRICE_GUARDRAIL_MODE = PRICE_GUARDRAIL_SAFE_CLIP

CANONICAL_FIELD_UI = {
    "date": "Дата операции",
    "product_id": "SKU / товар",
    "category": "Категория",
    "price": "Цена продажи",
    "quantity": "Продажи, шт.",
    "revenue": "Выручка",
    "cost": "Себестоимость",
    "discount": "Скидка",
    "freight_value": "Логистика / доставка",
    "stock": "Остаток",
    "promotion": "Промо / акция",
    "rating": "Рейтинг",
    "reviews_count": "Количество отзывов",
    "region": "Регион",
    "channel": "Канал продаж",
    "segment": "Сегмент",
}

SCENARIO_SLOT_LABELS = {
    "Scenario A": "Вариант 1",
    "Scenario B": "Вариант 2",
    "Scenario C": "Вариант 3",
}


def normalize_price_guardrail_mode(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value in {PRICE_GUARDRAIL_SAFE_CLIP, "safe", "clip", "защитный"}:
        return PRICE_GUARDRAIL_SAFE_CLIP
    if value in {
        PRICE_GUARDRAIL_EXTRAPOLATE, "extrapolate", "extrapolation", "manual", "exact", "strict", "exact_manual", "строгий", "экстраполяция"
    }:
        return PRICE_GUARDRAIL_EXTRAPOLATE
    return DEFAULT_PRICE_GUARDRAIL_MODE


def scenario_mode_label(mode_code: str) -> str:
    mapping = {
        "legacy_current": "Базовый режим — текущий план + простые сценарные множители.",
        "enhanced_local_factors": "Расширенный сценарный режим — дневной пересчёт факторов.",
        CATBOOST_FULL_FACTOR_MODE: "Факторная модель — прогноз спроса пересчитывается по доступным факторам.",
    }
    return mapping.get(str(mode_code), str(mode_code))



def data_quality_ui_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "ok": "Достаточно",
        "good": "Хорошее",
        "warning": "Есть ограничения",
        "advisory": "Есть ограничения",
        "diagnostic_only": "Только диагностика",
        "blocked": "Недостаточно для рекомендации",
        "unknown": "Проверьте диагностику",
        "": "Проверьте диагностику",
    }
    return mapping.get(raw, str(value or "Проверьте диагностику"))

def scenario_contract_label(contract_code: str) -> str:
    mapping = {
        "legacy_baseline+scenario_recompute": "Базовый режим: текущий план + простые сценарные множители.",
        "legacy_baseline+enhanced_local_factor_layer": "Расширенный сценарный режим: локальный слой факторов поверх текущего плана.",
        "daily_catboost_full_factors+model_reprediction": "Факторная модель: спрос пересчитывается по изменённым факторам.",
    }
    return mapping.get(str(contract_code), str(contract_code))


def effect_source_label(effect_source: str) -> str:
    mapping = {
        "scenario_engine_recompute_from_baseline": "Сценарный пересчёт от текущего плана",
        "baseline_daily_x_local_factor_layer": "Локальный слой факторов поверх текущего плана",
        "catboost_full_factor_reprediction": "Факторная модель пересчитала спрос по изменённым факторам",
        "catboost_boundary_plus_elasticity_extrapolation": "Факторная модель до безопасной границы + контролируемая ценовая оценка за пределами истории",
    }
    return mapping.get(str(effect_source), str(effect_source))


def safe_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def _safe_metric_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _html_safe(value: Any) -> str:
    return html_escape(str(value if value is not None else "—"))




def _build_model_quality_gate_dict(metrics: Dict[str, Any], rolling: Optional[Dict[str, Any]] = None, stockout_share: Any = None) -> Dict[str, Any]:
    src = dict(metrics or {})
    if "naive_wape" not in src and "best_naive_wape" in src:
        src["naive_wape"] = src.get("best_naive_wape")
    if "improvement_vs_naive_pct" not in src and "naive_improvement_pct" in src:
        src["improvement_vs_naive_pct"] = src.get("naive_improvement_pct")
    if rolling:
        src["rolling_retrain_backtest"] = rolling
        wapes = []
        for row in list((rolling or {}).get("windows", []) or []):
            try:
                val = float((row or {}).get("wape"))
                if np.isfinite(val):
                    wapes.append(val)
            except Exception:
                pass
        if wapes:
            src.setdefault("rolling_wape_max", max(wapes))
    if stockout_share is not None:
        src["stockout_share"] = stockout_share
    return evaluate_model_quality_gate(src).to_dict()


def _attach_run_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result or {})
    bundle = out.get("_trained_bundle", {}) or {}
    history = out.get("history_daily")
    if isinstance(history, pd.DataFrame):
        dataset_signature: Any = {
            "rows": int(len(history)),
            "columns": [str(c) for c in history.columns],
            "date_min": str(pd.to_datetime(history.get("date"), errors="coerce").min()) if "date" in history.columns else "",
            "date_max": str(pd.to_datetime(history.get("date"), errors="coerce").max()) if "date" in history.columns else "",
            "sales_sum": float(pd.to_numeric(history.get("sales", pd.Series(dtype="float64")), errors="coerce").fillna(0.0).sum()) if "sales" in history.columns else 0.0,
        }
    else:
        dataset_signature = str(type(history))
    meta = build_run_metadata(
        dataset_signature=dataset_signature,
        selected_sku=str((bundle.get("base_ctx", {}) or {}).get("product_id", out.get("selected_sku", "unknown"))),
        scenario_mode=str(out.get("analysis_scenario_calc_mode", bundle.get("analysis_scenario_calc_mode", "unknown"))),
        decision_status=str(out.get("decision_status", out.get("recommendation_gate", "analysis_created"))),
        model_quality=out.get("model_quality_gate", bundle.get("model_quality_gate", {})),
        data_quality=out.get("data_contract", bundle.get("data_contract", {})),
        warnings=list(out.get("warnings", []) or []),
        blockers=list(out.get("blockers", []) or []),
    ).to_dict()
    out["run_metadata"] = meta
    if isinstance(bundle, dict):
        bundle = dict(bundle)
        bundle["run_metadata"] = meta
        out["_trained_bundle"] = bundle
    return out

def resolve_scenario_calc_mode(mode: Optional[str]) -> str:
    if mode is None:
        return DEFAULT_SCENARIO_CALC_MODE
    mode_val = str(mode)
    return mode_val if mode_val in SCENARIO_CALC_MODES else DEFAULT_SCENARIO_CALC_MODE


def build_recommended_mode_status(mode: str, catboost_bundle: Optional[Dict[str, Any]] = None, fallback_reason: str = "") -> Dict[str, Any]:
    cb = dict(catboost_bundle or {})
    metrics = dict(cb.get("holdout_metrics", {}) or {})
    wape = safe_float_or_nan(metrics.get("wape", np.nan))
    naive_improvement = safe_float_or_nan(metrics.get("naive_improvement_pct", np.nan))
    rolling_verdict = str((cb.get("rolling_retrain_backtest", {}) or metrics.get("rolling_retrain_backtest", {}) or {}).get("verdict", ""))
    cb_enabled = bool(cb.get("enabled", False))
    if cb_enabled:
        recommended_mode = CATBOOST_FULL_FACTOR_MODE
        if not np.isfinite(wape):
            recommended_mode = "enhanced_local_factors"
            status = "quality_unknown"
            reason = "CatBoost доступен, но WAPE неизвестен; production-рекомендация заблокирована, используйте Enhanced/diagnostic what-if."
        elif wape <= 15.0 and np.isfinite(naive_improvement) and naive_improvement >= 10.0 and rolling_verdict in PRODUCTION_STABLE_ROLLING_VERDICTS:
            status = "recommended"
            reason = "CatBoost проходит production gate: WAPE ≤15%, улучшает naive baseline ≥10%, rolling backtest stable/moderately stable."
        elif wape <= 30.0:
            recommended_mode = "enhanced_local_factors"
            status = "test_recommended"
            reason = "CatBoost доступен, но качество допускает только controlled test, не автоматическую рекомендацию."
        elif wape <= 40.0:
            recommended_mode = "enhanced_local_factors"
            status = "experimental_only"
            reason = "CatBoost WAPE 30–40%; это только экспериментальная гипотеза."
        else:
            recommended_mode = "enhanced_local_factors"
            status = "not_recommended"
            reason = "CatBoost WAPE >40%; использовать как рекомендацию нельзя."
        if rolling_verdict in TEST_ONLY_ROLLING_VERDICTS and status == "recommended":
            recommended_mode = "enhanced_local_factors"
            status = "test_recommended"
            reason = "Rolling CatBoost backtest нестабилен; допустим только controlled test."
        elif (rolling_verdict in EXPERIMENTAL_ROLLING_VERDICTS or rolling_verdict.startswith("experimental")) and status in {"recommended", "test_recommended"}:
            recommended_mode = "enhanced_local_factors"
            status = "experimental_only"
            reason = "Rolling CatBoost backtest нестабилен; CatBoost не является production-рекомендацией."
        return {
            "recommended_mode": recommended_mode,
            "active_mode": str(mode),
            "status": status,
            "reason": reason,
            "holdout_wape": wape,
            "naive_improvement_pct": naive_improvement,
            "rolling_retrain_verdict": rolling_verdict,
        }
    return {
        "recommended_mode": "enhanced_local_factors",
        "active_mode": str(mode),
        "status": "catboost_unavailable",
        "reason": fallback_reason or f"CatBoost недоступен: {cb.get('reason', 'unknown')}. Используется Enhanced.",
        "holdout_wape": wape,
        "naive_improvement_pct": naive_improvement,
        "rolling_retrain_verdict": rolling_verdict,
    }


def attach_scenario_reproducibility(result: Dict[str, Any], trained_bundle: Dict[str, Any], params: Dict[str, Any], mode: str, price_guardrail_mode: str) -> Dict[str, Any]:
    result["scenario_reproducibility"] = build_scenario_reproducibility_id(
        trained_bundle=trained_bundle,
        scenario_params=params,
        mode=mode,
        guardrail_mode=price_guardrail_mode,
        model_version=ARTIFACT_SCHEMA_VERSION,
        code_signature=f"app_generation:{APP_GENERATION}",
        feature_schema_version="universal_csv_what_if_v1",
        config={"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "app_generation": APP_GENERATION},
    )
    result["scenario_run_id"] = result["scenario_reproducibility"]["scenario_run_id"]
    return result


def normalize_factor_report_for_ui(report: pd.DataFrame) -> pd.DataFrame:
    if report is None or len(report) == 0:
        return pd.DataFrame(columns=["feature", "used_in_active_model", "reason"])
    out = report.copy()
    if "feature" not in out.columns and "factor" in out.columns:
        out["feature"] = out["factor"]
    if "used_in_active_model" not in out.columns:
        if "usable_in_model" in out.columns:
            out["used_in_active_model"] = out["usable_in_model"].astype(bool)
        else:
            out["used_in_active_model"] = False
    if "reason" not in out.columns:
        if "reason_if_excluded" in out.columns:
            out["reason"] = out["reason_if_excluded"]
        else:
            out["reason"] = ""
    return out


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
    freight = pd.to_numeric(df.get("freight_value", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0)
    date_series = pd.to_datetime(df.get("date"), errors="coerce")

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
    promo_weeks = 0
    if len(df):
        wk = pd.DataFrame({"date": date_series, "promotion": promotion}).dropna(subset=["date"])
        if len(wk):
            wk["week"] = wk["date"].dt.to_period("W").astype(str)
            promo_weeks = int(wk.groupby("week")["promotion"].mean().gt(0.0).sum())
    promo_variability = float(promotion.std(ddof=0)) if len(promotion) else 0.0
    freight_changes = int(freight.diff().abs().gt(1e-9).sum()) if len(freight) else 0
    freight_mean = float(np.abs(freight.mean())) if len(freight) else 0.0
    freight_variation = float(freight.std(ddof=0) / max(freight_mean, 1e-9)) if len(freight) else 0.0
    price_cv = float(price_non_null.std(ddof=0) / max(abs(float(price_non_null.mean())), 1e-9)) if len(price_non_null) else 0.0
    price_stability = float(np.clip(1.0 - price_cv, 0.0, 1.0))

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
        "promo_weeks": promo_weeks,
        "promo_variability": promo_variability,
        "freight_changes": freight_changes,
        "freight_variation": freight_variation,
        "price_stability": price_stability,
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


def resolve_path_contracts(selected_forecaster: str, production_selected_candidate: str, scenario_calc_mode: str) -> Dict[str, str]:
    if str(scenario_calc_mode) == CATBOOST_FULL_FACTOR_MODE:
        return {
            "baseline_forecast_path": "daily_catboost_full_factor_baseline",
            "scenario_calculation_path": "catboost_full_factor_reprediction",
            "learned_uplift_path": "inactive_not_used_in_this_mode",
            "final_user_visible_path": "daily_catboost_full_factor_baseline + catboost_full_factor_reprediction",
            "production_selected_candidate": "catboost_full_factors",
        }
    baseline_forecast_path = (
        "weekly_ml_baseline"
        if str(selected_forecaster) == "weekly_model"
        else ("naive_lag1w_baseline" if str(selected_forecaster) == "naive_lag1w" else ("naive_ma4w_baseline" if str(selected_forecaster) == "naive_ma4w" else "deterministic_fallback_baseline"))
    )
    scenario_calculation_path = (
        "enhanced_local_factor_layer"
        if str(scenario_calc_mode) == "enhanced_local_factors"
        else "scenario_recompute"
    )
    learned_uplift_path = "inactive_production_diagnostic_only"
    final_user_visible_path = f"{baseline_forecast_path} + {scenario_calculation_path}"
    return {
        "baseline_forecast_path": baseline_forecast_path,
        "scenario_calculation_path": scenario_calculation_path,
        "learned_uplift_path": learned_uplift_path,
        "final_user_visible_path": final_user_visible_path,
        "production_selected_candidate": str(production_selected_candidate or "legacy_baseline"),
    }


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
    scenario_calc_mode: str = DEFAULT_SCENARIO_CALC_MODE,
    horizon_days: int = CONFIG["HORIZON_DAYS_DEFAULT"],
):
    analysis_scenario_calc_mode = resolve_scenario_calc_mode(scenario_calc_mode)
    if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE:
        catboost_result = _run_catboost_full_factor_analysis_universal(
            normalized_txn=normalized_txn,
            target_category=target_category,
            target_sku=target_sku,
            region=region,
            channel=channel,
            segment=segment,
            horizon_days=int(horizon_days),
        )
        cb_bundle = (catboost_result.get("_trained_bundle", {}) or {}).get("catboost_full_factor_bundle", {})
        if catboost_result.get("blocking_error_code") == "catboost_full_factors_unavailable":
            fallback_result = _run_existing_legacy_enhanced_analysis_universal(
                normalized_txn=normalized_txn,
                target_category=target_category,
                target_sku=target_sku,
                region=region,
                channel=channel,
                segment=segment,
                scenario_calc_mode="enhanced_local_factors",
                horizon_days=int(horizon_days),
            )
            fallback_result["recommended_mode_status"] = build_recommended_mode_status(
                "enhanced_local_factors",
                cb_bundle,
                fallback_reason=str(catboost_result.get("blocking_error_message", "CatBoost unavailable; fallback to Enhanced.")),
            )
            fallback_result["catboost_full_factor_attempt"] = {
                "enabled": False,
                "reason": cb_bundle.get("reason", catboost_result.get("blocking_error_message", "unknown")),
                "warnings": catboost_result.get("warnings", []),
            }
            return refresh_excel_export(fallback_result)
        catboost_result["recommended_mode_status"] = build_recommended_mode_status(CATBOOST_FULL_FACTOR_MODE, cb_bundle)
        return refresh_excel_export(catboost_result)
    return _run_existing_legacy_enhanced_analysis_universal(
        normalized_txn=normalized_txn,
        target_category=target_category,
        target_sku=target_sku,
        region=region,
        channel=channel,
        segment=segment,
        scenario_calc_mode=analysis_scenario_calc_mode,
        horizon_days=int(horizon_days),
    )


def _run_catboost_full_factor_analysis_universal(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    region: Optional[str] = None,
    channel: Optional[str] = None,
    segment: Optional[str] = None,
    horizon_days: int = CONFIG["HORIZON_DAYS_DEFAULT"],
) -> Dict[str, Any]:
    analysis_scenario_calc_mode = CATBOOST_FULL_FACTOR_MODE
    txn = normalized_txn.copy()
    dq_contract = getattr(txn, "attrs", {}).get("data_quality_contract", {}) or {}
    data_quality_gate = resolve_data_quality_gate(dq_contract)
    if len(txn) == 0:
        raise ValueError("Пустой датасет после нормализации.")
    daily_base = build_daily_from_transactions(
        txn,
        target_sku,
        target_category=target_category,
        region=region,
        channel=channel,
        segment=segment,
        include_extra_factors=True,
    )
    daily_base = robust_clean_dirty_data(daily_base)
    daily_base = build_feature_matrix(daily_base).dropna(subset=["sales", "price"]).reset_index(drop=True)
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()), n_days=int(horizon_days))
    base_ctx = current_price_context(daily_base)
    base_ctx["category"] = target_category
    base_ctx["product_id"] = target_sku
    latest_row = dict(base_ctx)
    latest_row["requested_price"] = float(base_ctx.get("price", safe_median(daily_base["price"], 1.0)))
    catboost_full_factor_bundle = train_catboost_full_factor_bundle(
        daily_base=daily_base,
        future_dates=future_dates,
    )
    warnings = list(catboost_full_factor_bundle.get("warnings", []))
    if not catboost_full_factor_bundle.get("enabled", False):
        return refresh_excel_export(
            {
                "history_daily": daily_base,
                "quality_report": {"holdout_metrics": {}},
                "feature_usage_report": pd.DataFrame(),
                "feature_report": pd.DataFrame(),
                "neutral_baseline_forecast": pd.DataFrame(),
                "as_is_forecast": pd.DataFrame(),
                "scenario_forecast": None,
                "delta_vs_as_is": {},
                "delta_vs_baseline": {},
                "warnings": warnings,
                "analysis_valid": False,
                "small_mode_info": detect_small_mode_info(daily_base),
                "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
                "analysis_scenario_calc_mode_label": scenario_mode_label(analysis_scenario_calc_mode),
                "blocking_error": True,
                "blocking_error_code": "catboost_full_factors_unavailable",
                "blocking_error_message": (
                    "Выбран CatBoost full factors, но модель не может быть обучена: "
                    f"{catboost_full_factor_bundle.get('reason', 'unknown')}."
                ),
                "data_quality_contract": dq_contract,
                "data_quality_gate": data_quality_gate,
                "data_contract": dq_contract.get("data_contract", {}),
                "target_semantics": dq_contract.get("target_semantics", {}),
                "blockers": list(data_quality_gate.get("blockers", []) or []),
                "catboost_full_factor_report": catboost_full_factor_bundle.get("feature_report", pd.DataFrame()),
                "catboost_full_factor_importances": pd.DataFrame(catboost_full_factor_bundle.get("feature_importances", [])),
                "_trained_bundle": {
                    "daily_base": daily_base,
                    "base_ctx": base_ctx,
                    "latest_row": latest_row,
                    "future_dates": future_dates,
                    "analysis_scenario_calc_mode": CATBOOST_FULL_FACTOR_MODE,
                    "catboost_full_factor_bundle": catboost_full_factor_bundle,
                    "data_quality_contract": dq_contract,
                    "data_quality_gate": data_quality_gate,
                    "data_contract": dq_contract.get("data_contract", {}),
                    "target_semantics": dq_contract.get("target_semantics", {}),
                },
            }
        )
    provisional_bundle = {
        "daily_base": daily_base,
        "base_ctx": base_ctx,
        "latest_row": latest_row,
        "future_dates": future_dates,
        "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
        "catboost_full_factor_bundle": catboost_full_factor_bundle,
    }
    neutral_overrides = {
        "discount": 0.0,
        "promotion": 0.0,
        "freight_value": float(safe_median(daily_base.get("freight_value", pd.Series([0.0])), 0.0)),
    }
    as_is_sim = predict_catboost_full_factor_projection(
        provisional_bundle,
        manual_price=float(base_ctx.get("price")),
        horizon_days=int(horizon_days),
        overrides={
            "discount": float(base_ctx.get("discount", 0.0)),
            "promotion": float(base_ctx.get("promotion", 0.0)),
            "freight_value": float(base_ctx.get("freight_value", 0.0)),
        },
    )
    baseline_price = float(safe_median(daily_base["price"], float(base_ctx.get("price", 1.0))))
    baseline_sim = predict_catboost_full_factor_projection(
        provisional_bundle,
        manual_price=baseline_price,
        horizon_days=int(horizon_days),
        overrides=neutral_overrides,
    )
    run_summary = {
        "config": {
            "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
            "analysis_scenario_calc_mode_label": scenario_mode_label(analysis_scenario_calc_mode),
            "model_backend": "catboost",
            "backend_reason": "catboost_full_factor_mode",
            "baseline_forecast_path": "daily_catboost_full_factor_baseline",
            "scenario_calculation_path": "catboost_full_factor_reprediction",
            "final_user_visible_path": "daily_catboost_full_factor_baseline + catboost_full_factor_reprediction",
            "final_active_path": "daily_catboost_full_factors+model_reprediction",
        },
        "catboost_full_factor": {
            "enabled": True,
            "feature_count": int(len(catboost_full_factor_bundle.get("feature_cols", []))),
            "cat_feature_count": int(len(catboost_full_factor_bundle.get("cat_feature_names", []))),
            "holdout_metrics": dict(catboost_full_factor_bundle.get("holdout_metrics", {})),
            "top_features": list(catboost_full_factor_bundle.get("feature_importances", []))[:20],
        },
        "metrics_summary": {"holdout_flat": dict(catboost_full_factor_bundle.get("holdout_metrics", {}))},
        "scenario_output_summary": {
            "model_backend": "catboost",
            "backend_reason": "catboost_full_factor_mode",
            "active_path_contract": "daily_catboost_full_factors+model_reprediction",
        },
    }
    model_quality_gate = _build_model_quality_gate_dict(
        dict(catboost_full_factor_bundle.get("holdout_metrics", {}) or {}),
        rolling=dict(catboost_full_factor_bundle.get("rolling_retrain_backtest", {}) or {}),
    )
    result = {
        "history_daily": daily_base,
        "quality_report": {"holdout_metrics": dict(catboost_full_factor_bundle.get("holdout_metrics", {}))},
        "model_quality_gate": model_quality_gate,
        "feature_usage_report": catboost_full_factor_bundle.get("feature_report", pd.DataFrame()),
        "feature_report": catboost_full_factor_bundle.get("feature_report", pd.DataFrame()),
        "neutral_baseline_forecast": baseline_sim["daily"],
        "as_is_forecast": as_is_sim["daily"],
        "scenario_forecast": None,
        "delta_vs_as_is": {},
        "delta_vs_baseline": {
            "demand_total": float(as_is_sim["daily"]["actual_sales"].sum() - baseline_sim["daily"]["actual_sales"].sum()),
            "revenue_total": float(as_is_sim["daily"]["revenue"].sum() - baseline_sim["daily"]["revenue"].sum()),
            "profit_total": float(as_is_sim["daily"]["profit"].sum() - baseline_sim["daily"]["profit"].sum()),
        },
        "warnings": warnings,
        "analysis_valid": True,
        "small_mode_info": detect_small_mode_info(daily_base),
        "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
        "analysis_scenario_calc_mode_label": scenario_mode_label(analysis_scenario_calc_mode),
        "holdout_metrics": pd.DataFrame([dict(catboost_full_factor_bundle.get("holdout_metrics", {}))]),
        "elasticity_map": {},
        "current_price": float(base_ctx.get("price")),
        "scenario_price": None,
        "current_profit": float(as_is_sim.get("profit_total", 0.0)),
        "data_quality_contract": dq_contract,
        "data_quality_gate": data_quality_gate,
        "data_contract": dq_contract.get("data_contract", {}),
        "target_semantics": dq_contract.get("target_semantics", {}),
        "blockers": list(data_quality_gate.get("blockers", []) or []),
        "analysis_run_summary_json": json.dumps(run_summary, ensure_ascii=False, indent=2).encode("utf-8"),
        "holdout_predictions_csv": catboost_full_factor_bundle.get("holdout_predictions", pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        "analysis_baseline_vs_as_is_csv": pd.DataFrame(
            {
                "date": pd.to_datetime(as_is_sim["daily"]["date"]).dt.strftime("%Y-%m-%d"),
                "as_is_sales": pd.to_numeric(as_is_sim["daily"]["actual_sales"], errors="coerce").fillna(0.0),
                "baseline_sales": pd.to_numeric(baseline_sim["daily"]["actual_sales"], errors="coerce").fillna(0.0),
            }
        ).to_csv(index=False).encode("utf-8"),
        "uplift_debug_report_csv": pd.DataFrame([{"status": "not_used_in_catboost_full_factor_mode"}]).to_csv(index=False).encode("utf-8"),
        "uplift_holdout_trace_csv": pd.DataFrame([{"status": "not_used_in_catboost_full_factor_mode"}]).to_csv(index=False).encode("utf-8"),
        "manual_scenario_summary_json": b"{}",
        "manual_scenario_daily_csv": b"",
        "feature_report_csv": catboost_full_factor_bundle.get("feature_report", pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        "catboost_full_factor_report": catboost_full_factor_bundle.get("feature_report", pd.DataFrame()),
        "catboost_full_factor_importances": pd.DataFrame(catboost_full_factor_bundle.get("feature_importances", [])),
        "_trained_bundle": {
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
            "cost_input_available": bool(getattr(txn, "attrs", {}).get("cost_input_available", bool("cost" in txn.columns))),
            "cost_is_proxy": bool(getattr(txn, "attrs", {}).get("cost_is_proxy", False)),
            "cost_source": str(getattr(txn, "attrs", {}).get("cost_source", "provided" if "cost" in txn.columns else "missing")),
            "data_quality_contract": dq_contract,
            "data_quality_gate": data_quality_gate,
            "data_contract": dq_contract.get("data_contract", {}),
            "target_semantics": dq_contract.get("target_semantics", {}),
            "model_quality_gate": model_quality_gate,
            "catboost_full_factor_bundle": catboost_full_factor_bundle,
        },
    }
    return refresh_excel_export(result)


def _run_existing_legacy_enhanced_analysis_universal(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    region: Optional[str] = None,
    channel: Optional[str] = None,
    segment: Optional[str] = None,
    scenario_calc_mode: str = DEFAULT_SCENARIO_CALC_MODE,
    horizon_days: int = CONFIG["HORIZON_DAYS_DEFAULT"],
):
    analysis_scenario_calc_mode = resolve_scenario_calc_mode(scenario_calc_mode)
    assert analysis_scenario_calc_mode in {"legacy_current", "enhanced_local_factors"}
    txn = normalized_txn.copy()
    dq_contract = getattr(txn, "attrs", {}).get("data_quality_contract", {}) or {}
    data_quality_gate = resolve_data_quality_gate(dq_contract)
    raw_columns = set(normalized_txn.columns.tolist())
    if len(txn) == 0:
        raise ValueError("Пустой датасет после нормализации.")
    daily_base = build_daily_from_transactions(
        txn,
        target_sku,
        target_category=target_category,
        region=region,
        channel=channel,
        segment=segment,
        include_extra_factors=False,
    )
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
    future_dates = forecast_future_dates(pd.Timestamp(daily_base["date"].max()), n_days=int(horizon_days))
    catboost_full_factor_bundle = None
    baseline_price = float(safe_median(daily_base["price"], float(base_ctx.get("price", 1.0))))
    neutral_overrides = {"discount": 0.0, "promotion": 0.0, "freight_value": float(safe_median(daily_base.get("freight_value", pd.Series([0.0])), 0.0))}
    as_is_sim = simulate_horizon_profit(latest_row, float(base_ctx.get("price")), future_dates, baseline_bundle, uplift_bundle, daily_base, base_ctx, shrunk_random_effects, fixed_log_price_coef)
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
        "as_is_demand": as_is_sim["daily"]["actual_sales"].values,
        "baseline_demand": baseline_sim["daily"]["actual_sales"].values,
        "delta_demand": as_is_sim["daily"]["actual_sales"].values - baseline_sim["daily"]["actual_sales"].values,
        "as_is_revenue": as_is_sim["daily"]["revenue"].values,
        "baseline_revenue": baseline_sim["daily"]["revenue"].values,
        "delta_revenue": as_is_sim["daily"]["revenue"].values - baseline_sim["daily"]["revenue"].values,
        "as_is_profit": as_is_sim["daily"]["profit"].values,
        "baseline_profit": baseline_sim["daily"]["profit"].values,
        "delta_profit": as_is_sim["daily"]["profit"].values - baseline_sim["daily"]["profit"].values,
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
        "catboost_full_factor_bundle": catboost_full_factor_bundle,
        "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
    }
    sensitivity_base = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)),
        scenario_calc_mode=analysis_scenario_calc_mode,
    )
    sensitivity_price_minus_5 = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)) * 0.95,
        scenario_calc_mode=analysis_scenario_calc_mode,
    )
    sensitivity_price_plus_5 = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)) * 1.05,
        scenario_calc_mode=analysis_scenario_calc_mode,
    )
    sensitivity_promo_plus_10pp = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)),
        overrides={"promotion": min(1.0, float(base_ctx.get("promotion", 0.0)) + 0.10)},
        scenario_calc_mode=analysis_scenario_calc_mode,
    )
    sensitivity_freight_plus_10pct = run_what_if_projection(
        sensitivity_trained_bundle,
        manual_price=float(base_ctx.get("price", baseline_price)),
        overrides={"freight_value": float(base_ctx.get("freight_value", 0.0)) * 1.10},
        scenario_calc_mode=analysis_scenario_calc_mode,
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
    path_contracts = resolve_path_contracts(
        selected_forecaster=selected_forecaster,
        production_selected_candidate=selected_candidate_name,
        scenario_calc_mode=analysis_scenario_calc_mode,
    )
    final_active_path = path_contracts["final_user_visible_path"]
    if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE and catboost_full_factor_bundle:
        catboost_feature_report = catboost_full_factor_bundle.get("feature_report", pd.DataFrame())
        catboost_feature_importances = pd.DataFrame(catboost_full_factor_bundle.get("feature_importances", []))
    else:
        catboost_feature_report = pd.DataFrame()
        catboost_feature_importances = pd.DataFrame()
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
            "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
            "analysis_scenario_calc_mode_label": scenario_mode_label(analysis_scenario_calc_mode),
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
            "baseline_forecast_path": path_contracts["baseline_forecast_path"],
            "scenario_calculation_path": path_contracts["scenario_calculation_path"],
            "learned_uplift_path": path_contracts["learned_uplift_path"],
            "final_user_visible_path": path_contracts["final_user_visible_path"],
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
        "catboost_full_factor": {
            "enabled": bool((catboost_full_factor_bundle or {}).get("enabled", False)) if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE else False,
            "feature_count": int(len((catboost_full_factor_bundle or {}).get("feature_cols", []))) if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE else 0,
            "cat_feature_count": int(len((catboost_full_factor_bundle or {}).get("cat_feature_names", []))) if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE else 0,
            "holdout_metrics": dict((catboost_full_factor_bundle or {}).get("holdout_metrics", {})) if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE else {},
            "top_features": list((catboost_full_factor_bundle or {}).get("feature_importances", []))[:20] if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE else [],
        },
        "candidate_feature_readiness": candidate_feature_readiness,
        "warnings": warnings,
        "small_mode_info": small_mode_info,
        "feature_usage_report": feature_report.to_dict("records"),
        "scenario_inputs": {"as_is": {"price": float(base_ctx.get("price")), "discount": float(base_ctx.get("discount", 0.0))}, "neutral_baseline": neutral_overrides},
        "scenario_output_summary": {
            "artifact_scope": "analysis_only",
            "report_type": "analysis_only",
            "scenario_status": "as_is" if scenario_sim is None else "computed",
            "scenario_reason": "no_manual_scenario_applied" if scenario_sim is None else "",
            "manual_scenario_present": bool(scenario_sim is not None),
            "manual_scenario_generated": bool(scenario_sim is not None),
            "active_path_contract": final_active_path,
            "learned_uplift_contract": "inactive_production_diagnostic_only",
            "baseline_forecast_path": path_contracts["baseline_forecast_path"],
            "scenario_calculation_path": path_contracts["scenario_calculation_path"],
            "learned_uplift_path": path_contracts["learned_uplift_path"],
            "final_user_visible_path": path_contracts["final_user_visible_path"],
            "final_active_path": final_active_path,
            "production_selected_candidate": "legacy_baseline",
            "selection_mode": "diagnostic_comparison_runtime_frozen_to_legacy",
            "production_selection_reason": "v1_contract_runtime_frozen_to_legacy",
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
    effective_holdout_metrics = holdout_metrics
    if analysis_scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE and isinstance(catboost_full_factor_bundle, dict):
        effective_holdout_metrics = dict(catboost_full_factor_bundle.get("holdout_metrics", holdout_metrics) or holdout_metrics)
    model_quality_gate = _build_model_quality_gate_dict(effective_holdout_metrics)
    result = {
        "history_daily": daily_base,
        "quality_report": {"holdout_metrics": effective_holdout_metrics},
        "model_quality_gate": model_quality_gate,
        "feature_usage_report": feature_report,
        "feature_report": feature_report,
        "neutral_baseline_forecast": baseline_sim["daily"],
        "as_is_forecast": as_is_sim["daily"],
        "scenario_forecast": scenario_forecast,
        "delta_vs_as_is": delta_vs_as_is,
        "delta_vs_baseline": delta_vs_baseline,
        "warnings": warnings,
        "small_mode_info": small_mode_info,
        "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
        "analysis_scenario_calc_mode_label": scenario_mode_label(analysis_scenario_calc_mode),
        "holdout_metrics": pd.DataFrame([effective_holdout_metrics]),
        "elasticity_map": shrunk_random_effects,
        "current_price": float(base_ctx.get("price")),
        "scenario_price": None,
        "current_profit": float(as_is_sim.get("adjusted_profit", as_is_sim.get("total_profit", 0.0))),
        "cost_input_available": bool(getattr(txn, "attrs", {}).get("cost_input_available", bool("cost" in txn.columns and pd.to_numeric(txn.get("cost"), errors="coerce").notna().any()))),
        "cost_is_proxy": bool(getattr(txn, "attrs", {}).get("cost_is_proxy", False)),
        "cost_source": str(getattr(txn, "attrs", {}).get("cost_source", "provided" if "cost" in txn.columns else "missing")),
        "data_quality_contract": dq_contract,
        "data_quality_gate": data_quality_gate,
        "data_contract": dq_contract.get("data_contract", {}),
        "target_semantics": dq_contract.get("target_semantics", {}),
        "blockers": list(data_quality_gate.get("blockers", []) or []),
        "analysis_run_summary_json": json.dumps(run_summary, ensure_ascii=False, indent=2).encode("utf-8"),
        "holdout_predictions_csv": holdout_predictions.to_csv(index=False).encode("utf-8"),
        "holdout_weekly_diagnostics_csv": holdout_weekly_diagnostics.to_csv(index=False).encode("utf-8"),
        "uplift_debug_report_csv": uplift_debug_report.to_csv(index=False).encode("utf-8"),
        "uplift_holdout_trace_csv": uplift_holdout_trace.to_csv(index=False).encode("utf-8"),
        "analysis_baseline_vs_as_is_csv": analysis_baseline_vs_as_is.to_csv(index=False).encode("utf-8"),
        "manual_scenario_summary_json": b"{}",
        "manual_scenario_daily_csv": b"",
        "feature_report_csv": feature_report.to_csv(index=False).encode("utf-8"),
        "catboost_full_factor_report": catboost_feature_report,
        "catboost_full_factor_importances": catboost_feature_importances,
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
            "small_mode_info": small_mode_info,
            "analysis_scenario_calc_mode": analysis_scenario_calc_mode,
            "cost_input_available": bool(getattr(txn, "attrs", {}).get("cost_input_available", bool("cost" in txn.columns))),
            "cost_is_proxy": bool(getattr(txn, "attrs", {}).get("cost_is_proxy", False)),
            "cost_source": str(getattr(txn, "attrs", {}).get("cost_source", "provided" if "cost" in txn.columns else "missing")),
            "data_quality_contract": dq_contract,
            "data_quality_gate": data_quality_gate,
            "data_contract": dq_contract.get("data_contract", {}),
            "target_semantics": dq_contract.get("target_semantics", {}),
            "model_quality_gate": model_quality_gate,
            "catboost_full_factor_bundle": catboost_full_factor_bundle,
        },
    }
    return refresh_excel_export(result)


def _run_what_if_projection_legacy(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: Optional[int] = None,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float = 0.0,
    overrides: Optional[Dict[str, Any]] = None,
    price_guardrail_mode: str = DEFAULT_PRICE_GUARDRAIL_MODE,
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
    guardrail_warnings: List[str] = []
    train_prices = pd.to_numeric(trained_bundle["daily_base"].get("price", pd.Series([manual_price])), errors="coerce").dropna()
    train_min = float(train_prices.min()) if len(train_prices) else float(manual_price)
    train_max = float(train_prices.max()) if len(train_prices) else float(manual_price)
    price_lo = max(0.01, train_min)
    price_hi = max(price_lo, train_max)
    path_price_elasticity = float(trained_bundle.get("pooled_elasticity", CONFIG["PRIOR_ELASTICITY"]))

    def _clip_path_list(path_key: str, lo: float, hi: float, label: str) -> None:
        path = scenario_overrides.get(path_key)
        if not isinstance(path, list):
            return
        changed = False
        clipped_list = []
        for item in path:
            if not isinstance(item, dict):
                clipped_list.append(item)
                continue
            value = float(item.get("value", 0.0))
            clipped = float(np.clip(value, lo, hi))
            changed = changed or (abs(clipped - value) > 1e-12)
            cloned = dict(item)
            cloned["value"] = clipped
            clipped_list.append(cloned)
        scenario_overrides[path_key] = clipped_list
        if changed:
            guardrail_warnings.append(f"{label} был ограничен guardrails диапазоном [{lo:.3f}, {hi:.3f}].")

    def _resolve_price_paths() -> None:
        path = scenario_overrides.get("price_path")
        if not isinstance(path, list):
            scalar_clip = compute_scenario_price_inputs(
                requested_price=float(manual_price),
                train_min=price_lo,
                train_max=price_hi,
                price_guardrail_mode=price_guardrail_mode,
                price_elasticity=path_price_elasticity,
                price_elasticity_prior=float(CONFIG["PRIOR_ELASTICITY"]),
            )
            scenario_overrides["model_price_path"] = []
            scenario_overrides["extrapolation_tail_multiplier_path"] = []
            scenario_overrides["extrapolation_price_ratio_path"] = []
            scenario_overrides["requested_price_path"] = []
            scenario_overrides["financial_price_path"] = []
            return
        requested_path: List[Dict[str, Any]] = []
        financial_path: List[Dict[str, Any]] = []
        model_path: List[Dict[str, Any]] = []
        tail_path: List[Dict[str, Any]] = []
        ratio_path: List[Dict[str, Any]] = []
        clipped_financial = False
        extrapolated = False
        for item in path:
            if not isinstance(item, dict):
                continue
            value = float(item.get("value", manual_price))
            resolved = compute_scenario_price_inputs(
                requested_price=value,
                train_min=price_lo,
                train_max=price_hi,
                price_guardrail_mode=price_guardrail_mode,
                price_elasticity=path_price_elasticity,
                price_elasticity_prior=float(CONFIG["PRIOR_ELASTICITY"]),
            )
            date_value = item.get("date")
            financial_value = float(resolved.get("financial_price", value))
            model_value = float(resolved.get("model_price", value))
            requested_path.append({**item, "date": date_value, "value": value})
            financial_path.append({**item, "date": date_value, "value": financial_value})
            model_path.append({**item, "date": date_value, "value": model_value})
            tail_path.append({"date": date_value, "value": float(resolved.get("extrapolation_tail_multiplier", 1.0))})
            ratio_path.append({"date": date_value, "value": float(resolved.get("extrapolation_price_ratio", 1.0))})
            clipped_financial = clipped_financial or bool(resolved.get("clip_applied", False))
            extrapolated = extrapolated or bool(resolved.get("extrapolation_applied", False))
        scenario_overrides["requested_price_path"] = requested_path
        scenario_overrides["price_path"] = financial_path
        scenario_overrides["financial_price_path"] = financial_path
        scenario_overrides["model_price_path"] = model_path
        scenario_overrides["extrapolation_tail_multiplier_path"] = tail_path
        scenario_overrides["extrapolation_price_ratio_path"] = ratio_path
        scenario_overrides["__price_path_financial_clipped"] = bool(clipped_financial)
        scenario_overrides["__price_path_extrapolated"] = bool(extrapolated)
        if clipped_financial:
            guardrail_warnings.append(f"Price path был ограничен guardrails диапазоном [{price_lo:.3f}, {price_hi:.3f}].")
        if extrapolated:
            guardrail_warnings.append("Price path вне исторического диапазона: спрос посчитан на границе плюс elasticity tail, финансы — по введённой траектории.")

    _resolve_price_paths()
    _clip_path_list("discount_path", 0.0, 0.95, "Discount path")
    _clip_path_list("promo_path", 0.0, 0.70, "Promo path")
    freight_base = float(trained_bundle.get("base_ctx", {}).get("freight_value", 0.0))
    freight_lo = max(0.0, freight_base * 0.5)
    freight_hi = max(freight_lo + 1e-9, freight_base * 1.5 if freight_base > 0 else 1.0)
    _clip_path_list("freight_path", freight_lo, freight_hi, "Freight path")
    _clip_path_list("demand_multiplier_path", 0.70, 1.30, "Demand multiplier path")

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
    price_elasticity = float(trained_bundle.get("pooled_elasticity", CONFIG["PRIOR_ELASTICITY"]))
    clip = compute_scenario_price_inputs(
        requested_price=requested_price,
        train_min=train_min,
        train_max=train_max,
        price_guardrail_mode=price_guardrail_mode,
        price_elasticity=price_elasticity,
        price_elasticity_prior=float(CONFIG["PRIOR_ELASTICITY"]),
    )
    model_price = float(clip["model_price"])
    financial_price = float(clip["financial_price"])
    base_net_price_ref = float(max(0.01, current_price_raw * (1.0 - baseline_discount)))
    model_net_price = float(max(0.01, model_price * (1.0 - scenario_discount)))
    financial_net_price = float(max(0.01, financial_price * (1.0 - scenario_discount)))

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
        "price_guardrail_mode": str(clip["price_guardrail_mode"]),
        "requested_price_gross": float(manual_price),
        "safe_price_gross": float(model_price),
        "model_price_gross": float(model_price),
        "financial_price_gross": float(financial_price),
        "applied_price_gross": float(financial_price),
        "current_price_gross": float(current_price_raw),
        "current_discount": float(baseline_discount),
        "applied_discount": float(scenario_discount),
        "current_price_net": float(base_net_price_ref),
        "safe_price_net": float(model_net_price),
        "model_price_net": float(model_net_price),
        "financial_price_net": float(financial_net_price),
        "applied_price_net": float(financial_net_price),
        "promotion": float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
        "freight_value": float(scenario_freight),
        "cost": float(scenario_cost),
        "demand_multiplier": float(scenario_overrides.get("manual_shock_multiplier", 1.0)),
        "price_clipped": bool(clip.get("price_clipped", False)),
        "clip_applied": bool(clip.get("clip_applied", False)),
        "model_price_clipped": bool(clip.get("model_price_clipped", False)),
        "price_out_of_range": bool(clip.get("price_out_of_range", False)),
        "extrapolation_applied": bool(clip.get("extrapolation_applied", False)),
        "guardrail_warning_type": "economic_extrapolation" if bool(clip.get("extrapolation_applied", False)) else ("safe_clip" if bool(clip.get("clip_applied", False)) else ""),
        "clip_reason": clip.get("clip_reason"),
        "lower_clip": float(train_min) if np.isfinite(train_min) else None,
        "upper_clip": float(train_max) if np.isfinite(train_max) else None,
    }

    scenario_inputs = {
        "demand_price_baseline": float(base_net_price_ref),
        "demand_price_scenario": float(model_net_price),
        "gross_price_baseline": float(current_price_raw),
        "gross_price_scenario": float(model_price),
        "financial_gross_price_scenario": float(financial_price),
        "base_price": float(base_net_price_ref),
        "scenario_price": float(model_net_price),
        "baseline_net_price": float(base_net_price_ref),
        "scenario_net_price": float(financial_net_price),
        "price_elasticity": price_elasticity,
        "extrapolation_tail_multiplier": float(clip.get("extrapolation_tail_multiplier", 1.0)),
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
        abs(effective_scenario["requested_price_gross"] - effective_scenario["current_price_gross"]) > 1e-9
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
        daily["price"] = float(financial_price)
        daily["discount"] = float(scenario_discount)
        daily["cost"] = scenario_cost
        daily["freight_value"] = scenario_freight
        daily["net_unit_price"] = financial_net_price
        daily["requested_price_gross"] = float(requested_price)
        daily["safe_price_gross"] = float(model_price)
        daily["model_price_gross"] = float(model_price)
        daily["model_price_net"] = float(model_net_price)
        daily["price_for_model"] = float(model_price)
        daily["applied_price_gross"] = float(financial_price)
        daily["applied_price_net"] = float(financial_net_price)
        daily["scenario_price_gross"] = float(financial_price)
        daily["scenario_price_net"] = float(financial_net_price)
        daily["price_guardrail_mode"] = str(clip["price_guardrail_mode"])
        daily["price_out_of_range"] = bool(clip.get("price_out_of_range", False))
        daily["price_clipped"] = bool(clip.get("price_clipped", False))
        daily["clip_applied"] = bool(clip.get("clip_applied", False))
        daily["extrapolation_applied"] = bool(clip.get("extrapolation_applied", False))
        daily["model_boundary_price_gross"] = float(clip.get("model_boundary_price_gross", np.nan))
        daily["extrapolation_from_price_gross"] = float(clip.get("extrapolation_from_price_gross", np.nan))
        daily["extrapolation_to_price_gross"] = float(clip.get("extrapolation_to_price_gross", np.nan))
        daily["extrapolation_price_ratio"] = float(clip.get("extrapolation_price_ratio", 1.0))
        daily["extrapolation_tail_multiplier"] = float(clip.get("extrapolation_tail_multiplier", 1.0))
        daily["elasticity_used"] = float(clip.get("elasticity_used", np.nan))
        daily["elasticity_source"] = str(clip.get("elasticity_source", ""))
        daily["scenario_price_effect_source"] = str(clip.get("scenario_price_effect_source", "scenario_engine_recompute_from_baseline"))
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
    price_plausibility = 1.0 if is_price_plausible(effective_scenario["requested_price_gross"], train_min, train_max, margin=0.25) else 0.0
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
    path_contracts = resolve_path_contracts(
        selected_forecaster=str(baseline_bundle_meta.get("selected_forecaster", "weekly_model")),
        production_selected_candidate=str(baseline_bundle_meta.get("selected_candidate", "legacy_baseline")),
        scenario_calc_mode="legacy_current",
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
        "active_path_contract": path_contracts["final_user_visible_path"],
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
    support_info = build_scenario_support_info(history_daily, effective_scenario, scenario_overrides)
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
        "ood_flag": bool(not is_price_plausible(effective_scenario["requested_price_gross"], train_min, train_max, margin=0.25)),
        "requested_price": requested_price,
        "model_price": model_price,
        "price_for_model": model_price,
        "safe_price_gross": model_price,
        "financial_price": financial_price,
        "applied_price_gross": financial_price,
        "applied_price_net": financial_net_price,
        "current_price_raw": float(current_price_raw),
        "price_guardrail_mode": str(clip["price_guardrail_mode"]),
        "price_clipped": bool(clip["price_clipped"]),
        "clip_applied": bool(clip["clip_applied"]),
        "price_out_of_range": bool(clip["price_out_of_range"]),
        "extrapolation_applied": bool(clip["extrapolation_applied"]),
        "model_boundary_price_gross": float(clip.get("model_boundary_price_gross", np.nan)),
        "extrapolation_from_price_gross": float(clip.get("extrapolation_from_price_gross", np.nan)),
        "extrapolation_to_price_gross": float(clip.get("extrapolation_to_price_gross", np.nan)),
        "extrapolation_price_ratio": float(clip.get("extrapolation_price_ratio", 1.0)),
        "extrapolation_tail_multiplier": float(clip.get("extrapolation_tail_multiplier", 1.0)),
        "elasticity_used": float(clip.get("elasticity_used", np.nan)),
        "elasticity_source": str(clip.get("elasticity_source", "")),
        "clip_reason": str(clip["clip_reason"]),
        "net_price_supported": bool(net_support.get("net_price_supported", True)),
        "net_price_support": net_support,
        "scenario_price_effect_source": str(clip.get("scenario_price_effect_source", "scenario_engine_recompute_from_baseline")),
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
        "baseline_forecast_path": path_contracts["baseline_forecast_path"],
        "scenario_calculation_path": path_contracts["scenario_calculation_path"],
        "learned_uplift_path": path_contracts["learned_uplift_path"],
        "final_user_visible_path": path_contracts["final_user_visible_path"],
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
            "scenario_price": float(financial_price),
            "requested_price": float(requested_price),
            "model_price": float(model_price),
            "financial_price": float(financial_price),
            "base_promo": float(current_ctx.get("promotion", 0.0)),
            "scenario_promo": float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
            "base_freight": float(current_ctx.get("freight_value", 0.0)),
            "scenario_freight": float(scenario_freight),
            "shock_multiplier": float(demand_multiplier),
        },
        "scenario_calc_mode_label": scenario_mode_label("legacy_current"),
        "active_path_contract_label": scenario_contract_label(runtime_meta["active_path_contract"]),
        "effect_source_label": effect_source_label("scenario_engine_recompute_from_baseline"),
        "support_label": str(support_info.get("support_label", "medium")),
        "scenario_support_info": support_info,
        "legacy_or_enhanced_label": "legacy",
        "applied_path_summary": {
            "mode": scenario_mode_label("legacy_current"),
            "segment_count": 0,
            "price_net_min": float(effective_scenario.get("applied_price_net", np.nan)),
            "price_net_avg": float(effective_scenario.get("applied_price_net", np.nan)),
            "price_net_max": float(effective_scenario.get("applied_price_net", np.nan)),
            "promo_days": 0,
            "promo_share": float(1.0 if float(effective_scenario.get("promotion", 0.0)) > 0 else 0.0),
            "avg_freight": float(effective_scenario.get("freight_value", 0.0)),
            "avg_demand_multiplier": float(scenario_overrides.get("manual_shock_multiplier", 1.0)),
            "segment_shocks": list(scenario_overrides.get("shocks", [])) if isinstance(scenario_overrides.get("shocks", []), list) else [],
            "support_label": str(support_info.get("support_label", "unknown")),
            "warnings": list(support_info.get("warnings", [])),
        },
        "effective_scenario": effective_scenario,
        "confidence_factors": scenario_result.get("confidence", {}),
        "warnings": scenario_result.get("warnings", []) + backend_warning + ([str(net_support.get("net_price_warning", ""))] if net_support.get("net_price_warning") else []) + list(support_info.get("warnings", [])),
    }


def _run_what_if_projection_enhanced(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: Optional[int] = None,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float = 0.0,
    overrides: Optional[Dict[str, Any]] = None,
    price_guardrail_mode: str = DEFAULT_PRICE_GUARDRAIL_MODE,
) -> Dict[str, Any]:
    legacy_result = _run_what_if_projection_legacy(
        trained_bundle=trained_bundle,
        manual_price=manual_price,
        freight_multiplier=freight_multiplier,
        demand_multiplier=demand_multiplier,
        horizon_days=horizon_days,
        discount_multiplier=discount_multiplier,
        cost_multiplier=cost_multiplier,
        stock_cap=stock_cap,
        overrides=overrides,
        price_guardrail_mode=price_guardrail_mode,
    )
    scenario_overrides = dict(overrides or {})
    scenario_overrides.setdefault("freight_multiplier", float(freight_multiplier))
    scenario_overrides.setdefault("discount_multiplier", float(discount_multiplier))
    scenario_overrides.setdefault("cost_multiplier", float(cost_multiplier))
    if stock_cap:
        scenario_overrides["stock_cap"] = float(stock_cap)
    if float(demand_multiplier) != 1.0:
        scenario_overrides["manual_shock_multiplier"] = float(demand_multiplier)
    guardrail_warnings: List[str] = []
    train_prices = pd.to_numeric(trained_bundle["daily_base"].get("price", pd.Series([manual_price])), errors="coerce").dropna()
    train_min = float(train_prices.min()) if len(train_prices) else float(manual_price)
    train_max = float(train_prices.max()) if len(train_prices) else float(manual_price)
    price_lo = max(0.01, train_min)
    price_hi = max(price_lo, train_max)
    path_price_elasticity = float(trained_bundle.get("pooled_elasticity", CONFIG["PRIOR_ELASTICITY"]))

    def _clip_path_list(path_key: str, lo: float, hi: float, label: str) -> None:
        path = scenario_overrides.get(path_key)
        if not isinstance(path, list):
            return
        changed = False
        clipped: List[Dict[str, Any]] = []
        for item in path:
            if not isinstance(item, dict):
                continue
            value = float(item.get("value", 0.0))
            new_value = float(np.clip(value, lo, hi))
            changed = changed or (abs(new_value - value) > 1e-12)
            x = dict(item)
            x["value"] = new_value
            clipped.append(x)
        scenario_overrides[path_key] = clipped
        if changed:
            guardrail_warnings.append(f"{label} был ограничен guardrails диапазоном [{lo:.3f}, {hi:.3f}].")

    def _resolve_price_paths() -> None:
        path = scenario_overrides.get("price_path")
        if not isinstance(path, list):
            scenario_overrides.setdefault("model_price_path", [])
            scenario_overrides.setdefault("extrapolation_tail_multiplier_path", [])
            scenario_overrides.setdefault("extrapolation_price_ratio_path", [])
            scenario_overrides.setdefault("requested_price_path", [])
            scenario_overrides.setdefault("financial_price_path", [])
            scenario_overrides.setdefault("extrapolation_tail_multiplier", float(legacy_result.get("extrapolation_tail_multiplier", 1.0)))
            scenario_overrides.setdefault("extrapolation_price_ratio", float(legacy_result.get("extrapolation_price_ratio", 1.0)))
            return
        requested_path: List[Dict[str, Any]] = []
        financial_path: List[Dict[str, Any]] = []
        model_path: List[Dict[str, Any]] = []
        tail_path: List[Dict[str, Any]] = []
        ratio_path: List[Dict[str, Any]] = []
        clipped_financial = False
        extrapolated = False
        for item in path:
            if not isinstance(item, dict):
                continue
            value = float(item.get("value", manual_price))
            resolved = compute_scenario_price_inputs(
                requested_price=value,
                train_min=price_lo,
                train_max=price_hi,
                price_guardrail_mode=price_guardrail_mode,
                price_elasticity=path_price_elasticity,
                price_elasticity_prior=float(CONFIG["PRIOR_ELASTICITY"]),
            )
            date_value = item.get("date")
            financial_value = float(resolved.get("financial_price", value))
            model_value = float(resolved.get("model_price", value))
            requested_path.append({**item, "date": date_value, "value": value})
            financial_path.append({**item, "date": date_value, "value": financial_value})
            model_path.append({**item, "date": date_value, "value": model_value})
            tail_path.append({"date": date_value, "value": float(resolved.get("extrapolation_tail_multiplier", 1.0))})
            ratio_path.append({"date": date_value, "value": float(resolved.get("extrapolation_price_ratio", 1.0))})
            clipped_financial = clipped_financial or bool(resolved.get("clip_applied", False))
            extrapolated = extrapolated or bool(resolved.get("extrapolation_applied", False))
        scenario_overrides["requested_price_path"] = requested_path
        scenario_overrides["price_path"] = financial_path
        scenario_overrides["financial_price_path"] = financial_path
        scenario_overrides["model_price_path"] = model_path
        scenario_overrides["extrapolation_tail_multiplier_path"] = tail_path
        scenario_overrides["extrapolation_price_ratio_path"] = ratio_path
        scenario_overrides["__price_path_financial_clipped"] = bool(clipped_financial)
        scenario_overrides["__price_path_extrapolated"] = bool(extrapolated)
        if clipped_financial:
            guardrail_warnings.append(f"Price path был ограничен guardrails диапазоном [{price_lo:.3f}, {price_hi:.3f}].")
        if extrapolated:
            guardrail_warnings.append("Price path вне исторического диапазона: спрос посчитан на границе плюс elasticity tail, финансы — по введённой траектории.")

    _resolve_price_paths()
    _clip_path_list("discount_path", 0.0, 0.95, "Discount path")
    _clip_path_list("promo_path", 0.0, 0.70, "Promo path")
    freight_base = float(trained_bundle.get("base_ctx", {}).get("freight_value", 0.0))
    freight_lo = max(0.0, freight_base * 0.5)
    freight_hi = max(freight_lo + 1e-9, freight_base * 1.5 if freight_base > 0 else 1.0)
    _clip_path_list("freight_path", freight_lo, freight_hi, "Freight path")
    _clip_path_list("demand_multiplier_path", 0.70, 1.30, "Demand multiplier path")

    base_history = trained_bundle["daily_base"].copy()
    current_ctx = dict(trained_bundle["base_ctx"])
    latest_row_current = dict(trained_bundle["latest_row"])
    future_dates = trained_bundle["future_dates"]
    if horizon_days is not None:
        future_dates = forecast_future_dates(pd.Timestamp(base_history["date"].max()), n_days=int(horizon_days))
    baseline_sim = simulate_horizon_profit(
        latest_row_current,
        float(current_ctx.get("price", manual_price)),
        future_dates,
        trained_bundle["baseline_bundle"],
        trained_bundle["uplift_bundle"],
        base_history,
        current_ctx,
        trained_bundle["elasticity_map"],
        trained_bundle["pooled_elasticity"],
        overrides={
            "promotion": float(current_ctx.get("promotion", 0.0)),
            "freight_multiplier": 1.0,
            "discount_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "manual_shock_multiplier": 1.0,
        },
    )
    baseline_daily = baseline_sim["daily"].copy()
    effective = legacy_result.get("effective_scenario", {})
    shocks = list(scenario_overrides.get("shocks", [])) if isinstance(scenario_overrides.get("shocks", []), list) else []
    has_demand_path = bool(scenario_overrides.get("demand_multiplier_path"))
    if float(demand_multiplier) != 1.0 and len(future_dates) and not has_demand_path:
        shocks.append(
            {
                "shock_name": "demand_multiplier",
                "shock_type": "percent",
                "shock_value": float(demand_multiplier) - 1.0,
                "start_date": str(pd.to_datetime(future_dates["date"]).min().date()),
                "end_date": str(pd.to_datetime(future_dates["date"]).max().date()),
            }
        )

    small_mode_meta = trained_bundle.get("small_mode_info")
    if not isinstance(small_mode_meta, dict):
        small_mode_meta = {"small_mode": bool(trained_bundle.get("small_mode", False))}
    enhanced = run_enhanced_scenario(
        baseline_daily=baseline_daily,
        current_ctx=current_ctx,
        future_dates=future_dates,
        scenario_overrides=scenario_overrides,
        pooled_elasticity=float(trained_bundle.get("pooled_elasticity", CONFIG["PRIOR_ELASTICITY"])),
        small_mode_info=dict(small_mode_meta),
        requested_price=float(manual_price),
        model_price=float(legacy_result.get("model_price", manual_price)),
        financial_price=float(legacy_result.get("financial_price", legacy_result.get("applied_price_gross", manual_price))),
        baseline_discount=float(current_ctx.get("discount", 0.0)),
        scenario_discount=float(effective.get("applied_discount", current_ctx.get("discount", 0.0))),
        baseline_cost=float(current_ctx.get("cost", current_ctx.get("price", manual_price) * CONFIG["COST_PROXY_RATIO"])),
        scenario_cost=float(effective.get("cost", current_ctx.get("cost", 0.0))),
        baseline_freight=float(current_ctx.get("freight_value", 0.0)),
        scenario_freight=float(effective.get("freight_value", current_ctx.get("freight_value", 0.0))),
        shocks=shocks,
    )
    daily = baseline_daily.copy()
    profile = enhanced["scenario_profile"]
    daily["pred_sales"] = profile["scenario_demand_raw"].to_numpy(dtype=float)
    daily["unconstrained_demand"] = profile["scenario_demand_raw"].to_numpy(dtype=float)
    daily["actual_sales"] = profile["actual_sales"].to_numpy(dtype=float)
    daily["lost_sales"] = profile["lost_sales"].to_numpy(dtype=float)
    daily["revenue"] = profile["revenue"].to_numpy(dtype=float)
    daily["profit"] = profile["profit"].to_numpy(dtype=float)
    daily["price"] = profile["scenario_price_gross"].to_numpy(dtype=float)
    daily["discount"] = profile["scenario_discount"].to_numpy(dtype=float)
    daily["net_unit_price"] = profile["scenario_price_net"].to_numpy(dtype=float)
    daily["promotion"] = profile["scenario_promotion"].to_numpy(dtype=float)
    daily["freight_value"] = profile["scenario_freight_value"].to_numpy(dtype=float)
    daily["cost"] = profile["scenario_cost"].to_numpy(dtype=float)
    daily["stock"] = profile["available_stock"].to_numpy(dtype=float)
    for scalar_col in [
        "price_guardrail_mode",
        "price_out_of_range",
        "price_clipped",
        "clip_applied",
        "extrapolation_applied",
        "guardrail_warning_type",
        "model_boundary_price_gross",
        "extrapolation_from_price_gross",
        "extrapolation_to_price_gross",
        "extrapolation_price_ratio",
        "extrapolation_tail_multiplier",
        "scenario_price_effect_source",
    ]:
        daily[scalar_col] = legacy_result.get(scalar_col, (legacy_result.get("effective_scenario", {}) or {}).get(scalar_col, np.nan))
    for col in [
        "price_effect",
        "promo_effect",
        "freight_effect",
        "standard_multiplier",
        "shock_multiplier",
        "shock_units",
        "scenario_demand_raw",
        "lost_sales",
        "requested_price_gross",
        "safe_price_gross",
        "model_price_gross",
        "model_price_net",
        "price_for_model",
        "applied_price_gross",
        "applied_price_net",
        "scenario_price_gross",
        "scenario_price_net",
        "extrapolation_tail_multiplier",
        "extrapolation_price_ratio",
    ]:
        if col in profile.columns:
            daily[col] = pd.to_numeric(profile[col], errors="coerce").to_numpy(dtype=float)
    daily["final_multiplier"] = (
        pd.to_numeric(daily.get("standard_multiplier", pd.Series(np.ones(len(daily)))), errors="coerce").fillna(1.0)
        * pd.to_numeric(daily.get("shock_multiplier", pd.Series(np.ones(len(daily)))), errors="coerce").fillna(1.0)
    )

    conf_obj = enhanced.get("confidence", {})
    support_info = build_scenario_support_info_from_paths(base_history, profile, scenario_overrides)
    price_score = float((conf_obj.get("price", {}) or {}).get("score", legacy_result.get("confidence", 0.5)))
    promo_score = float((conf_obj.get("promo", {}) or {}).get("score", 0.5))
    freight_score = float((conf_obj.get("freight", {}) or {}).get("score", 0.5))
    if int(support_info.get("local_price_support_days", 0)) < 14:
        price_score = min(price_score, 0.55)
    if int(support_info.get("unique_price_points", 0)) < 4:
        price_score = min(price_score, 0.50)
    if int(support_info.get("price_changes", 0)) == 0:
        price_score = min(price_score, 0.35)
    if abs(float(effective.get("promotion", 0.0)) - float(current_ctx.get("promotion", 0.0))) > 1e-9:
        if int(support_info.get("promo_active_days", 0)) < 14 or int(support_info.get("promo_change_days", 0)) < 6:
            promo_score = min(promo_score, 0.45)
    if float(support_info.get("promotion_positive_share", 0.0)) < 0.05:
        promo_score = min(promo_score, 0.35)
    if abs(float(effective.get("freight_value", 0.0)) - float(current_ctx.get("freight_value", 0.0))) > 1e-9:
        if int(support_info.get("freight_change_days", 0)) < 8:
            freight_score = min(freight_score, 0.45)
    if float(support_info.get("freight_variation", 0.0)) < 0.02:
        freight_score = min(freight_score, 0.35)
    conf_score = float(
        np.clip(
            0.5 * price_score + 0.25 * promo_score + 0.25 * freight_score,
            0.05,
            0.98,
        )
    )
    low_factors = int(price_score <= 0.45) + int(promo_score <= 0.45) + int(freight_score <= 0.45)
    if low_factors >= 2:
        conf_score = min(conf_score, 0.45)
    if low_factors >= 3:
        conf_score = min(conf_score, 0.35)
    conf_label = "Высокая" if conf_score >= 0.75 else ("Средняя" if conf_score >= 0.45 else "Низкая")
    path_extrapolated = bool(scenario_overrides.get("__price_path_extrapolated", False))
    path_clipped = bool(scenario_overrides.get("__price_path_financial_clipped", False))
    path_tail = pd.to_numeric(profile.get("extrapolation_tail_multiplier", pd.Series(np.ones(len(profile)))), errors="coerce").fillna(1.0)
    path_ratio = pd.to_numeric(profile.get("extrapolation_price_ratio", pd.Series(np.ones(len(profile)))), errors="coerce").fillna(1.0)
    valid_path_elasticity = bool(np.isfinite(path_price_elasticity) and VALID_ELASTICITY_MIN <= path_price_elasticity <= VALID_ELASTICITY_MAX)
    path_elasticity_used = float(path_price_elasticity if valid_path_elasticity else CONFIG["PRIOR_ELASTICITY"])
    if valid_path_elasticity:
        path_elasticity_source = "pooled_price_elasticity"
    elif np.isfinite(path_price_elasticity):
        path_elasticity_source = "fallback_prior_out_of_safe_range"
    else:
        path_elasticity_source = "fallback_prior_invalid_elasticity"
    if path_extrapolated:
        daily["scenario_price_effect_source"] = "boundary_plus_elasticity_tail"
        daily["elasticity_used"] = path_elasticity_used
        daily["elasticity_source"] = path_elasticity_source
    out = dict(legacy_result)
    baseline_path = str(legacy_result.get("baseline_forecast_path", "weekly_ml_baseline"))
    scenario_path = "enhanced_local_factor_layer"
    learned_uplift_path = str(legacy_result.get("learned_uplift_path", "inactive_production_diagnostic_only"))
    visible_path = f"{baseline_path} + {scenario_path}"
    out.update(
        {
            "daily": daily,
            "demand_total": float(np.sum(profile["actual_sales"])),
            "revenue_total": float(np.sum(profile["revenue"])),
            "profit_total": float(np.sum(profile["profit"])),
            "profit_total_raw": float(np.sum(profile["profit"])),
            "profit_total_adjusted": float(np.sum(profile["profit"])),
            "lost_sales_total": float(np.sum(profile["lost_sales"])),
            "confidence": conf_score,
            "confidence_scenario": conf_score,
            "confidence_label": conf_label,
            "uncertainty": 1.0 - conf_score,
            "price_path_active": bool(scenario_overrides.get("requested_price_path")),
            "price_clipped": bool(path_clipped or legacy_result.get("price_clipped", False)),
            "clip_applied": bool(path_clipped or legacy_result.get("clip_applied", False)),
            "price_out_of_range": bool(path_extrapolated or legacy_result.get("price_out_of_range", False)),
            "extrapolation_applied": bool(path_extrapolated or legacy_result.get("extrapolation_applied", False)),
            "extrapolation_tail_multiplier": float(path_tail.mean()) if len(path_tail) else float(legacy_result.get("extrapolation_tail_multiplier", 1.0)),
            "extrapolation_price_ratio": float(path_ratio.mean()) if len(path_ratio) else float(legacy_result.get("extrapolation_price_ratio", 1.0)),
            "elasticity_used": path_elasticity_used if path_extrapolated else float(legacy_result.get("elasticity_used", np.nan)),
            "elasticity_source": path_elasticity_source if path_extrapolated else str(legacy_result.get("elasticity_source", "")),
            "scenario_price_effect_source": "boundary_plus_elasticity_tail" if path_extrapolated else str(legacy_result.get("scenario_price_effect_source", "baseline_daily_x_local_factor_layer")),
            "scenario_driver_mode": "baseline_daily_plus_local_factor_layer",
            "active_path_contract": visible_path,
            "baseline_forecast_path": baseline_path,
            "scenario_calculation_path": scenario_path,
            "learned_uplift_path": learned_uplift_path,
            "final_user_visible_path": visible_path,
            "scenario_calc_mode": "enhanced_local_factors",
            "scenario_engine_version": "v1_enhanced_local_factors",
            "effect_source": "baseline_daily_x_local_factor_layer",
            "legacy_or_enhanced_label": "enhanced",
            "scenario_calc_mode_label": scenario_mode_label("enhanced_local_factors"),
            "active_path_contract_label": scenario_contract_label(visible_path),
            "effect_source_label": effect_source_label("baseline_daily_x_local_factor_layer"),
            "support_label": str(support_info.get("support_label", "medium")),
            "scenario_support_info": support_info,
            "applied_path_summary": {
                "mode": scenario_mode_label("enhanced_local_factors"),
                "segment_count": int(support_info.get("segment_count", 0)),
                "price_net_min": float(support_info.get("path_price_min", np.nan)),
                "price_net_avg": float(support_info.get("path_price_avg", np.nan)),
                "price_net_max": float(support_info.get("path_price_max", np.nan)),
                "promo_days": int(support_info.get("path_promo_days", 0)),
                "promo_share": float(support_info.get("path_promo_share", 0.0)),
                "avg_freight": float(support_info.get("path_avg_freight", 0.0)),
                "avg_demand_multiplier": float(support_info.get("path_avg_demand_multiplier", 1.0)),
                "segment_shocks": list((scenario_overrides or {}).get("shocks", [])),
                "support_label": str(support_info.get("support_label", "unknown")),
                "warnings": list(support_info.get("warnings", [])) + list(guardrail_warnings),
            },
            "effect_breakdown": enhanced["effect_breakdown"],
            "daily_effects_summary": profile[
                ["date", "price_effect", "promo_effect", "freight_effect", "standard_multiplier", "shock_multiplier", "shock_units"]
            ].to_dict("records"),
            "warnings": list(enhanced.get("warnings", [])) + list(support_info.get("warnings", [])) + list(guardrail_warnings),
            "confidence_factors": conf_obj,
            "effects": {
                "price_effect": float(np.mean(enhanced["price_effect_vector"])) if len(enhanced["price_effect_vector"]) else 1.0,
                "promo_effect": float(np.mean(enhanced["promo_effect_vector"])) if len(enhanced["promo_effect_vector"]) else 1.0,
                "freight_effect": float(np.mean(enhanced["freight_effect_vector"])) if len(enhanced["freight_effect_vector"]) else 1.0,
                "stock_effect": 1.0,
                "shock_multiplier_mean": float(np.mean(enhanced["shock_multiplier"])) if len(enhanced["shock_multiplier"]) else 1.0,
            },
        }
    )
    return out



def _series_min_max(frame: pd.DataFrame, column: str, fallback: float = np.nan) -> Tuple[float, float]:
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        val = float(fallback) if np.isfinite(fallback) else np.nan
        return val, val
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        val = float(fallback) if np.isfinite(fallback) else np.nan
        return val, val
    return float(values.min()), float(values.max())


def _finalize_scenario_contract(result: Dict[str, Any], mode: str, price_guardrail_mode: str) -> Dict[str, Any]:
    out = dict(result or {})
    daily = out.get("daily")
    effective = dict(out.get("effective_scenario") or {})
    if not isinstance(daily, pd.DataFrame):
        daily = pd.DataFrame()
    requested_fallback = float(out.get("requested_price", effective.get("requested_price_gross", np.nan)))
    model_fallback = float(out.get("model_price", out.get("price_for_model", effective.get("model_price_gross", effective.get("safe_price_gross", np.nan)))))
    financial_fallback = float(out.get("financial_price", effective.get("financial_price_gross", effective.get("applied_price_gross", np.nan))))
    req_min, req_max = _series_min_max(daily, "requested_price_gross", requested_fallback)
    model_min, model_max = _series_min_max(daily, "model_price_gross", model_fallback)
    fin_min, fin_max = _series_min_max(daily, "applied_price_gross", financial_fallback)
    raw_requested_min, raw_requested_max = req_min, req_max
    is_path = bool(out.get("price_path_active", False) or abs(req_max - req_min) > 1e-9 or abs(model_max - model_min) > 1e-9 or abs(fin_max - fin_min) > 1e-9)
    price_policy = {
        "mode": str(price_guardrail_mode),
        "is_path": bool(is_path),
        "requested_price_min": raw_requested_min,
        "requested_price_max": raw_requested_max,
        "model_price_min": model_min,
        "model_price_max": model_max,
        "financial_price_min": fin_min,
        "financial_price_max": fin_max,
        "applied_price_min": fin_min,
        "applied_price_max": fin_max,
        "extrapolation_applied": bool(out.get("extrapolation_applied", effective.get("extrapolation_applied", False))),
        "clip_applied": bool(out.get("clip_applied", effective.get("clip_applied", out.get("price_clipped", False)))),
        "price_out_of_range": bool(out.get("price_out_of_range", effective.get("price_out_of_range", False))),
        "extrapolation_tail_multiplier": float(out.get("extrapolation_tail_multiplier", 1.0)),
        "extrapolation_price_ratio": float(out.get("extrapolation_price_ratio", 1.0)),
        "elasticity_source": str(out.get("elasticity_source", effective.get("elasticity_source", ""))),
    }
    out["price_policy"] = {**dict(out.get("price_policy") or {}), **price_policy}
    out.setdefault("factor_policy", {
        "mode": str(mode),
        "effect_source": str(out.get("effect_source", out.get("scenario_price_effect_source", ""))),
        "learned_uplift_active": bool(out.get("learned_uplift_active", False)),
        "demand_shock_is_manual_hypothesis": True,
    })
    out.setdefault("calculation_trace", {
        "baseline_source": str(out.get("baseline_forecast_path", "unknown")),
        "scenario_mode": str(mode),
        "price_effect_source": str(out.get("scenario_price_effect_source", out.get("effect_source", "unknown"))),
        "promo_effect_source": str(out.get("effect_source", "scenario_layer")),
        "freight_effect_source": str(out.get("effect_source", "scenario_layer")),
        "shock_source": "manual_shock_overlay",
        "guardrail_mode": str(price_guardrail_mode),
        "extrapolation_policy": "boundary_plus_elasticity_tail" if price_policy["extrapolation_applied"] else "none",
        "cost_policy": "provided" if not bool(out.get("cost_proxied", out.get("cost_is_proxy", False))) else "proxy_price_65",
        "stock_policy": "stock_cap_if_provided; missing_stock_never_evidence_of_no_stockout",
    })
    out.setdefault("guardrail_warnings", [w for w in out.get("warnings", []) if "guardrail" in str(w).lower() or "экстраполя" in str(w).lower() or "диапаз" in str(w).lower()])
    gate = resolve_recommendation_gate(
        price_policy=price_policy,
        data_quality={"flat_history": bool(out.get("flat_history", False))},
        model_quality={"wape": out.get("wape", out.get("holdout_wape"))},
        factor_policy={
            **dict(out.get("factor_policy") or {}),
            "manual_demand_shock_main_driver": bool(out.get("manual_demand_shock_main_driver", False)),
        },
        economic_significance={
            "profit_delta_pct": out.get("profit_delta_pct", out.get("profit_delta")),
            "conservative_profit_delta_pct": out.get("conservative_profit_delta_pct"),
            "cost_proxied": out.get("cost_proxied", out.get("cost_is_proxy", False)),
            "cost_missing": out.get("cost_missing", False),
            "profit_action": True,
        },
        cost_policy={
            "cost_proxied": bool(out.get("cost_proxied", out.get("cost_is_proxy", False))),
            "cost_missing": bool(out.get("cost_missing", False)),
            "profit_action": True,
        },
        decision_reliability={"warnings": out.get("guardrail_warnings", []), "allow_unknown_wape_for_test_recommendation": True},
    )
    out["calculation_gate"] = gate["calculation_gate"]
    out["recommendation_gate"] = gate["recommendation_gate"]
    out["recommendation_gate_reasons"] = gate["reasons"]
    out["recommendation_gate_warnings"] = gate["warnings"]
    out["scenario_contract"] = {
        "scenario_id": str(out.get("scenario_run_id", out.get("scenario_id", "scenario"))),
        "scenario_mode": str(mode),
        "baseline_source": str((out.get("calculation_trace") or {}).get("baseline_source", out.get("baseline_source", "unknown"))),
        "effect_source": str((out.get("calculation_trace") or {}).get("price_effect_source", out.get("effect_source", "unknown"))),
        "requested_inputs": {
            **dict(out.get("scenario_requested_inputs") or {}),
            "manual_price": requested_fallback,
            "price": requested_fallback,
        },
        "applied_inputs": {
            "model_price": model_fallback,
            "financial_price": financial_fallback,
            "applied_discount": float(pd.to_numeric(daily.get("discount", pd.Series([np.nan])), errors="coerce").median()) if isinstance(daily, pd.DataFrame) and "discount" in daily.columns else None,
            "applied_promotion": float(pd.to_numeric(daily.get("promotion", pd.Series([np.nan])), errors="coerce").median()) if isinstance(daily, pd.DataFrame) and "promotion" in daily.columns else None,
            "applied_freight_value": float(pd.to_numeric(daily.get("freight_value", pd.Series([np.nan])), errors="coerce").median()) if isinstance(daily, pd.DataFrame) and "freight_value" in daily.columns else None,
            "applied_factor_overrides": dict((out.get("scenario_requested_inputs") or {}).get("factor_overrides") or {}),
        },
        "price_policy": out.get("price_policy", {}),
        "cost_policy": {"cost_source": out.get("cost_source"), "cost_is_proxy": out.get("cost_is_proxy"), "profit_is_reliable": out.get("profit_is_reliable")},
        "stock_policy": {"stock_cap": out.get("stock_cap"), "stock_missing_never_evidence_no_stockout": True},
        "factor_policy": out.get("factor_policy", {}),
        "model_quality_gate": out.get("model_quality_gate", {"wape": out.get("wape", out.get("holdout_wape"))}),
        "recommendation_gate": gate,
        "calculation_vs_recommendation": {
            "calculation_allowed": gate.get("calculation_gate") != "blocked",
            "recommendation_allowed": bool((gate.get("usage_policy") or {}).get("can_recommend_action", False)),
            "recommendation_status": gate.get("decision_status", gate.get("recommendation_status")),
        },
        "calculation_gate": gate["calculation_gate"],
    }
    if gate["warnings"]:
        out["guardrail_warnings"] = list(dict.fromkeys(list(out.get("guardrail_warnings") or []) + gate["warnings"]))
    return out

def _attach_cost_truth_to_scenario(result: Dict[str, Any], trained_bundle: Dict[str, Any], requested_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(result or {})
    source = str(trained_bundle.get("cost_source", "provided" if trained_bundle.get("cost_input_available", False) else "proxy_price_65"))
    out.setdefault("cost_source", source)
    out.setdefault("cost_input_available", bool(trained_bundle.get("cost_input_available", source == "provided")))
    out.setdefault("cost_is_proxy", bool(trained_bundle.get("cost_is_proxy", source == "proxy_price_65")))
    out.setdefault("cost_proxied", bool(out.get("cost_is_proxy", False)))
    out.setdefault("cost_missing", source == "missing")
    out.setdefault("profit_is_reliable", source == "provided")
    out.setdefault("profit_recommendation_allowed", source == "provided")
    if requested_inputs is not None:
        out["scenario_requested_inputs"] = dict(requested_inputs or {})
    if trained_bundle.get("model_quality_gate"):
        out.setdefault("model_quality_gate", trained_bundle.get("model_quality_gate"))
    return out

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
    factor_overrides: Optional[Dict[str, Any]] = None,
    scenario_calc_mode: Optional[str] = None,
    price_guardrail_mode: str = DEFAULT_PRICE_GUARDRAIL_MODE,
) -> Dict[str, Any]:
    price_guardrail_mode = normalize_price_guardrail_mode(price_guardrail_mode)
    if scenario_calc_mode is not None and str(scenario_calc_mode) not in SCENARIO_CALC_MODES:
        raise ValueError(f"Unknown scenario_calc_mode: {scenario_calc_mode}")
    selected_mode = scenario_calc_mode if scenario_calc_mode is not None else trained_bundle.get("analysis_scenario_calc_mode")
    mode = resolve_scenario_calc_mode(selected_mode)
    scenario_audit_params = {
        "manual_price": manual_price,
        "freight_multiplier": freight_multiplier,
        "demand_multiplier": demand_multiplier,
        "horizon_days": horizon_days,
        "discount_multiplier": discount_multiplier,
        "cost_multiplier": cost_multiplier,
        "stock_cap": stock_cap,
        "overrides": overrides or {},
        "factor_overrides": factor_overrides or {},
    }
    if mode == CATBOOST_FULL_FACTOR_MODE:
        result = predict_catboost_full_factor_projection(
            trained_bundle=trained_bundle,
            manual_price=manual_price,
            freight_multiplier=freight_multiplier,
            demand_multiplier=demand_multiplier,
            horizon_days=horizon_days,
            discount_multiplier=discount_multiplier,
            cost_multiplier=cost_multiplier,
            stock_cap=stock_cap,
            overrides=overrides,
            factor_overrides=factor_overrides,
            price_guardrail_mode=price_guardrail_mode,
        )
        return attach_scenario_reproducibility(
            _finalize_scenario_contract(_attach_cost_truth_to_scenario(result, trained_bundle, scenario_audit_params), mode, price_guardrail_mode),
            trained_bundle,
            scenario_audit_params,
            mode,
            price_guardrail_mode,
        )
    if mode == "legacy_current":
        result = _run_what_if_projection_legacy(
            trained_bundle=trained_bundle,
            manual_price=manual_price,
            freight_multiplier=freight_multiplier,
            demand_multiplier=demand_multiplier,
            horizon_days=horizon_days,
            discount_multiplier=discount_multiplier,
            cost_multiplier=cost_multiplier,
            stock_cap=stock_cap,
            overrides=overrides,
            price_guardrail_mode=price_guardrail_mode,
        )
        result["scenario_calc_mode"] = "legacy_current"
        result.setdefault("scenario_engine_version", "v1_legacy_current")
        result.setdefault("effect_source", "scenario_engine_recompute_from_baseline")
        result.setdefault("effect_breakdown", {})
        result.setdefault("daily_effects_summary", [])
        result.setdefault("legacy_or_enhanced_label", "legacy")
        result.setdefault("scenario_calc_mode_label", scenario_mode_label("legacy_current"))
        return attach_scenario_reproducibility(
            _finalize_scenario_contract(_attach_cost_truth_to_scenario(result, trained_bundle, scenario_audit_params), mode, price_guardrail_mode),
            trained_bundle,
            scenario_audit_params,
            mode,
            price_guardrail_mode,
        )
    if mode == "enhanced_local_factors":
        result = _run_what_if_projection_enhanced(
            trained_bundle=trained_bundle,
            manual_price=manual_price,
            freight_multiplier=freight_multiplier,
            demand_multiplier=demand_multiplier,
            horizon_days=horizon_days,
            discount_multiplier=discount_multiplier,
            cost_multiplier=cost_multiplier,
            stock_cap=stock_cap,
            overrides=overrides,
            price_guardrail_mode=price_guardrail_mode,
        )
        return attach_scenario_reproducibility(
            _finalize_scenario_contract(_attach_cost_truth_to_scenario(result, trained_bundle, scenario_audit_params), mode, price_guardrail_mode),
            trained_bundle,
            scenario_audit_params,
            mode,
            price_guardrail_mode,
        )
    raise ValueError(f"Unknown scenario_calc_mode: {scenario_calc_mode}")


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
    scenario["date"] = pd.to_datetime(scenario["date"], errors="coerce").dt.normalize()
    as_is = align_forecasts_by_scenario_dates(as_is, scenario)
    baseline = align_forecasts_by_scenario_dates(baseline, scenario)
    if "lost_sales" not in scenario.columns:
        scenario["lost_sales"] = 0.0
    if "profit" not in scenario.columns:
        scenario["profit"] = 0.0
    if "cost" not in scenario.columns:
        scenario["cost"] = np.nan
    scenario_price_gross_export = (
        scenario["scenario_price_gross"]
        if "scenario_price_gross" in scenario.columns
        else scenario["applied_price_gross"]
        if "applied_price_gross" in scenario.columns
        else scenario["price"]
    )
    scenario_price_net_export = (
        scenario["scenario_price_net"]
        if "scenario_price_net" in scenario.columns
        else scenario["applied_price_net"]
        if "applied_price_net" in scenario.columns
        else scenario["net_unit_price"]
    )
    scenario = scenario.copy()
    scenario["scenario_price_gross_export"] = pd.to_numeric(scenario_price_gross_export, errors="coerce")
    scenario["scenario_price_net_export"] = pd.to_numeric(scenario_price_net_export, errors="coerce")
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
                    "scenario_price_gross_export",
                    "discount",
                    "scenario_price_net_export",
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
                    "scenario_price_gross_export": "scenario_price_gross",
                    "discount": "scenario_discount",
                    "scenario_price_net_export": "scenario_price_net",
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
    for effect_col in ["price_effect", "promo_effect", "freight_effect", "shock_multiplier", "standard_multiplier", "shock_units"]:
        if effect_col in scenario.columns:
            merged[effect_col] = pd.to_numeric(scenario[effect_col], errors="coerce").values
        else:
            merged[effect_col] = 0.0 if effect_col == "shock_units" else 1.0
    baseline_series = pd.to_numeric(merged.get("baseline_demand", pd.Series(np.zeros(len(merged)))), errors="coerce")
    scenario_series = pd.to_numeric(merged.get("scenario_demand", pd.Series(np.zeros(len(merged)))), errors="coerce")
    merged["final_multiplier"] = np.where(
        baseline_series.abs() > 1e-9,
        scenario_series / baseline_series,
        np.nan,
    )
    merged["scenario_calc_mode"] = str(what_if_result.get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
    merged["confidence_label"] = str(what_if_result.get("confidence_label", ""))
    merged["support_label"] = str(what_if_result.get("support_label", ""))
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged["series_id"] = str(result_dict.get("_trained_bundle", {}).get("base_ctx", {}).get("product_id", "unknown"))
    for col in [
        "price_guardrail_mode",
        "requested_price_gross",
        "safe_price_gross",
        "applied_price_gross",
        "applied_price_net",
        "price_out_of_range",
        "price_clipped",
        "guardrail_warning_type",
        "effect_breakdown_available",
        "effect_breakdown_note",
        "model_price_gross",
        "model_price_net",
        "price_for_model",
        "extrapolation_applied",
        "model_boundary_price_gross",
        "extrapolation_from_price_gross",
        "extrapolation_to_price_gross",
        "extrapolation_price_ratio",
        "boundary_model_demand",
        "elasticity_used",
        "elasticity_source",
        "extrapolation_tail_multiplier",
        "scenario_price_effect_source",
        "clip_applied",
    ]:
        merged[col] = scenario[col].values if col in scenario.columns else np.nan
    merged["extrapolation_applied"] = merged["extrapolation_applied"].fillna(False).astype(bool)
    merged["clip_applied"] = merged["clip_applied"].fillna(False).astype(bool)
    merged["extrapolation_tail_multiplier"] = pd.to_numeric(merged["extrapolation_tail_multiplier"], errors="coerce").fillna(1.0)
    merged["scenario_price_effect_source"] = merged["scenario_price_effect_source"].fillna("catboost_full_factor_reprediction")
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
            "baseline_demand",
            "baseline_revenue",
            "baseline_profit",
            "price_effect",
            "promo_effect",
            "freight_effect",
            "shock_multiplier",
            "standard_multiplier",
            "final_multiplier",
            "shock_units",
            "scenario_calc_mode",
            "confidence_label",
            "support_label",
            "scenario_price_gross",
            "scenario_discount",
            "scenario_price_net",
            "scenario_promotion",
            "scenario_freight_value",
            "scenario_cost",
            "lost_sales",
            "price_guardrail_mode",
            "requested_price_gross",
            "safe_price_gross",
            "applied_price_gross",
            "applied_price_net",
            "price_out_of_range",
            "price_clipped",
            "guardrail_warning_type",
            "effect_breakdown_available",
            "effect_breakdown_note",
            "model_price_gross",
            "model_price_net",
            "price_for_model",
            "extrapolation_applied",
            "model_boundary_price_gross",
            "extrapolation_from_price_gross",
            "extrapolation_to_price_gross",
            "extrapolation_price_ratio",
            "boundary_model_demand",
            "elasticity_used",
            "elasticity_source",
            "extrapolation_tail_multiplier",
            "scenario_price_effect_source",
            "clip_applied",
        ]
    ].copy()
    scenario_cost_ratio = pd.to_numeric(manual_daily.get("scenario_cost", np.nan), errors="coerce") / pd.to_numeric(
        manual_daily.get("scenario_price_net", np.nan), errors="coerce"
    ).replace(0, np.nan)
    scenario_freight_ratio = pd.to_numeric(manual_daily.get("scenario_freight_value", np.nan), errors="coerce") / pd.to_numeric(
        manual_daily.get("scenario_price_net", np.nan), errors="coerce"
    ).replace(0, np.nan)
    invalid_unit_econ = (
        (np.isfinite(float(scenario_cost_ratio.median())) and float(scenario_cost_ratio.median()) > 2.0)
        or (np.isfinite(float(scenario_freight_ratio.median())) and float(scenario_freight_ratio.median()) > 1.0)
    )
    warnings_export = list(what_if_result.get("warnings", []))
    if invalid_unit_econ:
        manual_daily["as_is_profit"] = np.nan
        manual_daily["scenario_profit"] = np.nan
        manual_daily["delta_profit"] = np.nan
        warnings_export.append("Себестоимость/логистика выглядят как total-суммы, прибыль отключена до проверки единиц измерения.")
    demand_delta_pct = float((manual_daily["delta_demand"].sum() / max(float(manual_daily["as_is_demand"].sum()), 1e-9)) * 100.0)
    revenue_delta_pct = float((manual_daily["delta_revenue"].sum() / max(float(manual_daily["as_is_revenue"].sum()), 1e-9)) * 100.0)
    profit_delta_pct = safe_signed_pct(float(manual_daily["delta_profit"].sum()), float(manual_daily["as_is_profit"].sum()))
    economic_label, _, _ = classify_economic_verdict(profit_delta_pct, demand_delta_pct, revenue_delta_pct)
    shape_quality_low = bool((result_dict.get("summary", {}) or {}).get("shape_quality_low", False) or (result_dict.get("diagnostics", {}) or {}).get("shape_quality_low", False))
    reliability_label, _, _ = classify_reliability_verdict(
        bool(what_if_result.get("ood_flag", False)),
        list(what_if_result.get("warnings", [])),
        str(what_if_result.get("confidence_label", "")),
        str(what_if_result.get("support_label", "")),
        shape_quality_low,
        bool((what_if_result.get("validation_gate", {}) or {}).get("ok", True)),
    )
    run_summary_cfg = (((result_dict.get("run_summary", {}) or {}).get("config", {}) or {}))
    baseline_forecast_path = str(
        what_if_result.get("baseline_forecast_path")
        or run_summary_cfg.get("baseline_forecast_path")
        or "weekly_ml_baseline"
    )
    scenario_calculation_path = str(
        what_if_result.get("scenario_calculation_path")
        or run_summary_cfg.get("scenario_calculation_path")
        or "enhanced_local_factor_layer"
    )
    learned_uplift_path = str(
        what_if_result.get("learned_uplift_path")
        or run_summary_cfg.get("learned_uplift_path")
        or "inactive_production_diagnostic_only"
    )
    final_user_visible_path = str(
        what_if_result.get("final_user_visible_path")
        or run_summary_cfg.get("final_user_visible_path")
        or f"{baseline_forecast_path} + {scenario_calculation_path}"
    )
    active_path_contract = str(what_if_result.get("active_path_contract") or final_user_visible_path)
    economic_verdict = economic_label
    reliability_verdict = str(what_if_result.get("reliability_verdict", "")).strip() or reliability_label
    learned_factor_confidence = "Не используется" if learned_uplift_path == "inactive_production_diagnostic_only" else "Активен"
    summary = {
        "artifact_scope": "analysis_with_manual_scenario",
        "report_type": "scenario_report",
        "manual_scenario_present": True,
        "manual_scenario_generated": True,
        "scenario_status": what_if_result.get("scenario_status", "as_is"),
        "requested_price": float(what_if_result.get("requested_price", np.nan)),
        "model_price": float(what_if_result.get("model_price", what_if_result.get("price_for_model", np.nan))),
        "applied_discount": float((what_if_result.get("effective_scenario", {}) or {}).get("applied_discount", np.nan)),
        "applied_price_net": float((what_if_result.get("effective_scenario", {}) or {}).get("applied_price_net", np.nan)),
        "modeled_price": float(what_if_result.get("model_price", what_if_result.get("price_for_model", np.nan))),
        "current_price_raw": float(what_if_result.get("current_price_raw", np.nan)),
        "clip_applied": bool(what_if_result.get("clip_applied", what_if_result.get("price_clipped", False))),
        "clip_reason": str(what_if_result.get("clip_reason", "")),
        "price_guardrail_mode": str((what_if_result.get("effective_scenario", {}) or {}).get("price_guardrail_mode", what_if_result.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE))),
        "requested_price_gross": float((what_if_result.get("effective_scenario", {}) or {}).get("requested_price_gross", what_if_result.get("requested_price", np.nan))),
        "safe_price_gross": float((what_if_result.get("effective_scenario", {}) or {}).get("safe_price_gross", what_if_result.get("safe_price_gross", np.nan))),
        "applied_price_gross": float((what_if_result.get("effective_scenario", {}) or {}).get("applied_price_gross", what_if_result.get("model_price", np.nan))),
        "price_out_of_range": bool((what_if_result.get("effective_scenario", {}) or {}).get("price_out_of_range", what_if_result.get("price_out_of_range", False))),
        "price_clipped": bool((what_if_result.get("effective_scenario", {}) or {}).get("price_clipped", what_if_result.get("price_clipped", False))),
        "guardrail_warning_type": str((what_if_result.get("effective_scenario", {}) or {}).get("guardrail_warning_type", what_if_result.get("guardrail_warning_type", ""))),
        "scenario_price_effect_source": str(what_if_result.get("scenario_price_effect_source", "")),
        "extrapolation_applied": bool(what_if_result.get("extrapolation_applied", False)),
        "model_price_gross": float(what_if_result.get("model_price", what_if_result.get("price_for_model", np.nan))),
        "price_for_model": float(what_if_result.get("price_for_model", np.nan)),
        "model_boundary_price_gross": float(what_if_result.get("model_boundary_price_gross", np.nan)),
        "extrapolation_from_price_gross": float(what_if_result.get("extrapolation_from_price_gross", np.nan)),
        "extrapolation_to_price_gross": float(what_if_result.get("extrapolation_to_price_gross", np.nan)),
        "extrapolation_price_ratio": float(what_if_result.get("extrapolation_price_ratio", 1.0)),
        "elasticity_used": float(what_if_result.get("elasticity_used", np.nan)),
        "elasticity_source": str(what_if_result.get("elasticity_source", "")),
        "extrapolation_tail_multiplier": float(what_if_result.get("extrapolation_tail_multiplier", 1.0)),
        "ood_flag": bool(what_if_result.get("ood_flag", False)),
        "horizon_days": int(len(scenario)),
        "scenario_demand_total": float(manual_daily["scenario_demand"].sum()),
        "scenario_revenue_total": float(manual_daily["scenario_revenue"].sum()),
        "scenario_profit_total": float(manual_daily["scenario_profit"].sum()),
        "scenario_vs_as_is_demand_pct": demand_delta_pct,
        "scenario_vs_as_is_revenue_pct": revenue_delta_pct,
        "scenario_vs_as_is_profit_pct": profit_delta_pct,
        "delta_vs_as_is": {
            "demand_total": float(manual_daily["delta_demand"].sum()),
            "revenue_total": float(manual_daily["delta_revenue"].sum()),
            "profit_total": float(manual_daily["delta_profit"].sum()),
        },
        "as_is_demand_total": float(manual_daily["as_is_demand"].sum()),
        "as_is_revenue_total": float(manual_daily["as_is_revenue"].sum()),
        "as_is_profit_total": float(manual_daily["as_is_profit"].sum()),
        "neutral_baseline_demand_total": float(merged["baseline_demand"].sum()),
        "neutral_baseline_revenue_total": float(merged["baseline_revenue"].sum()),
        "neutral_baseline_profit_total": float(merged["baseline_profit"].sum()),
        "delta_vs_neutral_baseline": {
            "demand_total": float(manual_daily["scenario_demand"].sum() - merged["baseline_demand"].sum()),
            "revenue_total": float(manual_daily["scenario_revenue"].sum() - merged["baseline_revenue"].sum()),
            "profit_total": float(manual_daily["scenario_profit"].sum() - merged["baseline_profit"].sum()),
        },
        "uncertainty_penalty": float(what_if_result.get("uncertainty_penalty", 0.0)),
        "confidence": float(what_if_result.get("confidence", np.nan)),
        "scenario_assumptions": what_if_result.get("applied_overrides", {}),
        "scenario_inputs_contract": what_if_result.get("scenario_inputs_contract", {}),
        "price_policy": what_if_result.get("price_policy", {}),
        "factor_policy": what_if_result.get("factor_policy", {}),
        "calculation_trace": what_if_result.get("calculation_trace", {}),
        "guardrail_warnings": what_if_result.get("guardrail_warnings", []),
        "applied_overrides": what_if_result.get("applied_overrides", {}),
        "scenario_forecast": manual_daily.to_dict("records"),
        "baseline_forecast_path": baseline_forecast_path,
        "scenario_calculation_path": scenario_calculation_path,
        "learned_uplift_path": learned_uplift_path,
        "final_user_visible_path": final_user_visible_path,
        "active_path_contract": active_path_contract,
        "final_active_path": final_user_visible_path,
        "model_backend": str(what_if_result.get("model_backend", "unknown")),
        "backend_reason": str(what_if_result.get("backend_reason", "")),
        "learned_uplift_contract": "inactive_production_diagnostic_only",
        "scenario_driver_mode": str(what_if_result.get("scenario_driver_mode", "unknown")),
        "weekly_driver_mode": str(what_if_result.get("weekly_driver_mode", "unknown")),
        "fallback_multiplier_used": bool(what_if_result.get("fallback_multiplier_used", False)),
        "fallback_reason": str(what_if_result.get("fallback_reason", "")),
        "scenario_calc_mode": str(what_if_result.get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
        "scenario_calc_mode_label": str(what_if_result.get("scenario_calc_mode_label", scenario_mode_label(str(what_if_result.get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))))),
        "scenario_engine_version": str(what_if_result.get("scenario_engine_version", "v1_legacy_current")),
        "effect_source": str(what_if_result.get("effect_source", "")),
        "effect_breakdown": what_if_result.get("effect_breakdown", {}),
        "legacy_or_enhanced_label": (
            "catboost_full_factors"
            if str(what_if_result.get("scenario_calc_mode")) == CATBOOST_FULL_FACTOR_MODE
            else (
                "enhanced_local_factors"
                if str(what_if_result.get("scenario_calc_mode")) == "enhanced_local_factors"
                else "legacy_current"
            )
        ),
        "confidence_label": str(what_if_result.get("confidence_label", "")),
        "support_label": str(what_if_result.get("support_label", "")),
        "economic_verdict": economic_verdict,
        "reliability_verdict": reliability_verdict,
        "forecast_confidence": ("Средняя" if shape_quality_low else str(what_if_result.get("confidence_label", "Низкая"))),
        "scenario_math_confidence": "Высокая",
        "learned_factor_confidence": learned_factor_confidence,
        "warnings": warnings_export,
        "segments_summary": {
            "path_segments_used": bool(((what_if_result.get("scenario_support_info", {}) or {})).get("path_segments_used", False)),
            "segment_count": int(((what_if_result.get("scenario_support_info", {}) or {})).get("segment_count", 0)),
            "has_time_varying_path": bool(((what_if_result.get("scenario_support_info", {}) or {})).get("has_time_varying_path", False)),
        },
    }
    return json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"), manual_daily.to_csv(index=False).encode("utf-8")


def build_excel_export_buffer(result_dict: Dict[str, Any], what_if_result: Optional[Dict[str, Any]] = None) -> BytesIO:
    history = result_dict.get("history_daily", pd.DataFrame()).copy()
    baseline = result_dict.get("neutral_baseline_forecast", pd.DataFrame()).copy()
    as_is = result_dict.get("as_is_forecast", pd.DataFrame()).copy()
    scenario = result_dict.get("scenario_forecast")
    if scenario is None and isinstance(what_if_result, dict):
        scenario = what_if_result.get("daily")
    scenario_df = scenario.copy() if isinstance(scenario, pd.DataFrame) else pd.DataFrame()
    holdout_metrics = result_dict.get("holdout_metrics", pd.DataFrame())
    if isinstance(holdout_metrics, dict):
        holdout_metrics = pd.DataFrame([holdout_metrics])
    if holdout_metrics is None or not isinstance(holdout_metrics, pd.DataFrame):
        holdout_metrics = pd.DataFrame()

    def _from_blob_csv(blob: Any) -> pd.DataFrame:
        if blob is None:
            return pd.DataFrame()
        try:
            if isinstance(blob, BytesIO):
                blob.seek(0)
                return pd.read_csv(blob)
            if isinstance(blob, (bytes, bytearray)) and len(blob):
                return pd.read_csv(BytesIO(blob))
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    def _json_to_df(blob: Any) -> pd.DataFrame:
        if blob is None:
            return pd.DataFrame()
        try:
            raw: Any = blob
            if isinstance(blob, BytesIO):
                blob.seek(0)
                raw = blob.read()
            if isinstance(raw, (bytes, bytearray)):
                if not raw:
                    return pd.DataFrame()
                obj = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, str):
                obj = json.loads(raw)
            elif isinstance(raw, dict):
                obj = raw
            else:
                return pd.DataFrame()
            return pd.json_normalize(obj, sep=".")
        except Exception:
            return pd.DataFrame()

    def _flatten_obj(prefix: str, payload: Any) -> List[Tuple[str, Any]]:
        rows: List[Tuple[str, Any]] = []
        if isinstance(payload, pd.DataFrame):
            rows.append((f"{prefix}.__rows__", int(len(payload))))
            rows.append((f"{prefix}.__cols__", ",".join([str(c) for c in payload.columns[:20]])))
            return rows
        if isinstance(payload, pd.Series):
            rows.append((f"{prefix}.__len__", int(len(payload))))
            return rows
        if isinstance(payload, np.ndarray):
            rows.append((f"{prefix}.__shape__", str(payload.shape)))
            return rows
        if isinstance(payload, dict):
            for k, v in payload.items():
                next_prefix = f"{prefix}.{k}" if prefix else str(k)
                rows.extend(_flatten_obj(next_prefix, v))
            return rows
        if isinstance(payload, list):
            if len(payload) > 50:
                rows.append((f"{prefix}.__len__", int(len(payload))))
                return rows
            for i, v in enumerate(payload):
                next_prefix = f"{prefix}[{i}]"
                rows.extend(_flatten_obj(next_prefix, v))
            return rows
        rows.append((prefix, payload))
        return rows

    def _sum(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or len(df) == 0:
            return float("nan")
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

    def non_empty_or_stub(df: pd.DataFrame, sheet_name: str, reason: str) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df
        return pd.DataFrame([{"status": "empty", "sheet": sheet_name, "reason": reason}])

    def _build_executive_summary(
        run_summary_obj: Dict[str, Any],
        run_summary_cfg: Dict[str, Any],
        scenario_output_summary: Dict[str, Any],
        manual_scenario_present: bool,
        scenario_calc_mode_value: str,
        active_path_contract: str,
        warnings_list: List[str],
    ) -> pd.DataFrame:
        holdout_flat = (((run_summary_obj.get("metrics_summary", {}) or {}).get("holdout_flat", {})) or {})
        base_ctx = ((result_dict.get("_trained_bundle", {}) or {}).get("base_ctx", {}) or {})
        history_local = result_dict.get("history_daily", pd.DataFrame())
        train_start = pd.to_datetime(history_local.get("date"), errors="coerce").min() if len(history_local) else pd.NaT
        train_end = pd.to_datetime(history_local.get("date"), errors="coerce").max() if len(history_local) else pd.NaT
        top_warnings = "; ".join([str(w) for w in warnings_list[:5]]) if warnings_list else ""
        return pd.DataFrame(
            [
                {
                    "sku": str(base_ctx.get("product_id", "unknown")),
                    "category": str(base_ctx.get("category", "unknown")),
                    "train_period_start": "" if pd.isna(train_start) else str(pd.Timestamp(train_start).date()),
                    "train_period_end": "" if pd.isna(train_end) else str(pd.Timestamp(train_end).date()),
                    "holdout_period": str(run_summary_cfg.get("holdout_period", "")),
                    "horizon_days": int(len(result_dict.get("neutral_baseline_forecast", pd.DataFrame()))),
                    "wape": float(holdout_flat.get("wape", np.nan)),
                    "mape": float(holdout_flat.get("mape", np.nan)),
                    "mae": float(holdout_flat.get("mae", np.nan)),
                    "rmse": float(holdout_flat.get("rmse", np.nan)),
                    "forecast_shape_warning": bool((scenario_output_summary or {}).get("shape_quality_low", False)),
                    "selected_baseline": str(run_summary_cfg.get("selected_candidate", "legacy_baseline")),
                    "production_selected_candidate": str(run_summary_cfg.get("production_selected_candidate", "legacy_baseline")),
                    "analysis_scenario_calc_mode": scenario_calc_mode_value,
                    "manual_scenario_status": "applied" if manual_scenario_present else "not_applied",
                    "active_scenario_path": active_path_contract,
                    "uplift_status": "diagnostic_only",
                    "top_warnings": top_warnings or ("Сценарий не применён. Это отчёт baseline-анализа." if not manual_scenario_present else ""),
                }
            ]
        )

    def _build_feature_usage_report(feature_report_df: pd.DataFrame, run_summary_cfg: Dict[str, Any]) -> pd.DataFrame:
        if not isinstance(feature_report_df, pd.DataFrame) or feature_report_df.empty:
            return pd.DataFrame([{"feature": "n/a", "group": "Found but not used", "reason_excluded": "feature_report_empty"}])
        df = feature_report_df.copy()
        feature_col = "feature" if "feature" in df.columns else ("factor_name" if "factor_name" in df.columns else ("name" if "name" in df.columns else None))
        if feature_col is None:
            return pd.DataFrame([{"feature": "n/a", "group": "Found but not used", "reason_excluded": "feature_column_missing"}])
        df["feature"] = df[feature_col].astype(str)
        baseline_features = set(run_summary_cfg.get("active_feature_columns", []) or [])
        scenario_only_known = {"price", "discount", "promotion", "freight_value", "demand_multiplier", "shock_units"}
        for col, default in {
            "found_in_raw": True,
            "present_in_daily": False,
            "present_in_weekly": False,
            "used_in_active_baseline": False,
            "used_in_scenario": False,
            "used_in_attempted_uplift": False,
            "used_in_active_uplift": False,
            "reason_excluded": "",
            "active_usage_reason": "",
        }.items():
            if col not in df.columns:
                df[col] = default
        df["used_in_active_baseline"] = (
            df["used_in_active_baseline"]
            | df.get("used_in_weekly_baseline", False)
            | df.get("used_in_weekly_model", False)
            | df.get("used_in_final_active_forecast", False)
        )
        df["used_in_attempted_uplift"] = (
            df["used_in_attempted_uplift"]
            | df.get("used_in_weekly_uplift_attempted", False)
            | df.get("used_in_uplift", False)
        )
        df["used_in_active_uplift"] = df["used_in_active_uplift"] | df.get("used_in_weekly_uplift_active", False)
        df["used_in_active_baseline"] = df["used_in_active_baseline"] | df["feature"].isin(baseline_features)
        df["used_in_scenario"] = df["used_in_scenario"] | df["feature"].isin(scenario_only_known)

        def _group_row(row: pd.Series) -> str:
            if bool(row.get("used_in_active_baseline", False)):
                return "Active baseline ML features"
            if bool(row.get("used_in_scenario", False)):
                return "Scenario-only factors"
            if bool(row.get("used_in_attempted_uplift", False)) or bool(row.get("used_in_active_uplift", False)):
                return "Diagnostic-only factors"
            return "Found but not used"

        df["group"] = df.apply(_group_row, axis=1)
        cols = [
            "feature",
            "group",
            "found_in_raw",
            "present_in_daily",
            "present_in_weekly",
            "used_in_active_baseline",
            "used_in_scenario",
            "used_in_attempted_uplift",
            "used_in_active_uplift",
            "reason_excluded",
            "active_usage_reason",
        ]
        return df[cols].drop_duplicates(subset=["feature", "group"]).sort_values(["group", "feature"]).reset_index(drop=True)

    excel_buffer = BytesIO()
    diagnostics_holdout_predictions = non_empty_or_stub(
        _from_blob_csv(result_dict.get("holdout_predictions_csv")),
        "D_holdout_predictions",
        "source_blob_empty",
    )
    diagnostics_feature_report = non_empty_or_stub(
        _from_blob_csv(result_dict.get("feature_report_csv")),
        "D_feature_report",
        "source_blob_empty",
    )
    diagnostics_baseline_vs_as_is = non_empty_or_stub(
        _from_blob_csv(result_dict.get("analysis_baseline_vs_as_is_csv")),
        "D_baseline_vs_as_is",
        "source_blob_empty",
    )
    diagnostics_uplift_debug = non_empty_or_stub(
        _from_blob_csv(result_dict.get("uplift_debug_report_csv")),
        "D_uplift_debug",
        "source_blob_empty",
    )
    diagnostics_uplift_trace = non_empty_or_stub(
        _from_blob_csv(result_dict.get("uplift_holdout_trace_csv")),
        "D_uplift_trace",
        "source_blob_empty",
    )
    diagnostics_run_summary = _json_to_df(result_dict.get("analysis_run_summary_json"))
    catboost_feature_report_sheet = non_empty_or_stub(
        result_dict.get("catboost_full_factor_report", pd.DataFrame()) if isinstance(result_dict.get("catboost_full_factor_report", pd.DataFrame()), pd.DataFrame) else pd.DataFrame(),
        "D_catboost_feature_report",
        "CatBoost full factor mode was not used",
    )
    catboost_importances_sheet = non_empty_or_stub(
        result_dict.get("catboost_full_factor_importances", pd.DataFrame()) if isinstance(result_dict.get("catboost_full_factor_importances", pd.DataFrame()), pd.DataFrame) else pd.DataFrame(),
        "D_catboost_importances",
        "CatBoost full factor mode was not used",
    )
    catboost_bundle_export = ((result_dict.get("_trained_bundle", {}) or {}).get("catboost_full_factor_bundle", {}) or {})
    catboost_holdout_predictions_sheet = non_empty_or_stub(
        catboost_bundle_export.get("holdout_predictions", pd.DataFrame()) if isinstance(catboost_bundle_export.get("holdout_predictions", pd.DataFrame()), pd.DataFrame) else pd.DataFrame(),
        "D_catboost_holdout_predictions",
        "CatBoost holdout predictions unavailable",
    )
    catboost_factor_catalog_sheet = non_empty_or_stub(
        catboost_bundle_export.get("factor_catalog", pd.DataFrame()) if isinstance(catboost_bundle_export.get("factor_catalog", pd.DataFrame()), pd.DataFrame) else pd.DataFrame(),
        "D_catboost_factor_catalog",
        "CatBoost factor catalog unavailable",
    )
    catboost_guardrails_sheet = non_empty_or_stub(
        pd.json_normalize(catboost_bundle_export.get("guardrails", {}), sep="."),
        "D_catboost_guardrails",
        "CatBoost guardrails unavailable",
    )
    catboost_scenario_overrides_sheet = non_empty_or_stub(
        pd.DataFrame(
            [
                {"feature": k, "value": v}
                for k, v in ((what_if_result or {}).get("applied_overrides", {}) or {}).items()
                if str(k).startswith("factor__")
            ]
        ),
        "D_catboost_scenario_overrides",
        "No CatBoost factor overrides in scenario",
    )
    what_if_applied_overrides_sheet = non_empty_or_stub(
        pd.DataFrame(
            [
                {"feature": str(k), "value": v}
                for k, v in ((what_if_result or {}).get("applied_overrides", {}) or {}).items()
            ]
        ),
        "D_what_if_applied_overrides",
        "No scenario overrides were applied",
    )
    manual_summary_df = _json_to_df(result_dict.get("manual_scenario_summary_json"))
    run_summary_obj = {}
    manual_summary_obj = {}
    try:
        raw_run = result_dict.get("analysis_run_summary_json", b"{}")
        if isinstance(raw_run, (bytes, bytearray)):
            run_summary_obj = json.loads(raw_run.decode("utf-8"))
    except Exception:
        run_summary_obj = {}
    try:
        raw_manual = result_dict.get("manual_scenario_summary_json", b"{}")
        if isinstance(raw_manual, (bytes, bytearray)):
            manual_summary_obj = json.loads(raw_manual.decode("utf-8"))
    except Exception:
        manual_summary_obj = {}

    run_summary_cfg = (run_summary_obj or {}).get("config", {}) if isinstance(run_summary_obj, dict) else {}
    scenario_output_summary = (run_summary_obj or {}).get("scenario_output_summary", {}) if isinstance(run_summary_obj, dict) else {}
    manual_scenario_present = bool(len(scenario_df) > 0)
    report_type = "scenario_report" if manual_scenario_present else "analysis_only"
    artifact_scope = "analysis_with_manual_scenario" if manual_scenario_present else "analysis_only"
    active_path_contract = str(
        ((what_if_result or {}).get("active_path_contract"))
        if manual_scenario_present
        else run_summary_cfg.get("final_active_path", scenario_output_summary.get("active_path_contract", "legacy_baseline+scenario_recompute"))
    )
    scenario_calc_mode_value = str((what_if_result or {}).get("scenario_calc_mode", "unknown")) if manual_scenario_present else "not_applied"
    if manual_summary_df.empty:
        manual_summary_stub = {
            "artifact_scope": "analysis_only",
            "report_type": "analysis_only",
            "scenario_status": "as_is",
            "scenario_reason": "no_manual_scenario_applied",
            "manual_scenario_present": False,
            "manual_scenario_generated": False,
            "scenario_calc_mode": "not_applied",
            "baseline_forecast_path": str(run_summary_cfg.get("baseline_forecast_path", "weekly_ml_baseline")),
            "scenario_calculation_path": str(run_summary_cfg.get("scenario_calculation_path", "scenario_recompute")),
            "learned_uplift_path": str(run_summary_cfg.get("learned_uplift_path", "inactive_production_diagnostic_only")),
            "final_user_visible_path": str(run_summary_cfg.get("final_user_visible_path", "weekly_ml_baseline + scenario_recompute")),
            "final_active_path": str(run_summary_cfg.get("final_active_path", scenario_output_summary.get("active_path_contract", "legacy_baseline+scenario_recompute"))),
        }
        if manual_scenario_present:
            manual_summary_stub.update(
                {
                    "artifact_scope": "analysis_with_manual_scenario",
                    "report_type": "scenario_report",
                    "scenario_status": "computed",
                    "scenario_reason": "",
                    "manual_scenario_present": True,
                    "manual_scenario_generated": True,
                    "scenario_calc_mode": str((what_if_result or {}).get("scenario_calc_mode", "unknown")),
                    "baseline_forecast_path": str((what_if_result or {}).get("baseline_forecast_path", run_summary_cfg.get("baseline_forecast_path", "weekly_ml_baseline"))),
                    "scenario_calculation_path": str((what_if_result or {}).get("scenario_calculation_path", run_summary_cfg.get("scenario_calculation_path", "enhanced_local_factor_layer"))),
                    "learned_uplift_path": str((what_if_result or {}).get("learned_uplift_path", "inactive_production_diagnostic_only")),
                    "final_user_visible_path": str((what_if_result or {}).get("final_user_visible_path", (what_if_result or {}).get("active_path_contract", ""))),
                    "final_active_path": str((what_if_result or {}).get("active_path_contract", "")),
                }
            )
        manual_summary_stub.update(
            {
                "active_path_contract": active_path_contract,
                "learned_uplift_contract": str(
                    scenario_output_summary.get("learned_uplift_contract", "inactive_production_diagnostic_only")
                ),
                "selected_candidate": str(run_summary_cfg.get("selected_candidate", "")),
                "production_selected_candidate": str(run_summary_cfg.get("production_selected_candidate", "")),
                "selection_mode": str(run_summary_cfg.get("selection_mode", "")),
                "production_selection_reason": str(run_summary_cfg.get("production_selection_reason", "")),
                "model_backend": str(run_summary_cfg.get("model_backend", scenario_output_summary.get("model_backend", "unknown"))),
                "backend_reason": str(run_summary_cfg.get("backend_reason", scenario_output_summary.get("backend_reason", ""))),
                "scenario_driver_mode": str(run_summary_cfg.get("scenario_driver_mode", scenario_output_summary.get("scenario_driver_mode", ""))),
                "weekly_driver_mode": str(run_summary_cfg.get("weekly_driver_mode", scenario_output_summary.get("weekly_driver_mode", ""))),
                "uplift_used_in_production": bool(run_summary_cfg.get("uplift_used_in_production", False)),
            }
        )
        manual_summary_df = pd.json_normalize(manual_summary_stub, sep=".")
    else:
        if "report_type" not in manual_summary_df.columns:
            manual_summary_df["report_type"] = report_type
        if "manual_scenario_present" not in manual_summary_df.columns:
            manual_summary_df["manual_scenario_present"] = manual_scenario_present
        if "manual_scenario_generated" not in manual_summary_df.columns:
            manual_summary_df["manual_scenario_generated"] = manual_scenario_present
        manual_summary_df["artifact_scope"] = artifact_scope
        manual_summary_df["report_type"] = report_type
        if not manual_scenario_present:
            manual_summary_df["scenario_calc_mode"] = "not_applied"

    holdout_metrics_sheet = holdout_metrics.copy() if isinstance(holdout_metrics, pd.DataFrame) else pd.DataFrame()
    if holdout_metrics_sheet.empty and isinstance(run_summary_obj, dict):
        holdout_flat = (((run_summary_obj.get("metrics_summary", {}) or {}).get("holdout_flat", {})) or {})
        if isinstance(holdout_flat, dict) and holdout_flat:
            holdout_metrics_sheet = pd.DataFrame([holdout_flat])
    if isinstance(run_summary_obj, dict):
        scenario_summary = (run_summary_obj.get("scenario_output_summary", {}) or {})
        cfg = (run_summary_obj.get("config", {}) or {})
        extended_metrics = {
            "baseline_wape": scenario_summary.get("wape_baseline", np.nan),
            "final_wape": scenario_summary.get("wape_final", np.nan),
            "baseline_mape": scenario_summary.get("mape_baseline", np.nan),
            "final_mape": scenario_summary.get("mape_final", np.nan),
            "baseline_rmse": scenario_summary.get("rmse_baseline", np.nan),
            "final_rmse": scenario_summary.get("rmse_final", np.nan),
            "corr_baseline": scenario_summary.get("corr_baseline", np.nan),
            "corr_final": scenario_summary.get("corr_final", np.nan),
            "std_ratio_baseline": scenario_summary.get("std_ratio_baseline", np.nan),
            "std_ratio_final": scenario_summary.get("std_ratio_final", np.nan),
            "shape_quality_low": scenario_summary.get("shape_quality_low", np.nan),
            "uplift_holdout_keep": scenario_summary.get("uplift_holdout_keep", np.nan),
            "best_naive_wape": cfg.get("best_naive_wape", np.nan),
            "selected_forecaster": cfg.get("selected_forecaster", ""),
        }
        if holdout_metrics_sheet.empty:
            holdout_metrics_sheet = pd.DataFrame([extended_metrics])
        else:
            for key, value in extended_metrics.items():
                if key not in holdout_metrics_sheet.columns:
                    holdout_metrics_sheet[key] = value
    holdout_metrics_sheet = non_empty_or_stub(holdout_metrics_sheet, "D_metrics", "source_blob_empty")

    metrics_rows: List[Dict[str, Any]] = []
    for k, v in _flatten_obj("run_summary", run_summary_obj):
        metrics_rows.append({"metric_group": "run_summary", "metric_name": k, "metric_value": v})
    for k, v in _flatten_obj("manual_summary", manual_summary_obj):
        metrics_rows.append({"metric_group": "manual_summary", "metric_name": k, "metric_value": v})
    for k, v in _flatten_obj("quality_report", result_dict.get("quality_report", {})):
        metrics_rows.append({"metric_group": "quality_report", "metric_name": k, "metric_value": v})
    for k, v in _flatten_obj("holdout_metrics", result_dict.get("holdout_metrics", {})):
        metrics_rows.append({"metric_group": "holdout_metrics", "metric_name": k, "metric_value": v})
    if isinstance(what_if_result, dict):
        for k, v in _flatten_obj("what_if_result", what_if_result):
            metrics_rows.append({"metric_group": "what_if_result", "metric_name": k, "metric_value": v})

    metrics_all = pd.DataFrame(metrics_rows)
    if len(metrics_all):
        metrics_all = metrics_all.drop_duplicates(subset=["metric_name", "metric_value"]).sort_values(["metric_group", "metric_name"])
    metrics_all = non_empty_or_stub(metrics_all, "E_metrics_all", "source_blob_empty")
    diagnostics_run_summary = non_empty_or_stub(diagnostics_run_summary, "D_run_summary", "source_blob_empty")
    manual_summary_df = non_empty_or_stub(manual_summary_df, "C_manual_summary", "manual_scenario_not_applied")

    as_is_demand = _sum(as_is, "actual_sales")
    as_is_revenue = _sum(as_is, "revenue")
    as_is_profit = _sum(as_is, "profit")
    baseline_demand = _sum(baseline, "actual_sales")
    baseline_revenue = _sum(baseline, "revenue")
    baseline_profit = _sum(baseline, "profit")
    scenario_demand = _sum(scenario_df, "actual_sales") if manual_scenario_present else float("nan")
    scenario_revenue = _sum(scenario_df, "revenue") if manual_scenario_present else float("nan")
    scenario_profit = _sum(scenario_df, "profit") if manual_scenario_present else float("nan")

    export_summary = pd.DataFrame(
        [
            {
                "plan": "as_is",
                "present": True,
                "artifact_scope": artifact_scope,
                "scenario_status": "as_is",
                "report_type": report_type,
                "demand_total": as_is_demand,
                "revenue_total": as_is_revenue,
                "profit_total": as_is_profit,
                "delta_demand_vs_as_is": 0.0,
                "delta_revenue_vs_as_is": 0.0,
                "delta_profit_vs_as_is": 0.0,
                "delta_demand_vs_baseline": as_is_demand - baseline_demand,
                "delta_revenue_vs_baseline": as_is_revenue - baseline_revenue,
                "delta_profit_vs_baseline": as_is_profit - baseline_profit,
            },
            {
                "plan": "neutral_baseline",
                "present": True,
                "artifact_scope": artifact_scope,
                "scenario_status": "baseline",
                "report_type": report_type,
                "demand_total": baseline_demand,
                "revenue_total": baseline_revenue,
                "profit_total": baseline_profit,
                "delta_demand_vs_as_is": baseline_demand - as_is_demand,
                "delta_revenue_vs_as_is": baseline_revenue - as_is_revenue,
                "delta_profit_vs_as_is": baseline_profit - as_is_profit,
                "delta_demand_vs_baseline": 0.0,
                "delta_revenue_vs_baseline": 0.0,
                "delta_profit_vs_baseline": 0.0,
            },
            {
                "plan": "manual_scenario",
                "present": manual_scenario_present,
                "artifact_scope": artifact_scope,
                "scenario_status": "computed" if manual_scenario_present else "not_applied",
                "report_type": report_type,
                "demand_total": scenario_demand,
                "revenue_total": scenario_revenue,
                "profit_total": scenario_profit,
                "delta_demand_vs_as_is": scenario_demand - as_is_demand if manual_scenario_present else float("nan"),
                "delta_revenue_vs_as_is": scenario_revenue - as_is_revenue if manual_scenario_present else float("nan"),
                "delta_profit_vs_as_is": scenario_profit - as_is_profit if manual_scenario_present else float("nan"),
                "delta_demand_vs_baseline": scenario_demand - baseline_demand if manual_scenario_present else float("nan"),
                "delta_revenue_vs_baseline": scenario_revenue - baseline_revenue if manual_scenario_present else float("nan"),
                "delta_profit_vs_baseline": scenario_profit - baseline_profit if manual_scenario_present else float("nan"),
            },
        ]
    )
    scenario_warnings = list((what_if_result or {}).get("warnings", [])) if isinstance(what_if_result, dict) else []
    executive_summary = _build_executive_summary(
        run_summary_obj=run_summary_obj if isinstance(run_summary_obj, dict) else {},
        run_summary_cfg=run_summary_cfg,
        scenario_output_summary=scenario_output_summary,
        manual_scenario_present=manual_scenario_present,
        scenario_calc_mode_value=scenario_calc_mode_value,
        active_path_contract=active_path_contract,
        warnings_list=scenario_warnings,
    )
    scenario_summary = pd.DataFrame(
        [
            {
                "scenario_applied": bool(manual_scenario_present),
                "baseline_demand": baseline_demand,
                "scenario_demand": scenario_demand,
                "delta_demand_pct": ((scenario_demand - baseline_demand) / max(abs(baseline_demand), 1e-9) * 100.0) if manual_scenario_present else float("nan"),
                "baseline_revenue": baseline_revenue,
                "scenario_revenue": scenario_revenue,
                "delta_revenue_pct": ((scenario_revenue - baseline_revenue) / max(abs(baseline_revenue), 1e-9) * 100.0) if manual_scenario_present else float("nan"),
                "baseline_profit": baseline_profit,
                "scenario_profit": scenario_profit,
                "delta_profit_pct": ((scenario_profit - baseline_profit) / max(abs(baseline_profit), 1e-9) * 100.0) if manual_scenario_present else float("nan"),
                "avg_price": float(_sum(scenario_df, "price") / max(len(scenario_df), 1)) if manual_scenario_present and len(scenario_df) else float("nan"),
                "avg_discount": float(_sum(scenario_df, "discount") / max(len(scenario_df), 1)) if manual_scenario_present and len(scenario_df) else float("nan"),
                "promo_days": int(pd.to_numeric(scenario_df.get("promotion", pd.Series([])), errors="coerce").fillna(0.0).gt(0).sum()) if manual_scenario_present else 0,
                "avg_freight_value": float((what_if_result or {}).get("applied_path_summary", {}).get("avg_freight", np.nan)) if manual_scenario_present else float("nan"),
                "freight_multiplier": float(((what_if_result or {}).get("effective_scenario", {}) or {}).get("freight_multiplier", np.nan)) if manual_scenario_present else float("nan"),
                "manual_shocks": (
                    int(len(((what_if_result or {}).get("applied_overrides", {}) or {}).get("shocks", [])))
                    + int(abs(float(((what_if_result or {}).get("applied_overrides", {}) or {}).get("manual_shock_multiplier", 1.0)) - 1.0) > 1e-9)
                )
                if manual_scenario_present
                else 0,
                "note": "" if manual_scenario_present else "Сценарий не применён. Это только анализ baseline.",
            }
        ]
    )
    scenario_user_summary = pd.DataFrame(
        [
            {
                "Сценарий применён": bool(manual_scenario_present),
                "Текущий план: спрос": as_is_demand,
                "Сценарий: спрос": scenario_demand,
                "Изменение спроса, %": ((scenario_demand - as_is_demand) / max(abs(as_is_demand), 1e-9) * 100.0) if manual_scenario_present else float("nan"),
                "Текущий план: выручка": as_is_revenue,
                "Сценарий: выручка": scenario_revenue,
                "Изменение выручки, %": ((scenario_revenue - as_is_revenue) / max(abs(as_is_revenue), 1e-9) * 100.0) if manual_scenario_present else float("nan"),
                "Текущий план: прибыль": as_is_profit,
                "Сценарий: прибыль": scenario_profit,
            "Изменение прибыли, %": safe_signed_pct((scenario_profit - as_is_profit), as_is_profit) if manual_scenario_present else float("nan"),
                "Вывод": (
                    "Сценарий не применён. Это только анализ текущего плана."
                    if not manual_scenario_present
                    else (
                        "Сценарий улучшает спрос и прибыль относительно текущего плана."
                        if scenario_profit >= as_is_profit and scenario_demand >= as_is_demand
                        else (
                            "Сценарий может быть экономически лучше текущего плана, но проверьте модельную оценку спроса."
                            if scenario_profit >= as_is_profit
                            else (
                                "Сценарий увеличивает спрос, но снижает прибыль. Проверьте скидку, цену и затраты."
                                if scenario_demand >= as_is_demand
                                else "Сценарий рассчитан, но ухудшает спрос, выручку и прибыль относительно текущего плана."
                            )
                        )
                    )
                ),
                "Рекомендация": build_user_recommendation(
                    manual_scenario_present,
                    float(scenario_profit - as_is_profit) if manual_scenario_present else 0.0,
                    len(scenario_warnings),
                    str((what_if_result or {}).get("confidence_label", "Низкая")),
                ),
            }
        ]
    )
    feature_usage = _build_feature_usage_report(diagnostics_feature_report, run_summary_cfg)
    effect_breakdown_df = pd.DataFrame([dict((what_if_result or {}).get("effect_breakdown", {}))]) if manual_scenario_present else pd.DataFrame(
        [{"status": "not_applied", "reason": "manual_scenario_not_applied"}]
    )
    daily_effects_df = pd.DataFrame((what_if_result or {}).get("daily_effects_summary", [])) if manual_scenario_present else pd.DataFrame(
        [{"status": "not_applied", "reason": "manual_scenario_not_applied"}]
    )

    recommended_mode_status_obj = dict(result_dict.get("recommended_mode_status", {}) or {})
    recommendation_gate_obj = dict((what_if_result or {}).get("recommendation_gate_details", {}) or {}) if isinstance(what_if_result, dict) else {}
    decision_gate_df = pd.DataFrame([
        {
            "main_status": str((what_if_result or {}).get("recommendation_gate", recommendation_gate_obj.get("decision_status", "not_available"))) if isinstance(what_if_result, dict) else "not_available",
            "calculation_gate": str((what_if_result or {}).get("calculation_gate", recommendation_gate_obj.get("calculation_gate", "not_available"))) if isinstance(what_if_result, dict) else "not_available",
            "can_show_forecast": bool((recommendation_gate_obj.get("usage_policy", {}) or {}).get("can_show_forecast", True)),
            "can_show_what_if": bool((recommendation_gate_obj.get("usage_policy", {}) or {}).get("can_show_what_if", True)),
            "can_recommend_action": bool((recommendation_gate_obj.get("usage_policy", {}) or {}).get("can_recommend_action", False)),
            "top_reasons": "; ".join([str(x) for x in (what_if_result or {}).get("recommendation_gate_reasons", recommendation_gate_obj.get("reasons", []))[:3]]) if isinstance(what_if_result, dict) else "",
            "warnings": "; ".join([str(x) for x in (what_if_result or {}).get("recommendation_gate_warnings", recommendation_gate_obj.get("warnings", []))[:5]]) if isinstance(what_if_result, dict) else "",
            "recommended_mode_status": str(recommended_mode_status_obj.get("status", "")),
            "recommended_mode_reason": str(recommended_mode_status_obj.get("reason", "")),
            "important_factor_ood": bool((what_if_result or {}).get("important_factor_ood", False)) if isinstance(what_if_result, dict) else False,
            "factor_ood_flag": bool((what_if_result or {}).get("factor_ood_flag", (what_if_result or {}).get("ood_flag", False))) if isinstance(what_if_result, dict) else False,
            "monotonicity_label": "Проверка экономической правдоподобности, не абсолютный бизнес-закон",
        }
    ])
    scenario_audit_df = pd.json_normalize((what_if_result or {}).get("scenario_reproducibility", {}), sep=".") if isinstance(what_if_result, dict) else pd.DataFrame()
    scenario_audit_df = non_empty_or_stub(scenario_audit_df, "Scenario Audit", "scenario_reproducibility_unavailable")
    model_quality_source = dict((what_if_result or {}).get("scenario_engine_meta", {}).get("holdout_metrics", {}) or {}) if isinstance(what_if_result, dict) else {}
    model_quality_source.update({f"recommended_mode.{k}": v for k, v in recommended_mode_status_obj.items()})
    model_quality_df = pd.json_normalize(model_quality_source, sep=".") if model_quality_source else holdout_metrics_sheet.copy()
    model_quality_df = non_empty_or_stub(model_quality_df, "Model Quality", "model_quality_unavailable")
    limitations_rows = [
        {"limitation": "predicted_demand_raw = модельный спрос до складского ограничения; это не доказанный истинный спрос."},
        {"limitation": "realized_sales_after_stock = ожидаемые продажи после складского ограничения."},
        {"limitation": "Модель обучена на наблюдаемых продажах; при stockout истинный спрос мог быть выше."},
    ]
    if recommended_mode_status_obj.get("reason"):
        limitations_rows.append({"limitation": str(recommended_mode_status_obj.get("reason"))})
    if isinstance(what_if_result, dict):
        limitations_rows.extend({"limitation": str(w)} for w in list(what_if_result.get("warnings", []))[:20])
    limitations_df = pd.DataFrame(limitations_rows).drop_duplicates()



    run_metadata_df = pd.DataFrame([{"path": k, "value": v} for k, v in _flatten_obj("run_metadata", result_dict.get("run_metadata", {}))])
    run_metadata_df = non_empty_or_stub(run_metadata_df, "Run Metadata", "run_metadata_unavailable")

    audit_contract_source = result_dict.get("data_contract") or ((result_dict.get("_trained_bundle", {}) or {}).get("data_contract", {}))
    data_contract_df = pd.DataFrame([{"path": k, "value": v} for k, v in _flatten_obj("data_contract", audit_contract_source)])
    data_contract_df = non_empty_or_stub(data_contract_df, "Data Contract", "data_contract_unavailable")
    scenario_contract_source = (what_if_result or {}).get("scenario_contract", what_if_result or {}) if isinstance(what_if_result, dict) else {}
    scenario_contract_df = pd.DataFrame([{"path": k, "value": v} for k, v in _flatten_obj("scenario_contract", scenario_contract_source)])
    scenario_contract_df = non_empty_or_stub(scenario_contract_df, "Scenario Contract", "scenario_contract_unavailable")
    passport_source = result_dict.get("decision_passport") or (what_if_result or {}).get("decision_passport", {}) if isinstance(what_if_result, dict) else result_dict.get("decision_passport", {})
    if not passport_source:
        passport_source = {
            "decision_status": (what_if_result or {}).get("recommendation_gate", "not_recommended") if isinstance(what_if_result, dict) else "not_recommended",
            "test_plan": {"duration_days": 14, "primary_metric": "profit", "rollback_condition": "profit_delta < -2% or demand_delta < -10%"},
            "rollback_plan": {"rollback_action": "return_to_previous_price_or_factor_values", "monitoring_frequency": "daily"},
            "assumptions": ["Decision passport generated from export audit trail."],
        }
    decision_passport_df = pd.DataFrame([{"path": k, "value": v} for k, v in _flatten_obj("decision_passport", passport_source)])
    decision_passport_df = non_empty_or_stub(decision_passport_df, "Decision Passport", "decision_passport_unavailable")
    audit_trail_df = pd.DataFrame([
        {"audit_block": "Data Contract", "present": not data_contract_df.empty},
        {"audit_block": "Field Sources", "present": True},
        {"audit_block": "Unit Economics Check", "present": True},
        {"audit_block": "Model Quality Gate", "present": True},
        {"audit_block": "Scenario Contract", "present": not scenario_contract_df.empty},
        {"audit_block": "Decision Ranking", "present": True},
        {"audit_block": "Decision Passport", "present": not decision_passport_df.empty},
        {"audit_block": "Rejected Options", "present": True},
        {"audit_block": "Предупреждения", "present": True},
        {"audit_block": "Blockers", "present": True},
        {"audit_block": "Test Plan", "present": True},
        {"audit_block": "Rollback Plan", "present": True},
    ])

    sheet_meta: List[Dict[str, Any]] = [
        {"group": "A_CORE_INPUTS", "sheet": "A_history", "description": "История продаж и факторов.", "df": history, "optional": False, "note_absent": ""},
        {"group": "B_FORECASTS", "sheet": "B_as_is", "description": "Текущий план (без изменений).", "df": as_is, "optional": False, "note_absent": ""},
        {"group": "B_FORECASTS", "sheet": "B_neutral_baseline", "description": "Нейтральный baseline.", "df": baseline, "optional": False, "note_absent": ""},
        {"group": "B_FORECASTS", "sheet": "B_manual_scenario", "description": "Ручной сценарий (если применён).", "df": scenario_df, "optional": True, "note_absent": "Manual scenario was not applied"},
        {"group": "C_SUMMARY", "sheet": "C_export_summary", "description": "Итоги и дельты.", "df": export_summary, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "C_manual_summary", "description": "Контракт и метаданные сценария.", "df": manual_summary_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Executive Summary", "description": "Пользовательская сводка v1.", "df": executive_summary, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Scenario Summary", "description": "Сводка baseline vs scenario.", "df": scenario_summary, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "User Scenario Summary", "description": "Пользовательская сводка: текущий план vs сценарий.", "df": scenario_user_summary, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "scenario_vs_baseline_summary", "description": "Delta сценария к baseline.", "df": scenario_summary, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "effect_breakdown", "description": "Разложение эффектов сценария.", "df": effect_breakdown_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "daily_effects_summary", "description": "Дневные эффекты сценария.", "df": daily_effects_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Feature Usage", "description": "Что реально используется в baseline/scenario/diagnostics.", "df": feature_usage, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Decision Gate", "description": "Главный статус: можно рекомендовать, только тест, экспериментально или не рекомендовать.", "df": decision_gate_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Scenario Audit", "description": "Воспроизводимость сценария: hash, run id и параметры расчёта.", "df": scenario_audit_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Model Quality", "description": "Качество модели и статус рекомендуемого режима.", "df": model_quality_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Limitations", "description": "Ограничения интерпретации спроса, stockout, OOD и gate.", "df": limitations_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Data Contract", "description": "Truth contract: field sources, proxies, target semantics.", "df": data_contract_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Scenario Contract", "description": "Calculation contract for scenario result.", "df": scenario_contract_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Decision Passport", "description": "Final decision artifact with assumptions, risks, test and rollback plan.", "df": decision_passport_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Audit Trail", "description": "Presence of required production audit blocks.", "df": audit_trail_df, "optional": False, "note_absent": ""},
        {"group": "C_SUMMARY", "sheet": "Run Metadata", "description": "Production audit metadata: run id, dataset hash, versions, warnings and blockers.", "df": run_metadata_df, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_metrics", "description": "Метрики holdout.", "df": holdout_metrics_sheet, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_holdout_predictions", "description": "Дневные предсказания holdout.", "df": diagnostics_holdout_predictions, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_feature_report", "description": "Использование признаков.", "df": diagnostics_feature_report, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_baseline_vs_as_is", "description": "Сравнение baseline и as-is.", "df": diagnostics_baseline_vs_as_is, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_uplift_debug", "description": "Диагностика uplift.", "df": diagnostics_uplift_debug, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_uplift_trace", "description": "Трассировка uplift.", "df": diagnostics_uplift_trace, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_run_summary", "description": "Сводный JSON прогона (flattened).", "df": diagnostics_run_summary, "optional": False, "note_absent": ""},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_feature_report", "description": "Факторы CatBoost full factor mode.", "df": catboost_feature_report_sheet, "optional": True, "note_absent": "CatBoost full factor mode was not used"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_importances", "description": "Feature importance CatBoost full factor mode.", "df": catboost_importances_sheet, "optional": True, "note_absent": "CatBoost full factor mode was not used"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_feature_importance", "description": "Deprecated alias: Feature importance CatBoost full factor mode.", "df": catboost_importances_sheet, "optional": True, "note_absent": "CatBoost full factor mode was not used"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_holdout_predictions", "description": "Daily holdout predictions CatBoost.", "df": catboost_holdout_predictions_sheet, "optional": True, "note_absent": "CatBoost holdout predictions unavailable"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_holdout", "description": "Deprecated alias: Daily holdout predictions CatBoost.", "df": catboost_holdout_predictions_sheet, "optional": True, "note_absent": "CatBoost holdout predictions unavailable"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_factor_catalog", "description": "Каталог факторов CatBoost.", "df": catboost_factor_catalog_sheet, "optional": True, "note_absent": "CatBoost factor catalog unavailable"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_guardrails", "description": "Guardrails CatBoost факторов.", "df": catboost_guardrails_sheet, "optional": True, "note_absent": "CatBoost guardrails unavailable"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_catboost_scenario_overrides", "description": "Deprecated alias: переопределения factor__* в сценарии.", "df": catboost_scenario_overrides_sheet, "optional": True, "note_absent": "No CatBoost factor overrides in scenario"},
        {"group": "D_DIAGNOSTICS", "sheet": "D_what_if_applied_overrides", "description": "Все применённые overrides сценария.", "df": what_if_applied_overrides_sheet, "optional": True, "note_absent": "No scenario overrides were applied"},
        {"group": "E_METRICS", "sheet": "E_metrics_all", "description": "Все доступные метрики (flattened).", "df": metrics_all, "optional": False, "note_absent": ""},
    ]

    readme_rows: List[Dict[str, Any]] = []
    for item in sheet_meta:
        df = item["df"] if isinstance(item.get("df"), pd.DataFrame) else pd.DataFrame()
        present = bool(len(df) > 0) if item.get("optional", False) else True
        rows_count = int(len(df)) if present else 0
        status = "ok"
        note = ""
        if not present:
            status = "not_generated"
            note = item.get("note_absent", "")
        elif {"status", "reason", "sheet"}.issubset(set(df.columns)):
            status = "stub"
            note = str(df.iloc[0].get("reason", ""))
        readme_rows.append(
            {
                "sheet": item["sheet"],
                "group": item["group"],
                "description": item["description"],
                "present": present,
                "rows": rows_count,
                "status": status,
                "note": note,
                "report_type": report_type,
            }
        )
    report_index = pd.DataFrame(readme_rows)

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        report_index.to_excel(writer, sheet_name="README", index=False)
        history.to_excel(writer, sheet_name="A_history", index=False)
        as_is.to_excel(writer, sheet_name="B_as_is", index=False)
        baseline.to_excel(writer, sheet_name="B_neutral_baseline", index=False)
        holdout_metrics_sheet.to_excel(writer, sheet_name="D_metrics", index=False)
        export_summary.to_excel(writer, sheet_name="C_export_summary", index=False)
        manual_summary_df.to_excel(writer, sheet_name="C_manual_summary", index=False)
        executive_summary.to_excel(writer, sheet_name="Executive Summary", index=False)
        scenario_summary.to_excel(writer, sheet_name="Scenario Summary", index=False)
        scenario_user_summary.to_excel(writer, sheet_name="User Scenario Summary", index=False)
        scenario_summary.to_excel(writer, sheet_name="scenario_vs_baseline_summary", index=False)
        effect_breakdown_df.to_excel(writer, sheet_name="effect_breakdown", index=False)
        daily_effects_df.to_excel(writer, sheet_name="daily_effects_summary", index=False)
        feature_usage.to_excel(writer, sheet_name="Feature Usage", index=False)
        decision_gate_df.to_excel(writer, sheet_name="Decision Gate", index=False)
        scenario_audit_df.to_excel(writer, sheet_name="Scenario Audit", index=False)
        model_quality_df.to_excel(writer, sheet_name="Model Quality", index=False)
        limitations_df.to_excel(writer, sheet_name="Limitations", index=False)
        data_contract_df.to_excel(writer, sheet_name="Data Contract", index=False)
        scenario_contract_df.to_excel(writer, sheet_name="Scenario Contract", index=False)
        decision_passport_df.to_excel(writer, sheet_name="Decision Passport", index=False)
        audit_trail_df.to_excel(writer, sheet_name="Audit Trail", index=False)
        run_metadata_df.to_excel(writer, sheet_name="Run Metadata", index=False)
        diagnostics_holdout_predictions.to_excel(writer, sheet_name="D_holdout_predictions", index=False)
        diagnostics_feature_report.to_excel(writer, sheet_name="D_feature_report", index=False)
        diagnostics_baseline_vs_as_is.to_excel(writer, sheet_name="D_baseline_vs_as_is", index=False)
        diagnostics_uplift_debug.to_excel(writer, sheet_name="D_uplift_debug", index=False)
        diagnostics_uplift_trace.to_excel(writer, sheet_name="D_uplift_trace", index=False)
        diagnostics_run_summary.to_excel(writer, sheet_name="D_run_summary", index=False)
        catboost_feature_report_sheet.to_excel(writer, sheet_name="D_catboost_feature_report", index=False)
        catboost_importances_sheet.to_excel(writer, sheet_name="D_catboost_importances", index=False)
        catboost_importances_sheet.to_excel(writer, sheet_name="D_catboost_feature_importance", index=False)
        catboost_holdout_predictions_sheet.to_excel(writer, sheet_name="D_catboost_holdout_predictions", index=False)
        catboost_holdout_predictions_sheet.to_excel(writer, sheet_name="D_catboost_holdout", index=False)
        catboost_factor_catalog_sheet.to_excel(writer, sheet_name="D_catboost_factor_catalog", index=False)
        catboost_guardrails_sheet.to_excel(writer, sheet_name="D_catboost_guardrails", index=False)
        catboost_scenario_overrides_sheet.to_excel(writer, sheet_name="D_catboost_scenario_overrides", index=False)
        what_if_applied_overrides_sheet.to_excel(writer, sheet_name="D_what_if_applied_overrides", index=False)
        metrics_all.to_excel(writer, sheet_name="E_metrics_all", index=False)
        if manual_scenario_present:
            scenario_df.to_excel(writer, sheet_name="B_manual_scenario", index=False)
    excel_buffer.seek(0)
    return excel_buffer


def refresh_excel_export(result_dict: Dict[str, Any], what_if_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result_dict = _attach_run_metadata(result_dict)
    result_dict["excel_buffer"] = build_excel_export_buffer(result_dict, what_if_result)
    return result_dict


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
        "scenario_mode": str((what_if_result or {}).get("scenario_calc_mode_label", scenario_mode_label((what_if_result or {}).get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)))),
        "calculation_contract": str((what_if_result or {}).get("active_path_contract_label", scenario_contract_label((what_if_result or {}).get("active_path_contract", active_path)))),
        "factor_source": str((what_if_result or {}).get("effect_source_label", effect_source_label((what_if_result or {}).get("effect_source", "")))),
        "support_label": str((what_if_result or {}).get("support_label", "unknown")),
        "model_quality": (what_if_result or {}).get("scenario_engine_meta", {}).get("holdout_metrics", {}),
        "rolling_backtest": (what_if_result or {}).get("scenario_engine_meta", {}).get("rolling_backtest", {}),
        "price_in_historical_range": bool(scenario_range_status.get("price") == "ok"),
        "factors_in_historical_range": bool(in_range and not (what_if_result or {}).get("ood_flag", False)),
        "cost_policy": "estimated_proxy" if bool((what_if_result or {}).get("cost_proxied", (what_if_result or {}).get("cost_is_proxy", False))) else "provided_or_not_used",
        "effect_bigger_than_model_error": "effect_smaller_than_model_error" not in (what_if_result or {}).get("recommendation_gate_reasons", []),
        "monotonicity_policy": (what_if_result or {}).get("monotonicity_policy", {}),
        "recommended_mode_status": results.get("recommended_mode_status", {}),
        "scenario_reproducibility": (what_if_result or {}).get("scenario_reproducibility", {}),
        "fallback_used": bool((what_if_result or {}).get("fallback_multiplier_used", False)),
        "limitations": list(dict.fromkeys([
            "Модель обучена на наблюдаемых продажах; при stockout это может быть ниже истинного спроса.",
            *warnings_local,
            *list((what_if_result or {}).get("warnings", [])),
        ])),
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
                "active_scenario_calc_mode": str((what_if_result or {}).get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                "active_scenario_calc_mode_label": str((what_if_result or {}).get("scenario_calc_mode_label", scenario_mode_label((what_if_result or {}).get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)))),
                "active_path_contract_label": str((what_if_result or {}).get("active_path_contract_label", scenario_contract_label((what_if_result or {}).get("active_path_contract", "")))),
                "effect_source_label": str((what_if_result or {}).get("effect_source_label", effect_source_label((what_if_result or {}).get("effect_source", "")))),
                "support_label": str((what_if_result or {}).get("support_label", "")),
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
        "trust_block": trust,
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
    deltas = calculate_scenario_deltas(as_is_forecast, scenario_forecast)
    as_is_units = float(deltas["base_units"])
    as_is_revenue = float(deltas["base_revenue"])
    as_is_profit = float(deltas["base_profit"])
    scenario_units = float(deltas["scenario_units"])
    scenario_revenue = float(deltas["scenario_revenue"])
    scenario_profit = float(deltas["scenario_profit"])

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
        "scenario_calc_mode": str((what_if_result or {}).get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
        "scenario_calc_mode_label": str((what_if_result or {}).get("scenario_calc_mode_label", scenario_mode_label((what_if_result or {}).get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)))),
        "active_path_contract": str((what_if_result or {}).get("active_path_contract", "")),
        "legacy_or_enhanced_label": str((what_if_result or {}).get("legacy_or_enhanced_label", "legacy")),
        "effect_source": str((what_if_result or {}).get("effect_source", "")),
        "confidence": float((what_if_result or {}).get("confidence", np.nan)),
        "support_label": str((what_if_result or {}).get("support_label", "")),
        "effect_breakdown": (what_if_result or {}).get("effect_breakdown", {}),
        "warnings": list((what_if_result or {}).get("warnings", [])),
        "segments_summary": {
            "path_segments_used": bool(((what_if_result or {}).get("scenario_support_info", {}) or {}).get("path_segments_used", False)),
            "segment_count": int(((what_if_result or {}).get("scenario_support_info", {}) or {}).get("segment_count", 0)),
            "has_time_varying_path": bool(((what_if_result or {}).get("scenario_support_info", {}) or {}).get("has_time_varying_path", False)),
        },
    }


def collect_current_form_values(
    manual_price: float,
    discount: float,
    promo_value: float,
    freight_mult: float,
    demand_mult: float,
    hdays: int,
    scenario_calc_mode: str = DEFAULT_SCENARIO_CALC_MODE,
    price_guardrail_mode: str = DEFAULT_PRICE_GUARDRAIL_MODE,
) -> Dict[str, Any]:
    return {
        "manual_price": float(manual_price),
        "discount": float(discount),
        "promo_value": float(promo_value),
        "freight_mult": float(freight_mult),
        "demand_mult": float(demand_mult),
        "hdays": int(hdays),
        "scenario_calc_mode": str(scenario_calc_mode),
        "price_guardrail_mode": normalize_price_guardrail_mode(price_guardrail_mode),
    }


def build_applied_scenario_snapshot(
    wr: Dict[str, Any],
    manual_price: float,
    discount: float,
    promo_value: float,
    freight_mult: float,
    demand_mult: float,
    hdays: int,
    scenario_calc_mode: str = DEFAULT_SCENARIO_CALC_MODE,
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
        "price_guardrail_mode": str(effective.get("price_guardrail_mode", wr.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE))),
        "requested_price_gross": float(effective.get("requested_price_gross", wr.get("requested_price", manual_price))),
        "safe_price_gross": float(effective.get("safe_price_gross", wr.get("safe_price_gross", np.nan))),
        "price_out_of_range": bool(effective.get("price_out_of_range", wr.get("price_out_of_range", False))),
        "freight_mult": float(freight_mult),
        "demand_mult": float(demand_mult),
        "horizon_days": int(hdays),
        "scenario_status": str(wr.get("scenario_status", "as_is")),
        "scenario_calc_mode": str(wr.get("scenario_calc_mode", scenario_calc_mode)),
        "scenario_calc_mode_label": str(wr.get("scenario_calc_mode_label", scenario_mode_label(str(wr.get("scenario_calc_mode", scenario_calc_mode))))),
        "active_path_contract": str(wr.get("active_path_contract", "")),
        "active_path_contract_label": str(wr.get("active_path_contract_label", scenario_contract_label(str(wr.get("active_path_contract", ""))))),
        "confidence_label": str(wr.get("confidence_label", "")),
        "support_label": str(wr.get("support_label", "")),
        "warnings": list(wr.get("warnings", []))[:5],
        "shock_count": int(len((wr.get("applied_overrides", {}) or {}).get("shocks", []))) if isinstance(wr.get("applied_overrides", {}), dict) else 0,
        "effect_breakdown": dict(wr.get("effect_breakdown", {})),
        "applied_path_summary": dict(wr.get("applied_path_summary", {})),
    }


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return True


def fmt_money_total(value: Any) -> str:
    if _is_nan(value):
        return "—"
    return f"₽ {float(value):,.0f}".replace(",", " ")


def fmt_price(value: Any) -> str:
    if _is_nan(value):
        return "—"
    return f"₽ {float(value):,.2f}".replace(",", " ")


def fmt_units(value: Any) -> str:
    if _is_nan(value):
        return "—"
    return f"{float(value):,.1f} шт.".replace(",", " ")


def fmt_pct_abs(value: Any) -> str:
    if _is_nan(value):
        return "—"
    return f"{float(value):.1f}%"


def fmt_pct_delta(value: Any) -> str:
    if _is_nan(value):
        return "—"
    v = float(value)
    if abs(v) < 1e-12:
        return "0.0%"
    return f"{v:+.1f}%"



def _finite_or_default(value: Any, default: float) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _fmt_decision_context_value(value: Any, suffix: str = "", decimals: int = 2) -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{decimals}f}{suffix}"
    except Exception:
        return "—"


def build_decision_current_context(results: Dict[str, Any], context_source: str) -> Dict[str, Any]:
    bundle = results.get("_trained_bundle", {}) or {}
    base_ctx = dict(bundle.get("base_ctx", {}) or {})
    base_price = _finite_or_default(base_ctx.get("price"), _finite_or_default(results.get("current_price"), 0.0))
    base_discount = _finite_or_default(base_ctx.get("discount"), 0.0)
    base_promotion = _finite_or_default(base_ctx.get("promotion"), 0.0)
    base_freight = _finite_or_default(base_ctx.get("freight_value"), 0.0)
    base_cost = _finite_or_default(base_ctx.get("cost"), base_price * CONFIG["COST_PROXY_RATIO"])

    if context_source != "applied_scenario":
        return {
            "price": base_price,
            "discount": base_discount,
            "promotion": base_promotion,
            "freight_value": base_freight,
            "cost": base_cost,
            "factor_overrides": {},
            "context_source": "base_ctx",
        }

    snap = st.session_state.get("applied_scenario_snapshot") or {}
    wr = st.session_state.get("what_if_result") or {}
    effective = (wr.get("effective_scenario") or {}) if isinstance(wr, dict) else {}

    price = _finite_or_default(
        snap.get("manual_price_applied"),
        _finite_or_default(
            wr.get("requested_price"),
            _finite_or_default(
                wr.get("applied_price_gross"),
                _finite_or_default(effective.get("applied_price_gross"), base_price),
            ),
        ),
    )
    discount = _finite_or_default(snap.get("discount_applied"), base_discount)
    promotion = _finite_or_default(snap.get("promo_applied"), base_promotion)
    freight_mult = _finite_or_default(snap.get("freight_mult", snap.get("freight_requested_multiplier")), 1.0)
    freight_value = _finite_or_default(snap.get("freight_applied"), base_freight * freight_mult)

    return {
        "price": price,
        "discount": discount,
        "promotion": promotion,
        "freight_value": freight_value,
        "cost": _finite_or_default(base_ctx.get("cost"), price * CONFIG["COST_PROXY_RATIO"]),
        "factor_overrides": dict(st.session_state.get("last_factor_overrides", {}) or {}),
        "context_source": "applied_scenario",
    }


def decision_status_label(status: str) -> str:
    mapping = {
        "recommended": "Можно рассмотреть через пилот",
        "approve": "Можно рассмотреть через пилот",
        "approved": "Можно рассмотреть через пилот",
        "test_recommended": "Можно рассмотреть через пилот",
        "test_only": "Только через пилот",
        "experimental": "Экспериментальная гипотеза",
        "experimental_only": "Только экспериментальная гипотеза",
        "needs_validation": "Требует проверки",
        "not_recommended": "Не рекомендуется",
        "blocked": "Заблокировано ограничениями",
    }
    return mapping.get(str(status), str(status))


def decision_status_tone(status: str) -> str:
    s = str(status or "").lower()
    if s in {"recommended", "approve", "approved", "test_recommended"}:
        return "success"
    if s in {"test_only", "experimental", "experimental_only", "needs_validation", "warning"}:
        return "warning"
    return "danger"


def action_type_label(action_type: str) -> str:
    mapping = {
        "price_change": "Изменить цену",
        "discount_change": "Изменить скидку",
        "promotion_change": "Изменить промо",
        "freight_change": "Изменить логистику",
        "demand_shock": "Внешний спрос",
    }
    return mapping.get(str(action_type), str(action_type).replace("_", " "))


def objective_label(objective: str) -> str:
    mapping = {"profit": "Прибыль", "revenue": "Выручка", "demand": "Спрос", "risk_reduction": "Снижение риска"}
    return mapping.get(str(objective), str(objective))


def risk_level_label(risk: str) -> str:
    mapping = {"low": "Низкий", "medium": "Средний", "high": "Высокий", "critical": "Критический", "n/a": "—"}
    return mapping.get(str(risk).lower(), str(risk))


def test_scope_label(scope: str) -> str:
    mapping = {"controlled_test": "Контролируемый пилот", "ab_test": "A/B-тест", "pilot": "Пилот", "manual_review": "Ручная проверка"}
    return mapping.get(str(scope), str(scope))


def metric_label(metric: str) -> str:
    mapping = {"gross_profit": "Валовая прибыль", "profit": "Прибыль", "revenue": "Выручка", "demand": "Спрос", "sales": "Продажи", "margin": "Маржа"}
    return mapping.get(str(metric), str(metric))


def build_user_friendly_decision_candidates_table(table: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(table or [], start=1):
        action_type = row.get("action_type")
        eff = row.get("expected_effect", {}) or {}
        rel = row.get("full_reliability") or row.get("reliability") or {}
        rows.append({
            "Вариант": f"Вариант {idx}",
            "Действие": action_type_label(action_type),
            "Сейчас": row.get("current_value", "—"),
            "Предлагается": row.get("target_value", "—"),
            "Прибыль": fmt_pct_delta(_safe_metric_float(eff.get("conservative_profit_delta_pct", eff.get("profit_delta_pct", 0.0)))),
            "Выручка": fmt_pct_delta(_safe_metric_float(eff.get("revenue_delta_pct", 0.0))),
            "Спрос": fmt_pct_delta(_safe_metric_float(eff.get("demand_delta_pct", 0.0))),
            "Риск": risk_level_label(rel.get("risk_level", "n/a")) if isinstance(rel, dict) else "—",
            "Надёжность": f"{_safe_metric_float(rel.get('score', 0.0)):.0f}/100" if isinstance(rel, dict) else "—",
        })
    return pd.DataFrame(rows)


def render_price_optimizer_summary(opt: Dict[str, Any]) -> None:
    if not isinstance(opt, dict) or len(opt) == 0:
        render_empty_state("Подбор цены ещё не запускался", "Нажмите кнопку ниже, чтобы проверить несколько цен через текущий механизм what-if.")
        return
    title = str(opt.get("recommendation_title", "Результат анализа цены"))
    text = str(opt.get("recommendation_text", ""))
    status = str(opt.get("status", ""))
    css = "scenario-info-inline"
    if status in {"price_increase_recommended", "price_decrease_recommended"}:
        css = "scenario-success-inline"
    elif status in {"insufficient_support", "risky_only", "no_valid_candidates", "optimizer_error"}:
        css = "scenario-warning-inline"
    elif status in {"no_profit_data"}:
        css = "scenario-danger-inline"
    msg_text = "Подходящий вариант цены не найден при текущих ограничениях. Попробуйте расширить диапазон проверки или изменить цель." if status in {"no_valid_candidates", "optimizer_error"} else text
    st.markdown(f'<div class="{css}"><b>{title}</b><br>{msg_text}</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Текущая цена", fmt_price(opt.get("current_price", np.nan)))
    price_label = "Лучший найденный вариант цены" if str(opt.get("status", "")) in ACTIONABLE_PRICE_OPT_STATUSES else "Лучший расчётный вариант"
    c2.metric(price_label, fmt_price(opt.get("recommended_price", np.nan)))
    c3.metric("Ожидаемая прибыль", fmt_money_total(opt.get("recommended_profit", np.nan)), fmt_pct_delta(opt.get("profit_delta_pct", np.nan)))
    conf = float(opt.get("confidence", 0.0)) if np.isfinite(float(opt.get("confidence", 0.0))) else np.nan
    c4.metric("Надёжность", str(opt.get("confidence_label", "—")) or "—", f"{conf:.2f}" if np.isfinite(conf) else "—")
    rec_min, rec_max = opt.get("recommended_price_min", np.nan), opt.get("recommended_price_max", np.nan)
    if np.isfinite(float(rec_min)) and np.isfinite(float(rec_max)):
        st.info(f"Рекомендуемый диапазон: **{fmt_price(rec_min)} — {fmt_price(rec_max)}**.")
    d1, d2, d3 = st.columns(3)
    d1.metric("Спрос", fmt_units(opt.get("recommended_demand", np.nan)), fmt_pct_delta(opt.get("demand_delta_pct", np.nan)))
    d2.metric("Выручка", fmt_money_total(opt.get("recommended_revenue", np.nan)), fmt_pct_delta(opt.get("revenue_delta_pct", np.nan)))
    d3.metric("Прибыль", fmt_money_total(opt.get("recommended_profit", np.nan)), fmt_pct_delta(opt.get("profit_delta_pct", np.nan)))
    st.caption("Изменения сравниваются с текущей ценой при выбранных условиях сценария.")
    warnings_list = opt.get("warnings", [])
    if isinstance(warnings_list, list) and warnings_list:
        with st.expander("Ограничения и предупреждения", expanded=False):
            for w in warnings_list[:8]:
                st.warning(str(w))


def render_price_optimizer_chart(opt: Dict[str, Any]) -> None:
    candidates = opt.get("candidates", pd.DataFrame()) if isinstance(opt, dict) else pd.DataFrame()
    if not isinstance(candidates, pd.DataFrame) or len(candidates) == 0:
        st.info("Нет таблицы вариантов для графика.")
        return
    plot_df = candidates.copy()
    plot_df["Цена"] = pd.to_numeric(plot_df["price"], errors="coerce")
    plot_df["Прибыль"] = pd.to_numeric(plot_df["profit"], errors="coerce")
    plot_df["Статус"] = np.where(plot_df.get("valid_for_recommendation", False).astype(bool), "Можно рекомендовать", "Только ориентир / риск")
    fig = px.line(plot_df, x="Цена", y="Прибыль", markers=True, color="Статус")
    current_price = opt.get("current_price", np.nan)
    recommended_price = opt.get("recommended_price", np.nan)
    status = str(opt.get("status", ""))
    recommended_label = "Рекомендация" if status in ACTIONABLE_PRICE_OPT_STATUSES else "Расчётный вариант"
    if np.isfinite(float(current_price)):
        fig.add_vline(x=float(current_price), line_dash="dash", annotation_text="Текущая цена", annotation_position="top left")
    if recommended_price is not None and np.isfinite(float(recommended_price)):
        fig.add_vline(x=float(recommended_price), line_dash="dot", annotation_text=recommended_label, annotation_position="top right")
    fig.update_layout(**_plot_layout("Прибыль при разных ценах"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_price_optimizer_table(opt: Dict[str, Any]) -> None:
    candidates = opt.get("candidates", pd.DataFrame()) if isinstance(opt, dict) else pd.DataFrame()
    if not isinstance(candidates, pd.DataFrame) or len(candidates) == 0:
        return
    table = candidates.copy()
    current_profit = float(opt.get("current_profit", np.nan))
    if np.isfinite(current_profit):
        table["profit_delta_pct"] = ((pd.to_numeric(table["profit"], errors="coerce") - current_profit) / max(abs(current_profit), 1e-9)) * 100.0
    else:
        table["profit_delta_pct"] = np.nan
    table["status_text"] = np.where(table.get("valid_for_recommendation", False).astype(bool), "Можно рекомендовать", "Только ориентир / риск")
    view = table[["price", "demand", "revenue", "profit", "profit_delta_pct", "margin_pct", "confidence_label", "support_label", "status_text"]].copy()
    view = view.rename(columns={"price": "Цена", "demand": "Спрос", "revenue": "Выручка", "profit": "Прибыль", "profit_delta_pct": "Δ прибыли, %", "margin_pct": "Маржа, %", "confidence_label": "Надёжность", "support_label": "Поддержка", "status_text": "Статус"})
    st.dataframe(view.sort_values("Прибыль", ascending=False).head(10), use_container_width=True, hide_index=True)


def fmt_pp_delta(value: Any) -> str:
    if _is_nan(value):
        return "—"
    v = float(value)
    if abs(v) < 1e-12:
        return "0.0 п.п."
    return f"{v:+.1f} п.п."


def fmt_money(value: Any) -> str:
    return fmt_money_total(value)


def fmt_pct(value: Any) -> str:
    return fmt_pct_delta(value)


def fmt_pp(value: Any) -> str:
    return fmt_pp_delta(value)


def delta_class(value: Any) -> str:
    if _is_nan(value):
        return "delta-neutral"
    v = float(value)
    if v > 0:
        return "delta-positive"
    if v < 0:
        return "delta-negative"
    return "delta-neutral"


def multiplier_to_pct(mult: Any) -> float:
    if _is_nan(mult):
        return float("nan")
    return (float(mult) - 1.0) * 100.0


def pct_to_multiplier(pct: Any) -> float:
    if _is_nan(pct):
        return 1.0
    return 1.0 + float(pct) / 100.0


def percent_slider_to_share(
    label: str,
    state_key: str,
    min_pct: int,
    max_pct: int,
    step: int = 1,
    default_share: float = 0.0,
) -> float:
    current_share = float(st.session_state.get(state_key, default_share))
    current_pct = int(np.clip(round(current_share * 100.0), min_pct, max_pct))
    selected_pct = st.slider(
        label,
        min_value=min_pct,
        max_value=max_pct,
        value=current_pct,
        step=step,
        format="%d%%",
    )
    share = float(selected_pct) / 100.0
    st.session_state[state_key] = share
    return share


def align_forecasts_by_scenario_dates(base_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or scenario_df is None or len(scenario_df) == 0:
        return pd.DataFrame(columns=list(base_df.columns) if isinstance(base_df, pd.DataFrame) else [])
    base = base_df.copy()
    scenario = scenario_df.copy()
    if "date" not in base.columns or "date" not in scenario.columns:
        raise ValueError("Для выравнивания прогнозов требуется колонка date в базе и сценарии.")
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    scenario["date"] = pd.to_datetime(scenario["date"], errors="coerce").dt.normalize()
    base = base.dropna(subset=["date"])
    scenario = scenario.dropna(subset=["date"])
    if base["date"].duplicated().any() or scenario["date"].duplicated().any():
        raise ValueError("В базе или сценарии найдены дублирующиеся даты.")
    aligned = base[base["date"].isin(set(scenario["date"]))].copy().sort_values("date").reset_index(drop=True)
    scenario_sorted = scenario.sort_values("date").reset_index(drop=True)
    if len(aligned) != len(scenario_sorted):
        raise ValueError(f"Base/scenario horizon mismatch: base={len(aligned)}, scenario={len(scenario_sorted)}")
    if not aligned["date"].equals(scenario_sorted["date"]):
        raise ValueError("Даты базы и сценария не совпадают после выравнивания.")
    return aligned


def calculate_scenario_deltas(as_is_forecast: pd.DataFrame, scenario_forecast: pd.DataFrame) -> Dict[str, float]:
    as_is_aligned = align_forecasts_by_scenario_dates(as_is_forecast, scenario_forecast)

    base_units = float(as_is_aligned["actual_sales"].sum()) if len(as_is_aligned) else 0.0
    base_revenue = float(as_is_aligned["revenue"].sum()) if len(as_is_aligned) else 0.0
    base_profit = float(as_is_aligned["profit"].sum()) if ("profit" in as_is_aligned.columns and len(as_is_aligned)) else float("nan")

    scenario_units = float(scenario_forecast["actual_sales"].sum()) if len(scenario_forecast) else 0.0
    scenario_revenue = float(scenario_forecast["revenue"].sum()) if len(scenario_forecast) else 0.0
    scenario_profit = float(scenario_forecast["profit"].sum()) if ("profit" in scenario_forecast.columns and len(scenario_forecast)) else float("nan")

    delta_units = scenario_units - base_units
    delta_revenue = scenario_revenue - base_revenue
    delta_profit = scenario_profit - base_profit if np.isfinite(scenario_profit) and np.isfinite(base_profit) else float("nan")

    delta_units_pct = (delta_units / max(abs(base_units), 1e-9) * 100.0) if np.isfinite(base_units) else float("nan")
    delta_revenue_pct = (delta_revenue / max(abs(base_revenue), 1e-9) * 100.0) if np.isfinite(base_revenue) else float("nan")
    delta_profit_pct = safe_signed_pct(delta_profit, base_profit) if np.isfinite(delta_profit) else float("nan")

    base_margin_pct = (base_profit / max(abs(base_revenue), 1e-9) * 100.0) if np.isfinite(base_profit) else float("nan")
    scenario_margin_pct = (scenario_profit / max(abs(scenario_revenue), 1e-9) * 100.0) if np.isfinite(scenario_profit) else float("nan")

    return {
        "base_units": base_units,
        "base_revenue": base_revenue,
        "base_profit": base_profit,
        "scenario_units": scenario_units,
        "scenario_revenue": scenario_revenue,
        "scenario_profit": scenario_profit,
        "delta_units": delta_units,
        "delta_revenue": delta_revenue,
        "delta_profit": delta_profit,
        "delta_units_pct": delta_units_pct,
        "delta_revenue_pct": delta_revenue_pct,
        "delta_profit_pct": delta_profit_pct,
        "base_margin_pct": base_margin_pct,
        "scenario_margin_pct": scenario_margin_pct,
    }


def render_scenario_status_card(status: str, selected_mode_label: str, snapshot: Optional[Dict[str, Any]] = None) -> None:
    _ = selected_mode_label
    _ = snapshot
    state_map = {
        "as_is": (
            "Базовый прогноз построен",
            "Сценарий ещё не применён. Измените параметры и нажмите «Рассчитать сценарий».",
            "Текущий план",
            "status-badge neutral",
        ),
        "dirty": (
            "Есть неприменённые изменения",
            "Поля сценария изменены. Результаты ниже относятся к последнему рассчитанному сценарию.",
            "Не рассчитано",
            "status-badge warning",
        ),
        "applied": (
            "Сценарий рассчитан",
            "Ниже показано сравнение сценария с текущим планом.",
            "Рассчитано",
            "status-badge success",
        ),
        "saved": (
            "Сценарий рассчитан и сохранён",
            "Сценарий сохранён как вариант сравнения.",
            "Сохранено",
            "status-badge success",
        ),
    }
    title, subtitle, pill, pill_class = state_map.get(status, state_map["as_is"])
    st.markdown(
        f"""
<div class="scenario-status-card">
  <div class="scenario-status-left">
    <div class="scenario-status-title">{title}</div>
    <div class="scenario-status-subtitle">{subtitle}</div>
  </div>
  <div class="{pill_class}">{pill}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_scenario_preview(
    current_price: float,
    manual_price: float,
    discount: float,
    promo_value: float,
    freight_change_pct: float,
    demand_shock_pct: float,
    hdays: int,
    segment_count: int = 0,
    scenario_changed: bool = False,
) -> None:
    net_price = float(manual_price) * (1.0 - float(discount))
    rows = [
        ("Текущая цена", fmt_price(current_price)),
        ("Новая цена до скидки", fmt_price(manual_price)),
        ("Цена после скидки", fmt_price(net_price)),
        ("Скидка", fmt_pct_abs(float(discount) * 100.0)),
        ("Промо", fmt_pct_abs(float(promo_value) * 100.0)),
        ("Логистика", fmt_pct_delta(freight_change_pct)),
        ("Ручная поправка спроса", fmt_pct_delta(demand_shock_pct)),
        ("Период", f"{int(hdays)} дней"),
        ("Сегменты", f"{int(segment_count)}"),
        ("Статус", "Изменения ещё не рассчитаны" if scenario_changed else "Нет новых изменений"),
    ]
    st.markdown('<div class="scenario-preview">', unsafe_allow_html=True)
    st.markdown('<div class="scenario-card-title">Что будет рассчитано</div>', unsafe_allow_html=True)
    for label, val in rows:
        st.markdown(
            f'<div class="preview-row"><div class="preview-label">{label}</div><div class="preview-value">{val}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_result_kpi_grid(metrics: Dict[str, Dict[str, Any]]) -> None:
    st.markdown('<div class="result-kpi-grid">', unsafe_allow_html=True)
    for key in ["Спрос", "Выручка", "Прибыль", "Маржа"]:
        item = metrics[key]
        delta_text = item.get("delta_text", "—")
        base_text = item.get("base_text", "—")
        dclass = delta_class(item.get("delta_numeric", np.nan))
        st.markdown(
            (
                '<div class="result-kpi-card">'
                f'<div class="result-kpi-label">{key}</div>'
                f'<div class="result-kpi-value">{item.get("value", "—")}</div>'
                f'<div class="result-kpi-delta {dclass}">{delta_text}</div>'
                f'<div class="scenario-help">База: {base_text}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_business_summary(revenue_delta: float, profit_delta: float, demand_delta: float, margin_delta: float) -> None:
    if abs(float(profit_delta)) < 1e-6:
        css = "scenario-warning-inline"
        text = "Сценарий почти не меняет прибыль. Решение стоит принимать по дополнительным ограничениям бизнеса."
    elif float(profit_delta) > 0:
        css = "scenario-success-inline"
        text = "Сценарий улучшает прибыль относительно текущего плана."
    else:
        css = "scenario-danger-inline"
        text = "Сценарий снижает прибыль: прирост спроса не компенсирует ухудшение маржи."
    extra = []
    if demand_delta > 0 and profit_delta < 0:
        extra.append("Спрос растёт, но прибыль падает — вероятно, скидка или затраты снижают маржинальность сильнее, чем растёт объём.")
    if demand_delta < 0 and profit_delta > 0:
        extra.append("Спрос снижается, но прибыль растёт — сценарий может быть выгоден за счёт более высокой маржи.")
    extra_line = ""
    if extra:
        extra_line = f'<div class="scenario-help">{" ".join(extra)}</div>'
    st.markdown(
        f'<div class="{css}">{text}'
        f'<div class="scenario-help">Δ спроса: {fmt_units(demand_delta)} · Δ выручки: {fmt_money_total(revenue_delta)} · Δ маржи: {fmt_pp_delta(margin_delta)}</div>'
        f"{extra_line}</div>",
        unsafe_allow_html=True,
    )


def classify_scenario_assessment(
    profit_delta_pct: float,
    warnings_list: List[str],
    ood_flag: bool,
    confidence_label: str,
    shape_quality_low: bool,
    support_label: str,
) -> Tuple[str, str]:
    _ = warnings_list
    _ = ood_flag
    _ = confidence_label
    _ = shape_quality_low
    _ = support_label
    if profit_delta_pct <= -3.0:
        return "Рискован", "scenario-danger-inline"
    if profit_delta_pct >= 3.0:
        return "Потенциально выгоден", "scenario-success-inline"
    if -3.0 < profit_delta_pct < 3.0:
        return "Нейтральный", "scenario-warning-inline"
    return "Нейтральный", "scenario-warning-inline"


def classify_economic_verdict(profit_delta_pct: float, demand_delta_pct: float, revenue_delta_pct: float) -> Tuple[str, str, str]:
    if not np.isfinite(profit_delta_pct):
        return "Недостаточно данных", "scenario-warning-inline", "Не хватает данных для экономического вывода."
    if profit_delta_pct < 0:
        text = (
            f"Спрос {fmt_pct_delta(demand_delta_pct)}, выручка {fmt_pct_delta(revenue_delta_pct)}, "
            f"но прибыль {fmt_pct_delta(profit_delta_pct)}. Сценарий экономически невыгоден: рост объёма не компенсирует потерю маржи."
        )
        return "Невыгоден", "scenario-danger-inline", text
    if profit_delta_pct >= 0 and demand_delta_pct < 0:
        text = (
            f"Прибыль {fmt_pct_delta(profit_delta_pct)}, но спрос {fmt_pct_delta(demand_delta_pct)}. "
            "Финансово сценарий лучше, но есть риск потери объёма."
        )
        return "Проверить риск спроса", "scenario-warning-inline", text
    text = (
        f"Спрос {fmt_pct_delta(demand_delta_pct)}, выручка {fmt_pct_delta(revenue_delta_pct)}, прибыль {fmt_pct_delta(profit_delta_pct)}. "
        "Сценарий можно рассмотреть."
    )
    return "Можно рассмотреть", "scenario-success-inline", text


def safe_signed_pct(delta: float, base: float) -> float:
    if not np.isfinite(delta) or not np.isfinite(base) or abs(base) < 1e-9:
        return float("nan")
    return float(delta / abs(base) * 100.0)


def classify_reliability_verdict(
    ood_flag: bool,
    warnings_list: List[str],
    confidence_label: str,
    support_label: str,
    shape_quality_low: bool,
    validation_ok: bool,
) -> Tuple[str, str, str]:
    if not validation_ok:
        return "Низкая", "scenario-danger-inline", "Validation gate не пройден: результат невалиден и не должен использоваться."
    low_conf = str(confidence_label).lower() in {"низкая", "low"}
    low_support = str(support_label).lower() in {"low", "none", "низкая", "нет"}
    if ood_flag or low_conf or low_support or shape_quality_low:
        msg = "Есть сигналы риска (OOD/поддержка/форма прогноза). Используйте результат как ориентир, а не как гарантию."
        return "Требует осторожности", "scenario-warning-inline", msg
    if warnings_list:
        return "Средняя", "scenario-warning-inline", "Есть предупреждения: перед решением проверьте ограничения и диапазоны."
    return "Высокая", "scenario-success-inline", "Сценарий внутри допустимого диапазона, критичных предупреждений нет."


def render_profit_change_explanation(
    base_row: Dict[str, float],
    scenario_row: Dict[str, float],
    demand_delta_pct: float,
    profit_delta: float,
) -> None:
    base_unit_margin = float(base_row["unit_margin"])
    scenario_unit_margin = float(scenario_row["unit_margin"])
    trend = "вырос" if demand_delta_pct >= 0 else "снизился"
    profit_trend = "выросла" if profit_delta >= 0 else "снизилась"
    st.markdown(
        "Спрос "
        f"{trend} на {fmt_pct_delta(demand_delta_pct)}. "
        f"Цена после скидки была {fmt_price(base_row['net_price'])}, стала {fmt_price(scenario_row['net_price'])}. "
        f"Маржа на единицу была {fmt_money_total(base_unit_margin)}, стала {fmt_money_total(scenario_unit_margin)}. "
        f"Итоговая прибыль была {fmt_money_total(base_row['profit'])}, стала {fmt_money_total(scenario_row['profit'])}, поэтому прибыль {profit_trend}."
    )


def render_human_effect_breakdown(effect_breakdown: Dict[str, Any]) -> None:
    eb = effect_breakdown or {}
    def _effect(name: str, key: str) -> str:
        val = eb.get(key, np.nan)
        if _is_nan(val):
            return "—"
        return f"{fmt_pct_delta((float(val) - 1.0) * 100.0)} к спросу"
    stock_flag = eb.get("stock_constraint_active", None)
    stock_text = "ограничение применялось" if bool(stock_flag) else ("ограничение не применялось" if stock_flag is not None else "—")
    rows = [
        ("Цена", _effect("Цена", "price_effect_mean")),
        ("Промо", _effect("Промо", "promo_effect_mean")),
        ("Логистика", _effect("Логистика", "freight_effect_mean")),
        ("Внешний шок", _effect("Внешний шок", "shock_multiplier_mean")),
        ("Ограничение запасом", stock_text),
    ]
    st.markdown('<div class="effect-list">', unsafe_allow_html=True)
    for name, val in rows:
        st.markdown(f'<div class="effect-row"><div class="effect-name">{name}</div><div class="effect-value">{val}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_reliability_card(wr: Dict[str, Any], r: Dict[str, Any]) -> None:
    warnings_list = list((wr or {}).get("warnings", []))
    warnings_list.extend(list(((wr or {}).get("validation_gate", {}) or {}).get("warnings", [])))
    summary = ((r.get("summary", {}) if isinstance(r, dict) else {}) or {})
    diagnostics = ((r.get("diagnostics", {}) if isinstance(r, dict) else {}) or {})
    shape_quality_low = bool(summary.get("shape_quality_low", diagnostics.get("shape_quality_low", False)))
    confidence_label = str((wr or {}).get("confidence_label", "—"))
    support_label = str((wr or {}).get("support_label", "—"))
    reliability_label, _, reliability_text = classify_reliability_verdict(
        bool((wr or {}).get("ood_flag", False)),
        warnings_list,
        confidence_label,
        support_label,
        shape_quality_low,
        bool(((wr or {}).get("validation_gate", {}) or {}).get("ok", True)),
    )
    st.markdown(f"**Надёжность расчёта:** {reliability_label}")
    st.markdown(f"**Поддержка сценария историей:** {support_label}")
    st.markdown(f"**Уверенность расчёта:** {confidence_label}")
    st.markdown(f"**Комментарий:** {reliability_text}")
    holdout = (r.get("holdout_metrics", pd.DataFrame()) if isinstance(r, dict) else pd.DataFrame())
    wape = float(holdout.get("wape", pd.Series([np.nan])).iloc[0]) if len(holdout) else float("nan")
    mape = float(holdout.get("mape", pd.Series([np.nan])).iloc[0]) if len(holdout) else float("nan")
    summary = (r.get("summary", {}) if isinstance(r, dict) else {}) or {}
    corr_final = summary.get("corr_final", np.nan)
    std_ratio_final = summary.get("std_ratio_final", np.nan)
    st.markdown(
        f"**Метрики формы:** WAPE {wape:.1%}" if np.isfinite(wape) else "**Метрики формы:** WAPE —"
    )
    st.markdown(
        f"**MAPE:** {mape:.1%}" if np.isfinite(mape) else "**MAPE:** —"
    )
    st.markdown(
        f"**Корреляция формы (corr_final):** {float(corr_final):.2f}" if np.isfinite(float(corr_final)) else "**Корреляция формы (corr_final):** —"
    )
    st.markdown(
        f"**Отношение волатильности (std_ratio_final):** {float(std_ratio_final):.2f}" if np.isfinite(float(std_ratio_final)) else "**Отношение волатильности (std_ratio_final):** —"
    )
    checks = [
        f"OOD: {'да' if bool((wr or {}).get('ood_flag', False)) else 'нет'}",
        f"Ограничение цены: {'да' if bool((wr or {}).get('clip_applied', (wr or {}).get('price_clipped', False))) else 'нет'}",
        f"Fallback: {'да' if bool((wr or {}).get('fallback_multiplier_used', False)) else 'нет'}",
    ]
    st.markdown("**Проверки:** " + " · ".join(checks))
    if shape_quality_low:
        st.markdown(
            '<div class="scenario-warning-inline">Форма прогноза воспроизводится неидеально. Используйте сценарий как ориентир для сравнения, а не как точную гарантию дневных продаж.</div>',
            unsafe_allow_html=True,
        )
    if shape_quality_low and str(confidence_label).lower() in {"высокая", "high"}:
        st.markdown(
            '<div class="scenario-warning-inline">Поддержка сценария высокая, но форма прогноза требует осторожности.</div>',
            unsafe_allow_html=True,
        )
    if not warnings_list:
        st.markdown('<div class="scenario-success-inline">Критичных предупреждений нет.</div>', unsafe_allow_html=True)
    else:
        for w in warnings_list[:3]:
            st.markdown(f'<div class="scenario-warning-inline">{w}</div>', unsafe_allow_html=True)
        if len(warnings_list) > 3:
            with st.expander("Все предупреждения", expanded=False):
                for w in warnings_list[3:]:
                    st.write(str(w))


def validate_scenario_consistency(
    as_is_df: pd.DataFrame,
    wr: Dict[str, Any],
    tol_money: float = 0.01,
    tol_units: float = 1e-6,
    expected_hdays: Optional[int] = None,
) -> Dict[str, Any]:
    if not wr or not isinstance(wr.get("daily", None), pd.DataFrame):
        return {"ok": False, "errors": ["Сценарные данные отсутствуют."], "warnings": []}
    daily = wr["daily"].copy()
    errors: List[str] = []
    warns: List[str] = []
    if "date" not in daily.columns:
        errors.append("В сценарии отсутствует колонка date.")
    else:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        if daily["date"].isna().any():
            errors.append("В сценарии есть некорректные даты.")
        if daily["date"].duplicated().any():
            errors.append("В сценарии есть дублирующиеся даты.")
    if expected_hdays is not None and len(daily) != int(expected_hdays):
        errors.append(f"Горизонт сценария не совпадает с выбранным периодом: {len(daily)} vs {int(expected_hdays)}.")
    if pd.to_numeric(daily.get("actual_sales", 0.0), errors="coerce").isna().any():
        errors.append("В сценарии есть NaN в спросе.")
    if pd.to_numeric(daily.get("revenue", 0.0), errors="coerce").isna().any():
        errors.append("В сценарии есть NaN в выручке.")
    if pd.to_numeric(daily.get("profit", 0.0), errors="coerce").isna().any():
        errors.append("В сценарии есть NaN в прибыли.")
    if (pd.to_numeric(daily.get("actual_sales", 0.0), errors="coerce").fillna(0.0) < -tol_units).any():
        errors.append("В сценарии найден отрицательный спрос.")
    if "price" in daily.columns and (pd.to_numeric(daily["price"], errors="coerce").fillna(0.0) < -tol_money).any():
        errors.append("В сценарии найдена отрицательная цена.")

    sum_units = float(pd.to_numeric(daily.get("actual_sales", 0.0), errors="coerce").fillna(0.0).sum())
    sum_revenue = float(pd.to_numeric(daily.get("revenue", 0.0), errors="coerce").fillna(0.0).sum())
    sum_profit = float(pd.to_numeric(daily.get("profit", 0.0), errors="coerce").fillna(0.0).sum())
    if abs(sum_units - float(wr.get("demand_total", sum_units))) > tol_units:
        errors.append("Сумма дневного спроса не совпадает с итогом summary.")
    if abs(sum_revenue - float(wr.get("revenue_total", sum_revenue))) > tol_money:
        errors.append("Сумма дневной выручки не совпадает с итогом summary.")
    if abs(sum_profit - float(wr.get("profit_total", sum_profit))) > tol_money:
        errors.append("Сумма дневной прибыли не совпадает с итогом summary.")
    try:
        as_is_aligned = align_forecasts_by_scenario_dates(as_is_df, daily)
    except Exception as exc:
        errors.append(f"Невозможно выровнять текущий план и сценарий по датам: {exc}")
        as_is_aligned = pd.DataFrame(columns=as_is_df.columns)

    as_is_units = float(pd.to_numeric(as_is_aligned.get("actual_sales", 0.0), errors="coerce").fillna(0.0).sum())
    if abs((sum_units - as_is_units) - float(wr.get("demand_total", sum_units) - as_is_units)) > tol_units:
        errors.append("Δ спроса в summary не совпадает с дневными данными.")
    if "net_unit_price" in daily.columns:
        implied_revenue = pd.to_numeric(daily.get("actual_sales", 0.0), errors="coerce").fillna(0.0) * pd.to_numeric(daily.get("net_unit_price", 0.0), errors="coerce").fillna(0.0)
        revenue_delta = float(abs(implied_revenue.sum() - sum_revenue))
        if revenue_delta > max(tol_money, tol_money * max(abs(sum_revenue), 1.0)):
            warns.append("Выручка заметно отличается от demand × net_price (проверьте входные данные).")
    if bool(wr.get("fallback_multiplier_used", False)):
        warns.append("Использовался fallback multiplier.")
    if bool(wr.get("clip_applied", wr.get("price_clipped", False))):
        warns.append("Цена была скорректирована guardrails.")
    if bool(wr.get("ood_flag", False)):
        warns.append("Сценарий вышел за исторический диапазон (OOD).")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warns}


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
        "scenario_calc_mode": str(applied_snapshot.get("scenario_calc_mode", base_form["scenario_calc_mode"])),
        "price_guardrail_mode": normalize_price_guardrail_mode(applied_snapshot.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
    }
    if _same(current_form, applied_form):
        return "saved" if current_status == "saved" else "applied"
    return "dirty"


def render_scenario_status_banner(
    status: str,
    snapshot: Optional[Dict[str, Any]] = None,
    last_saved_slot: Optional[str] = None,
    selected_mode: str = DEFAULT_SCENARIO_CALC_MODE,
) -> None:
    _ = last_saved_slot
    render_scenario_status_card(status, scenario_mode_label(selected_mode), snapshot=snapshot)


def render_applied_scenario_block(snapshot: Optional[Dict[str, Any]]) -> None:
    if not snapshot:
        return
    open_surface("Параметры рассчитанного сценария")
    c1, c2 = st.columns(2)
    is_cb_full = str(snapshot.get("scenario_calc_mode", "")) == CATBOOST_FULL_FACTOR_MODE
    if is_cb_full:
        c1.metric("Введённая цена", fmt_price(snapshot.get("requested_price_gross", snapshot.get("manual_price_requested", np.nan))))
        c1.metric("Цена для финансового расчёта", fmt_price(snapshot.get("applied_price_gross", snapshot.get("manual_price_applied", np.nan))))
    else:
        c1.metric("Новая цена до скидки", fmt_price(snapshot.get("manual_price_applied", np.nan)))
    c1.metric("Скидка", fmt_pct_abs(float(snapshot.get("discount_applied", 0.0)) * 100.0))
    c1.metric("Цена после скидки", fmt_price(snapshot.get("net_price_applied", np.nan)))
    c1.metric("Промо", fmt_pct_abs(float(snapshot.get("promo_applied", 0.0)) * 100.0))
    c2.metric("Изменение логистики", fmt_pct_delta(multiplier_to_pct(snapshot.get("freight_mult", np.nan))))
    c2.metric("Внешний шок спроса", fmt_pct_delta(multiplier_to_pct(snapshot.get("demand_mult", np.nan))))
    c2.metric("Период", f"{int(snapshot.get('horizon_days', 0))} дней")
    if is_cb_full:
        mode_code = str(snapshot.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE))
        mode_label = {
            PRICE_GUARDRAIL_SAFE_CLIP: "Осторожно: считать по ближайшей безопасной цене",
            PRICE_GUARDRAIL_EXTRAPOLATE: "Экстраполяция вне границ",
        }.get(mode_code, mode_code)
        c2.metric("Что делать, если цена вне прошлой истории?", mode_label)
    if is_cb_full and str(snapshot.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)) == PRICE_GUARDRAIL_SAFE_CLIP and bool(snapshot.get("price_clipped", False)):
        st.warning("Цена была вне безопасного диапазона. В защитном режиме расчёт выполнен по безопасной границе.")
    if is_cb_full and str(snapshot.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)) == PRICE_GUARDRAIL_EXTRAPOLATE and bool(snapshot.get("price_out_of_range", False)):
        st.warning("Цена вне безопасного диапазона. Введённая цена сохранена для финансового расчёта, CatBoost рассчитал спрос только до безопасной границы, а участок дальше рассчитан через ценовую эластичность.")
    if is_cb_full:
        with st.expander("Технические детали сценария", expanded=False):
            st.metric("Безопасная граница", fmt_price(snapshot.get("safe_price_gross", np.nan)))
            st.metric("Цена для модели", fmt_price(snapshot.get("price_for_model", snapshot.get("model_price_gross", np.nan))))
            elasticity_val = safe_float_or_nan(snapshot.get("elasticity_used", np.nan))
            tail_val = safe_float_or_nan(snapshot.get("extrapolation_tail_multiplier", 1.0))
            st.metric("Эластичность", f"{elasticity_val:.2f}" if np.isfinite(elasticity_val) else "n/a")
            st.metric("Хвостовой множитель спроса", f"{tail_val:.3f}" if np.isfinite(tail_val) else "n/a")
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
    st.session_state.sensitivity_df = None
    st.session_state.price_optimizer_result_base = None
    st.session_state.price_optimizer_signature_base = None
    st.session_state.price_optimizer_context = None
    st.session_state.scenario_ui_status = "as_is"
    if clear_saved_slot:
        st.session_state.last_saved_slot = None
    base_ctx = results.get("_trained_bundle", {}).get("base_ctx", {})
    st.session_state["what_if_price"] = float(results.get("current_price", 0.0))
    st.session_state["what_if_discount"] = float(base_ctx.get("discount", 0.0))
    st.session_state["what_if_promo"] = float(np.clip(base_ctx.get("promotion", 0.0), 0.0, 1.0))
    st.session_state["what_if_freight_mult"] = 1.0
    st.session_state["what_if_demand_mult"] = 1.0
    st.session_state["what_if_freight_change_pct"] = 0.0
    st.session_state["what_if_demand_shock_pct"] = 0.0
    st.session_state["what_if_hdays"] = int(CONFIG["HORIZON_DAYS_DEFAULT"])
    st.session_state["what_if_calc_mode"] = str(results.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
    st.session_state["price_guardrail_mode"] = DEFAULT_PRICE_GUARDRAIL_MODE
    st.session_state["what_if_use_segments"] = False
    st.session_state.form_last_values = collect_current_form_values(
        st.session_state["what_if_price"],
        st.session_state["what_if_discount"],
        st.session_state["what_if_promo"],
        st.session_state["what_if_freight_mult"],
        st.session_state["what_if_demand_mult"],
        st.session_state["what_if_hdays"],
        st.session_state["what_if_calc_mode"],
        st.session_state["price_guardrail_mode"],
    )
    return refresh_excel_export(results)


def build_user_friendly_comparison_table(
    as_is_forecast: pd.DataFrame,
    current_scenario_forecast: Optional[pd.DataFrame],
    saved_scenarios: Dict[str, Dict[str, Any]],
    show_margin: bool = True,
) -> pd.DataFrame:
    as_is_base = as_is_forecast.copy()
    if current_scenario_forecast is not None and len(current_scenario_forecast):
        as_is_base = align_forecasts_by_scenario_dates(as_is_forecast, current_scenario_forecast)
    as_is_units = float(as_is_base["actual_sales"].sum()) if len(as_is_base) else 0.0
    as_is_revenue = float(as_is_base["revenue"].sum()) if len(as_is_base) else 0.0
    as_is_profit = float(as_is_base["profit"].sum()) if ("profit" in as_is_base.columns and len(as_is_base)) else float("nan")

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

    rows: List[Dict[str, Any]] = [_row("Текущий план", as_is_units, as_is_revenue, as_is_profit)]
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
            slot_name = SCENARIO_SLOT_LABELS.get(slot, slot)
            row = _row(slot_name, float(s.get("units", 0.0)), float(s.get("revenue", 0.0)), float(s.get("profit", np.nan)))
            row.update(
                {
                    "Цена до скидки": float(s.get("applied_price_gross", np.nan)),
                    "Цена после скидки": float(s.get("applied_price_net", np.nan)),
                    "Скидка": float(s.get("applied_discount", np.nan)),
                    "Промо": float(s.get("promotion", np.nan)),
                    "Логистика": float(s.get("freight_value", np.nan)),
                    "Надёжность": str(s.get("confidence_label", "")),
                    "Поддержка данных": str(s.get("support_label", "")),
                    "Цена ограничена защитой": "Да" if bool(s.get("price_clipped", False)) else "Нет",
                    "Режим": str(s.get("scenario_calc_mode_label", scenario_mode_label(str(s.get("scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))))),
                    "Предупреждения": int(len(s.get("warnings", []))) if isinstance(s.get("warnings", []), list) else 0,
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
    if "price_optimizer_result_base" not in st.session_state:
        st.session_state.price_optimizer_result_base = None
    if "price_optimizer_signature_base" not in st.session_state:
        st.session_state.price_optimizer_signature_base = None
    if "price_optimizer_context" not in st.session_state:
        st.session_state.price_optimizer_context = None
    if "price_optimizer_auto_run" not in st.session_state:
        st.session_state.price_optimizer_auto_run = False
    if "saved_scenarios" not in st.session_state:
        st.session_state.saved_scenarios = {}
    if "compare_slot" not in st.session_state:
        st.session_state.compare_slot = "Scenario A"
    if "active_workspace_tab" not in st.session_state:
        st.session_state.active_workspace_tab = PAGE_OVERVIEW
    if "scenario_ui_status" not in st.session_state:
        st.session_state.scenario_ui_status = "as_is"
    if "applied_scenario_snapshot" not in st.session_state:
        st.session_state.applied_scenario_snapshot = None
    if "form_last_values" not in st.session_state:
        st.session_state.form_last_values = None
    if "last_saved_slot" not in st.session_state:
        st.session_state.last_saved_slot = None
    if "what_if_calc_mode" not in st.session_state:
        st.session_state["what_if_calc_mode"] = DEFAULT_SCENARIO_CALC_MODE
    if "what_if_use_segments" not in st.session_state:
        st.session_state["what_if_use_segments"] = False
    if "what_if_freight_change_pct" not in st.session_state:
        st.session_state["what_if_freight_change_pct"] = 0.0
    if "what_if_demand_shock_pct" not in st.session_state:
        st.session_state["what_if_demand_shock_pct"] = 0.0

    apply_theme()

    def _set_page(page_name: str, rerun: bool = True) -> None:
        st.session_state["nav_page"] = page_name
        try:
            st.query_params["page"] = page_name
        except Exception:
            try:
                st.experimental_set_query_params(page=page_name)
            except Exception:
                pass
        if rerun:
            st.rerun()

    query_page = st.session_state.get("nav_page")
    if not query_page:
        query_page = st.query_params.get("page", "landing")
        if isinstance(query_page, list):
            query_page = query_page[0]
    query_page = str(query_page or "landing")
    if query_page not in {"landing", "app"}:
        query_page = "landing"
    st.session_state["nav_page"] = query_page

    if query_page == "landing":
        from ui.components import (
            render_landing_nav,
            render_landing_hero,
            render_landing_decisions,
            render_landing_pipeline,
            render_landing_outputs,
            render_landing_data_requirements,
            render_landing_limits,
            render_landing_cta,
            render_landing_footer,
        )

        nav_action = render_landing_nav()
        hero_action = render_landing_hero()
        render_landing_decisions()
        render_landing_pipeline()
        render_landing_outputs()
        render_landing_data_requirements()
        render_landing_limits()
        cta_action = render_landing_cta()
        render_landing_footer()
        if nav_action == "app" or hero_action == "app" or cta_action == "app":
            _set_page("app", rerun=False)
            query_page = "app"
        else:
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
        render_overview_hero,
        render_decision_mode_cards,
        render_verdict_panel,
        render_wizard_steps,
        open_surface,
        close_surface,
        render_page_header,
        render_stepper,
        render_help_callout,
        render_kpi_strip,
        render_decision_summary_card,
        render_workspace_guide,
        render_product_empty_state,
        humanize_feature_name,
    )

    render_top_header()


    def _plot_layout(title: str) -> dict:
        return dict(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PLOT_TEXT),
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
        if str(wr.get("scenario_calc_mode")) == CATBOOST_FULL_FACTOR_MODE:
            st.markdown(
                "1) До расчёта обучается дневная факторная модель на продажах, календаре, цене, скидках, промо, логистике и дополнительных факторах из файла.  \n"
                "2) Система строит будущий дневной набор факторов.  \n"
                "3) Изменённые пользователем цена/скидка/промо/логистика подставляются в будущие строки.  \n"
                "4) Факторная модель повторно прогнозирует спрос по этим изменённым факторам.  \n"
                "5) Выручка и прибыль считаются из спрогнозированного спроса, цены после скидки, себестоимости и логистики.  \n"
                "6) Ручная поправка спроса применяется отдельно как пользовательское допущение."
            )
        else:
            st.markdown(
                "1) Пользователь задаёт **цену до скидки**.  \n"
                "2) Система применяет **защитное ограничение** по историческому диапазону, если это нужно.  \n"
                "3) Затем применяется **скидка** и формируется **цена после скидки**.  \n"
                "4) Реакция спроса считается именно от **цены после скидки**.  \n"
                "5) Выручка/прибыль считаются из применённой цены, скидки, себестоимости и логистики."
            )
        st.markdown("#### Что именно система сравнивает")
        st.markdown(
            f"- Цена: запрошено `{requested_price}` → использовано в расчёте `{model_price}`.  \n"
            f"- Скидка: `{effective.get('applied_discount', wr.get('scenario_inputs_contract', {}).get('scenario_discount', 'n/a'))}`.  \n"
            f"- Цена после скидки: `{effective.get('applied_price_net', wr.get('scenario_inputs_contract', {}).get('scenario_net_price', 'n/a'))}`.  \n"
            f"- Промо: `{effective.get('promotion', 'n/a')}`.  \n"
            f"- Логистика: `{effective.get('freight_value', 'n/a')}`.  \n"
            f"- Причина ограничения цены: `{effective.get('clip_reason', wr.get('clip_reason', ''))}`.  \n"
            f"- Надёжность: `{confidence}`.  \n"
            f"- Прибыль: расчётная `{profit_raw}` и осторожная оценка `{profit_adjusted}`."
        )

    def _fmt_money(v: float) -> str:
        return f"₽ {v:,.0f}"

    def _fmt_units(v: float) -> str:
        return f"{v:,.1f} шт."

def _download_blob(payload: Any, default: bytes) -> bytes:
        if payload is None:
            return default
        if isinstance(payload, BytesIO):
            try:
                return payload.getvalue()
            except Exception:
                return default
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, bytes):
            return payload
        try:
            return bytes(payload)
        except Exception:
            return default


def build_scenario_support_info(
    history_daily: pd.DataFrame,
    effective_scenario: Dict[str, Any],
    scenario_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    history = history_daily.copy()
    history["date"] = pd.to_datetime(history.get("date"), errors="coerce")
    history = history.dropna(subset=["date"])
    scenario_overrides = dict(scenario_overrides or {})
    hist_days = int(len(history))
    recent_days = int(min(90, hist_days))
    recent = history.tail(recent_days) if hist_days else history
    sales = pd.to_numeric(history.get("sales", pd.Series(np.zeros(hist_days))), errors="coerce").fillna(0.0)
    price = pd.to_numeric(history.get("price", pd.Series(np.zeros(hist_days))), errors="coerce").fillna(0.0)
    discount = pd.to_numeric(history.get("discount", pd.Series(np.zeros(hist_days))), errors="coerce").fillna(0.0).clip(0.0, 0.95)
    promo = pd.to_numeric(history.get("promotion", pd.Series(np.zeros(hist_days))), errors="coerce").fillna(0.0)
    freight = pd.to_numeric(history.get("freight_value", pd.Series(np.zeros(hist_days))), errors="coerce").fillna(0.0)
    net_price = price * (1.0 - discount)
    scen_net = float(effective_scenario.get("applied_price_net", 0.0))
    scen_promo = float(effective_scenario.get("promotion", 0.0))
    scen_freight = float(effective_scenario.get("freight_value", 0.0))
    local_price_support = int(((net_price >= scen_net * 0.93) & (net_price <= scen_net * 1.07)).sum()) if hist_days else 0
    local_promo_support = int((((promo > 0).astype(int)) == (1 if scen_promo > 0 else 0)).sum()) if hist_days else 0
    local_freight_support = int(((freight >= scen_freight * 0.90) & (freight <= scen_freight * 1.10)).sum()) if hist_days else 0
    promo_change_days = int(promo.diff().abs().gt(1e-9).sum()) if hist_days else 0
    freight_change_days = int(freight.diff().abs().gt(1e-9).sum()) if hist_days else 0
    support_score = float(
        np.clip(
            0.45 * min(local_price_support / 21.0, 1.0)
            + 0.25 * min(local_promo_support / 21.0, 1.0)
            + 0.30 * min(local_freight_support / 21.0, 1.0),
            0.0,
            1.0,
        )
    )
    support_label = "high" if support_score >= 0.7 else ("medium" if support_score >= 0.45 else "low")
    warnings = []
    if local_price_support < 14:
        warnings.append("Низкая локальная поддержка цены в истории.")
    if scen_promo != float(history.get("promotion", pd.Series([0.0])).mean() if hist_days else 0.0) and promo_change_days < 6:
        warnings.append("Промо-сценарий слабо поддержан историческими изменениями.")
    if scen_freight != float(history.get("freight_value", pd.Series([0.0])).mean() if hist_days else 0.0) and freight_change_days < 8:
        warnings.append("Фрахт-сценарий слабо поддержан историей.")
    return {
        "history_days": hist_days,
        "recent_history_days": recent_days,
        "recent_nonzero_sales_days": int((pd.to_numeric(recent.get("sales", pd.Series(np.zeros(len(recent)))), errors="coerce").fillna(0.0) > 0).sum()) if len(recent) else 0,
        "unique_price_points": int(price.nunique()) if hist_days else 0,
        "price_range_pct": float((price.max() - price.min()) / max(abs(float(price.median())), 1e-9)) if hist_days else 0.0,
        "price_changes": int(price.diff().abs().gt(1e-9).sum()) if hist_days else 0,
        "price_stability": float(np.clip(1.0 - (price.std(ddof=0) / max(abs(float(price.mean())), 1e-9)), 0.0, 1.0)) if hist_days else 0.0,
        "promo_active_days": int((promo > 0).sum()) if hist_days else 0,
        "promo_change_days": promo_change_days,
        "promo_weeks": int(history.assign(week=history["date"].dt.to_period("W")).groupby("week")["promotion"].mean().gt(0).sum()) if hist_days else 0,
        "promo_variability": float(promo.std(ddof=0)) if hist_days else 0.0,
        "freight_change_days": freight_change_days,
        "freight_changes": freight_change_days,
        "freight_variation": float(freight.std(ddof=0) / max(abs(float(freight.mean())), 1e-9)) if hist_days else 0.0,
        "discount_unique_count": int(discount.nunique()) if hist_days else 0,
        "promotion_positive_share": float((promo > 0).mean()) if hist_days else 0.0,
        "local_price_support_days": local_price_support,
        "local_promo_support_days": local_promo_support,
        "local_freight_support_days": local_freight_support,
        "support_score": support_score,
        "support_label": support_label,
        "warnings": warnings,
        "path_segments_used": bool(scenario_overrides.get("segments")),
        "segment_count": int(len(scenario_overrides.get("segments", []))),
        "has_time_varying_path": bool(
            any(k in scenario_overrides for k in ["price_path", "discount_path", "promo_path", "freight_path", "cost_path", "demand_multiplier_path"])
        ),
    }


def build_scenario_support_info_from_paths(
    history_daily: pd.DataFrame,
    scenario_daily: pd.DataFrame,
    scenario_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = build_scenario_support_info(history_daily, {}, scenario_overrides or {})
    hist = history_daily.copy()
    hist["date"] = pd.to_datetime(hist.get("date"), errors="coerce")
    hist = hist.dropna(subset=["date"])
    hist_price = pd.to_numeric(hist.get("price", pd.Series(np.zeros(len(hist)))), errors="coerce").fillna(0.0)
    hist_discount = pd.to_numeric(hist.get("discount", pd.Series(np.zeros(len(hist)))), errors="coerce").fillna(0.0).clip(0.0, 0.95)
    hist_net = hist_price * (1.0 - hist_discount)
    hist_promo = pd.to_numeric(hist.get("promotion", pd.Series(np.zeros(len(hist)))), errors="coerce").fillna(0.0)
    hist_freight = pd.to_numeric(hist.get("freight_value", pd.Series(np.zeros(len(hist)))), errors="coerce").fillna(0.0)
    scen = scenario_daily.copy()
    scen_net = pd.to_numeric(scen.get("scenario_price_net", scen.get("net_unit_price", pd.Series(np.zeros(len(scen))))), errors="coerce").fillna(0.0)
    scen_promo = pd.to_numeric(scen.get("scenario_promotion", scen.get("promotion", pd.Series(np.zeros(len(scen))))), errors="coerce").fillna(0.0)
    scen_freight = pd.to_numeric(scen.get("scenario_freight_value", scen.get("freight_value", pd.Series(np.zeros(len(scen))))), errors="coerce").fillna(0.0)
    price_support_counts = [int(((hist_net >= x * 0.93) & (hist_net <= x * 1.07)).sum()) for x in scen_net.head(60)] if len(scen_net) else [0]
    promo_support_counts = [int((((hist_promo > 0).astype(int)) == (1 if x > 0 else 0)).sum()) for x in scen_promo.head(60)] if len(scen_promo) else [0]
    freight_support_counts = [int(((hist_freight >= x * 0.90) & (hist_freight <= x * 1.10)).sum()) for x in scen_freight.head(60)] if len(scen_freight) else [0]
    local_price = int(np.median(price_support_counts)) if len(price_support_counts) else 0
    local_promo = int(np.median(promo_support_counts)) if len(promo_support_counts) else 0
    local_freight = int(np.median(freight_support_counts)) if len(freight_support_counts) else 0
    support_score = float(
        np.clip(
            0.45 * min(local_price / 21.0, 1.0)
            + 0.25 * min(local_promo / 21.0, 1.0)
            + 0.30 * min(local_freight / 21.0, 1.0),
            0.0,
            1.0,
        )
    )
    support_label = "high" if support_score >= 0.7 else ("medium" if support_score >= 0.45 else "low")
    warnings = []
    if local_price < 14:
        warnings.append("Низкая локальная поддержка price-path в истории.")
    if local_promo < 14:
        warnings.append("Низкая локальная поддержка promo-path в истории.")
    if local_freight < 14:
        warnings.append("Низкая локальная поддержка freight-path в истории.")
    out = dict(base)
    out.update(
        {
            "local_price_support_days": local_price,
            "local_promo_support_days": local_promo,
            "local_freight_support_days": local_freight,
            "support_score": support_score,
            "support_label": support_label,
            "path_price_min": float(scen_net.min()) if len(scen_net) else np.nan,
            "path_price_avg": float(scen_net.mean()) if len(scen_net) else np.nan,
            "path_price_max": float(scen_net.max()) if len(scen_net) else np.nan,
            "path_promo_days": int((scen_promo > 0).sum()) if len(scen_promo) else 0,
            "path_promo_share": float((scen_promo > 0).mean()) if len(scen_promo) else 0.0,
            "path_avg_freight": float(scen_freight.mean()) if len(scen_freight) else 0.0,
            "path_avg_demand_multiplier": float(pd.to_numeric(scen.get("shock_multiplier", pd.Series(np.ones(len(scen)))), errors="coerce").fillna(1.0).mean()) if len(scen) else 1.0,
            "warnings": warnings,
        }
    )
    return out


def build_segment_paths(
    future_dates: pd.DataFrame,
    default_values: Dict[str, float],
    segments: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if len(future_dates) == 0:
        return {}, warnings
    dates = pd.to_datetime(future_dates["date"], errors="coerce")
    price = pd.Series(np.repeat(float(default_values.get("price", 0.0)), len(dates)), index=dates)
    discount = pd.Series(np.repeat(float(default_values.get("discount", 0.0)), len(dates)), index=dates)
    promo = pd.Series(np.repeat(float(default_values.get("promotion", 0.0)), len(dates)), index=dates)
    freight_mult = pd.Series(np.repeat(float(default_values.get("freight_multiplier", 1.0)), len(dates)), index=dates)
    demand_mult = pd.Series(np.repeat(float(default_values.get("demand_multiplier", 1.0)), len(dates)), index=dates)
    covered = pd.Series(np.zeros(len(dates), dtype=int), index=dates)
    shocks: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        start = pd.to_datetime(seg.get("start_date"), errors="coerce")
        end = pd.to_datetime(seg.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            warnings.append(f"Сегмент {i+1}: некорректные даты.")
            continue
        mask = (dates >= start) & (dates <= end)
        if int(mask.sum()) == 0:
            warnings.append(f"Сегмент {i+1}: диапазон вне горизонта.")
            continue
        if int(covered.loc[mask.values].sum()) > 0:
            warnings.append(f"Сегмент {i+1}: пересекается с другим сегментом.")
        covered.loc[mask.values] += 1
        price.loc[mask.values] = float(seg.get("price", default_values.get("price", 0.0)))
        discount.loc[mask.values] = float(seg.get("discount", default_values.get("discount", 0.0)))
        promo.loc[mask.values] = float(seg.get("promotion", default_values.get("promotion", 0.0)))
        freight_mult.loc[mask.values] = float(seg.get("freight_multiplier", default_values.get("freight_multiplier", 1.0)))
        demand_mult.loc[mask.values] = float(seg.get("demand_multiplier", default_values.get("demand_multiplier", 1.0)))
        shock_units = float(seg.get("shock_units", 0.0))
        if abs(shock_units) > 1e-9:
            shocks.append(
                {
                    "shock_name": f"segment_{i+1}_units",
                    "shock_type": "units",
                    "shock_value": shock_units,
                    "start_date": str(start.date()),
                    "end_date": str(end.date()),
                }
            )
    if int((covered == 0).sum()) > 0:
        warnings.append("Некоторые дни горизонта не покрыты сегментами: использованы базовые scalar-параметры.")
    return {
        "price_path": [{"date": str(d.date()), "value": float(v)} for d, v in price.items()],
        "discount_path": [{"date": str(d.date()), "value": float(v)} for d, v in discount.items()],
        "promo_path": [{"date": str(d.date()), "value": float(v)} for d, v in promo.items()],
        "freight_path": [{"date": str(d.date()), "value": float(default_values.get("freight_value", 0.0)) * float(v)} for d, v in freight_mult.items()],
        "demand_multiplier_path": [{"date": str(d.date()), "value": float(v)} for d, v in demand_mult.items()],
        "segments": segments,
        "segment_warnings": warnings,
        "segment_shocks": shocks,
    }, warnings


def render_upload_screen() -> dict[str, Any]:
    from ui.components import render_page_header, render_help_callout, render_wizard_steps

    universal_file = st.session_state.get("universal_file")
    upload_done = universal_file is not None
    universal_txn = None
    schema_valid = False
    schema_errors: list[str] = []
    universal_quality: dict[str, Any] = {}
    data_gate: dict[str, Any] = resolve_data_quality_gate(universal_quality)
    raw_for_select = None
    universal_mapping: dict[str, Optional[str]] = {}
    target_category, target_sku = None, None
    horizon_days = int(st.session_state.get("input_horizon_days", CONFIG["HORIZON_DAYS_DEFAULT"]))
    analysis_calc_mode = str(st.session_state.get("input_analysis_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
    auto_price_optimizer = bool(st.session_state.get("input_auto_price_optimizer", False))
    auto_run_after_upload = bool(st.session_state.get("input_auto_run_after_upload", True))
    run_requested = False
    auto_run_notice = ""
    if "upload_columns_confirmed" not in st.session_state:
        st.session_state["upload_columns_confirmed"] = False
    current_file_name = getattr(universal_file, "name", "") if universal_file is not None else ""
    if st.session_state.get("last_uploaded_file_name") != current_file_name:
        st.session_state["last_uploaded_file_name"] = current_file_name
        st.session_state["upload_columns_confirmed"] = False
        st.session_state["upload_auto_started_for_file"] = None

    def _reset_upload_columns_confirmation() -> None:
        st.session_state["upload_columns_confirmed"] = False

    preview = None
    if upload_done:
        preview = read_uploaded_table_safely(universal_file)
        auto_map = build_auto_mapping(list(preview.columns))
        for f in CANONICAL_FIELDS:
            existing = st.session_state.get(f"map_universal_{f.name}")
            guessed = auto_map.get(f.name)
            universal_mapping[f.name] = None if existing == "<не использовать>" else (existing or guessed)
        missing_required = validate_mapping_required_columns(universal_mapping)
        if missing_required:
            schema_errors.append(f"Отсутствуют обязательные поля: {', '.join(missing_required)}")
        else:
            try:
                universal_txn, universal_quality = normalize_transactions(preview, universal_mapping)
                data_gate = resolve_data_quality_gate(universal_quality)
                if data_gate["errors"]:
                    schema_errors.extend(data_gate["errors"])
                elif data_gate["status"] in {"blocked", "diagnostic_only"}:
                    schema_valid = False
                    blockers = (
                        data_gate.get("hard_blockers")
                        or data_gate.get("blockers")
                        or ["Данные не проходят минимальные требования для расчёта."]
                    )
                    schema_errors.extend([str(x) for x in blockers])
                else:
                    schema_valid = True
                    raw_for_select = universal_txn.copy().rename(
                        columns={"category": "product_category_name", "product_id": "product_id"}
                    )
            except Exception as exc:
                schema_errors.append(str(exc))

    single_object_ready = False
    if schema_valid and raw_for_select is not None and len(raw_for_select) > 0:
        category_col_for_auto = "product_category_name" if "product_category_name" in raw_for_select.columns else "category"
        auto_categories = raw_for_select[category_col_for_auto].dropna().astype(str).unique()
        if len(auto_categories) == 1:
            auto_skus = raw_for_select[raw_for_select[category_col_for_auto].astype(str) == str(auto_categories[0])]["product_id"].astype(str).dropna().unique()
            single_object_ready = len(auto_skus) == 1
    if schema_valid and single_object_ready and auto_run_after_upload and not st.session_state.get("upload_columns_confirmed"):
        st.session_state["upload_columns_confirmed"] = True
    if not schema_valid:
        st.session_state["upload_columns_confirmed"] = False
    columns_confirmed = bool(st.session_state.get("upload_columns_confirmed"))
    sku_done = bool(st.session_state.get("input_target_category") and st.session_state.get("input_target_sku"))
    active_step = 0 if not upload_done else (1 if not columns_confirmed else 2)
    if st.button("← На лендинг", key="upload_back_to_landing", use_container_width=False):
        st.session_state["nav_page"] = "landing"
        st.rerun()
    render_page_header("Загрузка данных", [
        "Шаг 1 из 3 — загрузите файл",
        "Шаг 2 из 3 — проверьте колонки",
        "Шаг 3 из 3 — запустите анализ",
    ][active_step])
    render_wizard_steps(active_step, ["Загрузите файл", "Проверьте колонки", "Запустите анализ"])

    st.markdown('<div class="landing-upload-anchor">', unsafe_allow_html=True)
    if active_step == 0:
        open_surface("Загрузите файл с историей продаж")
        st.markdown("""**Минимально нужно:**
- дата
- товар/SKU
- продажи
- цена

**Желательно:** себестоимость, скидка, промо, логистика, остатки, внешние факторы.""")
        universal_file = st.file_uploader("Файл CSV/XLSX", type=["csv", "xlsx"], key="universal_file")
        upload_done = universal_file is not None
        render_help_callout("Подсказка", "Поддерживаются CSV и XLSX. Подробные настройки появятся после загрузки файла.", "info")
        close_surface()

    elif active_step == 1:
        open_surface("Проверьте распознавание колонок")
        if preview is not None:
            auto_map = build_auto_mapping(list(preview.columns))
            rows = []
            for f in CANONICAL_FIELDS:
                if not f.required:
                    continue
                found = universal_mapping.get(f.name) or auto_map.get(f.name)
                rows.append({
                    "Поле системы": CANONICAL_FIELD_UI.get(f.name, f.name),
                    "Найденная колонка": found or "—",
                    "Статус": "Готово" if found else "Нужно выбрать",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with st.expander("Настроить сопоставление колонок", expanded=bool(schema_errors)):
                for f in CANONICAL_FIELDS:
                    choices = ["<не использовать>"] + list(preview.columns)
                    guessed = universal_mapping.get(f.name) or auto_map.get(f.name)
                    idx = choices.index(guessed) if guessed in choices else 0
                    display_name = CANONICAL_FIELD_UI.get(f.name, f.description or f.name)
                    required_mark = " *" if f.required else ""
                    selected = st.selectbox(
                        f"{display_name}{required_mark}",
                        choices,
                        index=idx,
                        key=f"map_universal_{f.name}",
                        help=f"Системное поле: {f.name}. {'Обязательное поле.' if f.required else 'Необязательное поле.'}",
                        on_change=_reset_upload_columns_confirmation,
                    )
                    universal_mapping[f.name] = None if selected == "<не использовать>" else selected
            if schema_errors:
                for err in schema_errors:
                    st.error(err)
            for warning in data_gate.get("warnings", []) or []:
                st.warning(str(warning))
            if schema_valid:
                st.success(f"Данные готовы к расчёту: {len(universal_txn):,} строк после нормализации.")
                st.caption("Предпросмотр нормализованных данных:")
                st.dataframe(universal_txn.head(10), use_container_width=True, height=240)
            confirm_disabled = not schema_valid
            if st.button(
                "Подтвердить колонки и продолжить",
                type="primary",
                use_container_width=True,
                disabled=confirm_disabled,
                key="confirm_upload_columns",
            ):
                st.session_state["upload_columns_confirmed"] = True
                st.rerun()
            if confirm_disabled:
                st.caption("Чтобы продолжить, сопоставьте обязательные колонки и исправьте ошибки качества данных.")
        close_surface()

    else:
        open_surface("Запустите анализ")
        st.caption("Базовый прогноз будет построен для выбранного SKU. После анализа вы перейдёте в workspace и сможете проверить бизнес-решение.")
        if st.button("Изменить сопоставление колонок", use_container_width=True, key="back_to_upload_columns"):
            st.session_state["upload_columns_confirmed"] = False
            st.rerun()
        if raw_for_select is not None and len(raw_for_select) > 0:
            category_col = "product_category_name" if "product_category_name" in raw_for_select.columns else "category"
            categories = sorted(raw_for_select[category_col].dropna().astype(str).unique())
            target_category = st.selectbox("Категория", categories, key="input_target_category") if categories else None
            if target_category is not None:
                sku_col = "product_id"
                skus = sorted(raw_for_select[raw_for_select[category_col].astype(str) == str(target_category)][sku_col].astype(str).dropna().unique())
                target_sku = st.selectbox("SKU", skus, key="input_target_sku") if skus else None
        horizon_days = st.slider("Горизонт прогноза, дней", 7, 90, int(CONFIG["HORIZON_DAYS_DEFAULT"]), 1, key="input_horizon_days")
        quality_status = data_quality_ui_label(data_gate.get("status"))
        st.markdown(f"**Качество данных:** {quality_status}")
        if target_category and target_sku:
            st.info(f"Будет рассчитан SKU: **{target_sku}** в категории **{target_category}**.")
        analysis_calc_mode = st.radio(
            "Режим расчёта (3 режима)",
            options=list(SCENARIO_CALC_MODES.keys()),
            format_func=lambda x: SCENARIO_CALC_MODES[x],
            index=list(SCENARIO_CALC_MODES.keys()).index(DEFAULT_SCENARIO_CALC_MODE),
            key="input_analysis_calc_mode",
            horizontal=False,
        )
        st.caption("Доступны 3 режима: базовый, расширенный сценарный и факторная модель. Если не уверены — оставьте режим по умолчанию.")
        auto_run_after_upload = st.checkbox(
            "Запускать расчёт автоматически после загрузки, если в файле один SKU и колонки распознаны",
            value=True,
            help="Если в файле несколько категорий или SKU, сначала нужно выбрать объект анализа, чтобы не запустить расчёт не для того товара.",
            key="input_auto_run_after_upload",
        )
        with st.expander("Дополнительные допущения", expanded=False):
            auto_price_optimizer = st.checkbox(
                "После анализа сразу показать лучший найденный вариант цены",
                value=False,
                help=(
                    "Оптимизатор не меняет расчёт спроса и не применяет цену автоматически. "
                    "Он только проверит несколько цен через текущий what-if механизм и покажет рекомендацию."
                ),
                key="input_auto_price_optimizer",
            )
        can_run_calculation = bool(schema_valid and data_gate.get("usage_policy", {}).get("can_run_calculation") and target_category and target_sku)
        auto_run_possible = bool(can_run_calculation and single_object_ready and auto_run_after_upload)
        auto_run_requested = bool(auto_run_possible and st.session_state.get("upload_auto_started_for_file") != current_file_name)
        if auto_run_requested:
            st.session_state["upload_auto_started_for_file"] = current_file_name
            auto_run_notice = "Файл содержит один SKU, колонки распознаны — запускаем расчёт автоматически."
            st.info(auto_run_notice)
        elif upload_done and schema_valid and auto_run_after_upload and not single_object_ready:
            st.caption("Автозапуск не включён: в файле несколько категорий/SKU, выберите объект анализа и нажмите кнопку запуска.")
        manual_run_requested = st.button("Запустить анализ", type="primary", use_container_width=True, disabled=not can_run_calculation)
        run_requested = bool(auto_run_requested or manual_run_requested)
        close_surface()
    st.markdown('</div>', unsafe_allow_html=True)

    return {
        "universal_txn": universal_txn,
        "schema_valid": schema_valid,
        "target_category": target_category,
        "target_sku": target_sku,
        "run_requested": run_requested,
        "horizon_days": horizon_days,
        "analysis_calc_mode": analysis_calc_mode,
        "auto_price_optimizer": bool(auto_price_optimizer),
        "auto_run_notice": auto_run_notice,
        "data_quality_gate": data_gate,
        "universal_quality": universal_quality,
    }


def build_ui_decision_summary(gate_ok_now: bool, profit_delta_pct: float, reliability_label_explicit: str) -> Dict[str, str]:
    reliability_text_l = str(reliability_label_explicit).strip().lower()
    reliability_risky = any(marker in reliability_text_l for marker in ["низк", "риск", "слаб", "недостат"])
    if not gate_ok_now:
        return {"decision_label": "Требует проверки", "tone": "warning", "reason": "Расчёт требует проверки: есть ограничения качества или параметры сценария выходят за надёжную область."}
    if np.isfinite(profit_delta_pct) and profit_delta_pct < 0:
        return {"decision_label": "Не рекомендуется", "tone": "danger", "reason": "Сценарий снижает прибыль относительно базового прогноза."}
    if np.isfinite(profit_delta_pct) and profit_delta_pct > 0 and reliability_risky:
        return {"decision_label": "Требует проверки", "tone": "warning", "reason": "Экономический эффект положительный, но надёжность расчёта ограничена."}
    if np.isfinite(profit_delta_pct) and profit_delta_pct > 0:
        return {"decision_label": "Можно рассмотреть", "tone": "success", "reason": "Сценарий повышает прибыль относительно базового прогноза без критичных сигналов риска."}
    return {"decision_label": "Требует проверки", "tone": "warning", "reason": "Экономический эффект близок к нейтральному."}


def render_decision_card(result: Dict[str, Any], wr: Dict[str, Any], deltas: Dict[str, float], gate_ok: bool, base_price: float, applied_price: float) -> str:
    profit_pct = float(deltas.get("delta_profit_pct", np.nan))
    confidence = safe_float_or_nan(wr.get("confidence", np.nan))
    if not gate_ok:
        verdict, tone = "Требует проверки", "warning"
    elif np.isfinite(profit_pct) and profit_pct > 3:
        verdict, tone = "Можно рассмотреть", "success"
    elif np.isfinite(profit_pct) and profit_pct < -3:
        verdict, tone = "Сценарий не рекомендуется", "danger"
    else:
        verdict, tone = "Нейтрально / требует проверки", "warning"
    if np.isfinite(confidence):
        conf_label = "Высокая" if confidence >= 0.75 else ("Средняя" if confidence >= 0.5 else "Низкая")
        rel = f"{conf_label} ({confidence * 100:.0f}%)"
    else:
        rel = str(wr.get("confidence_label", "Средняя"))
    if not gate_ok:
        reason = "Расчёт требует проверки: есть ограничения качества или параметры сценария выходят за надёжную область."
    elif abs(float(applied_price) - float(base_price)) < 1e-9 and float(deltas.get("delta_units_pct", np.nan)) < 0 and abs(float(deltas.get("delta_revenue_pct", 0.0)) - float(deltas.get("delta_units_pct", 0.0))) < 2.0:
        reason = "Основная причина — снижение спроса при близкой к исходной цене."
    elif float(deltas.get("scenario_margin_pct", np.nan)) < float(deltas.get("base_margin_pct", np.nan)):
        reason = "Основная причина — снижение маржи в сценарии."
    else:
        reason = "Сценарий меняет спрос, выручку и маржу; проверьте детализацию ниже."
    render_decision_summary_card(
        decision_label=verdict,
        tone=tone,
        reason=reason,
        metrics=[
            {"label": "Прибыль", "value": f"{_fmt_money(deltas.get('base_profit'))} → {_fmt_money(deltas.get('scenario_profit'))}", "delta": f"{_fmt_money(deltas.get('delta_profit'))} / {fmt_pct_delta(profit_pct)}"},
            {"label": "Спрос", "value": f"{_fmt_units(deltas.get('base_units'))} → {_fmt_units(deltas.get('scenario_units'))}", "delta": fmt_pct_delta(deltas.get('delta_units_pct', np.nan))},
            {"label": "Выручка", "value": f"{_fmt_money(deltas.get('base_revenue'))} → {_fmt_money(deltas.get('scenario_revenue'))}", "delta": fmt_pct_delta(deltas.get('delta_revenue_pct', np.nan))},
        ],
        economy_label=str(wr.get("economic_verdict", "—")),
        reliability_label=rel,
    )
    return verdict


def render_simple_reliability_card(wr: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
    conf = safe_float_or_nan(wr.get("confidence", np.nan))
    level = "средняя"
    if np.isfinite(conf):
        level = "высокая" if conf >= 0.75 else ("средняя" if conf >= 0.5 else "низкая")
    if bool(snapshot.get("price_clipped", False)) or bool(wr.get("price_clipped", False)):
        price_reason = "цена ограничена защитным режимом;"
    elif bool(wr.get("extrapolation_applied", False)):
        price_reason = "цена вне истории, применена экстраполяция;"
    else:
        price_reason = "цена в рабочем диапазоне истории;"
    reasons = [price_reason]
    reasons.append("внешний шок задан вручную;" if abs(float(snapshot.get("demand_mult", 1.0)) - 1.0) > 1e-9 else "внешний шок не задан;")
    reasons.append("есть предупреждения качества." if len((wr.get("warnings") or [])) else "критичных предупреждений нет.")
    st.markdown(f"**Надёжность расчёта: {level}**")
    st.markdown("Почему:\n- " + "\n- ".join(reasons))

if __name__ == "__main__":

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
                        scenario_calc_mode=str(ctx.get("analysis_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                        horizon_days=int(ctx.get("horizon_days", CONFIG["HORIZON_DAYS_DEFAULT"])),
                    )
                    if bool(ctx.get("auto_price_optimizer", False)) and not bool(results.get("blocking_error", False)):
                        try:
                            opt_overrides = {}
                            st.session_state.price_optimizer_signature_base = build_price_optimizer_signature(
                                current_price=float(results.get("current_price", 0.0)),
                                horizon_days=int(ctx.get("horizon_days", CONFIG["HORIZON_DAYS_DEFAULT"])),
                                scenario_calc_mode=str(results.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                                price_guardrail_mode=DEFAULT_PRICE_GUARDRAIL_MODE,
                                overrides=opt_overrides,
                                factor_overrides={},
                                freight_multiplier=1.0,
                                demand_multiplier=1.0,
                                candidate_count=25,
                                search_pct=0.20,
                            )
                            st.session_state.price_optimizer_result_base = analyze_price_optimization(
                                trained_bundle=results["_trained_bundle"],
                                current_price=float(results.get("current_price", 0.0)),
                                runner=run_what_if_projection,
                                horizon_days=int(ctx.get("horizon_days", CONFIG["HORIZON_DAYS_DEFAULT"])),
                                scenario_calc_mode=str(results.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                                price_guardrail_mode=DEFAULT_PRICE_GUARDRAIL_MODE,
                                overrides=opt_overrides,
                                factor_overrides={},
                                freight_multiplier=1.0,
                                demand_multiplier=1.0,
                                candidate_count=25,
                                search_pct=0.20,
                            )
                            st.session_state.active_workspace_tab = PAGE_OVERVIEW
                            st.session_state["workspace_tab_radio"] = PAGE_OVERVIEW
                        except Exception as exc:
                            st.session_state.price_optimizer_result_base = {"status": "optimizer_error", "recommendation_title": "Оптимизатор цены не был рассчитан", "recommendation_text": f"Базовый анализ выполнен, но оптимизатор цены завершился ошибкой: {exc}", "warnings": [str(exc)], "candidates": pd.DataFrame()}
                            st.session_state.price_optimizer_signature_base = None
                            st.session_state.price_optimizer_context = "base"
                    st.session_state.results = dict(results)
                    st.session_state["what_if_calc_mode"] = str(results.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
                    st.session_state.selected_category_for_results = ctx["target_category"]
                    st.session_state.selected_sku_for_results = ctx["target_sku"]
                    st.session_state.app_stage = "dashboard"
                    st.rerun()
        st.stop()

    r = st.session_state.results
    if bool(r.get("blocking_error", False)):
        st.error(str(r.get("blocking_error_message", "Расчёт остановлен из-за блокирующей ошибки.")))
        if r.get("blocking_error_code"):
            st.caption(f"Код ошибки: {r.get('blocking_error_code')}")
        for w in r.get("warnings", []):
            st.warning(str(w))
        st.stop()
    history_daily = r["history_daily"]
    current_forecast = r["as_is_forecast"]
    baseline_forecast = r["neutral_baseline_forecast"]
    has_applied_scenario = r.get("scenario_forecast") is not None and st.session_state.what_if_result is not None
    scenario_forecast = r["scenario_forecast"] if has_applied_scenario else current_forecast
    last_update = pd.Timestamp.utcnow().strftime("%d.%m.%Y")
    ui_status = st.session_state.get("scenario_ui_status", "as_is")
    status_map = {
        "as_is": ("База без изменений", "muted"),
        "dirty": ("Есть неприменённые изменения", "warning"),
        "applied": ("Сценарий применён", "success"),
        "saved": (f"Сценарий сохранён: {SCENARIO_SLOT_LABELS.get(st.session_state.get('last_saved_slot', ''), st.session_state.get('last_saved_slot', 'вариант'))}", "success"),
    }
    status_text, status_color = status_map.get(ui_status, ("База без изменений", "muted"))

    back_to_landing = render_object_header(
        object_title=str(st.session_state.get("selected_sku_for_results", "SKU")),
        status_text=status_text,
        horizon_text=f"{len(current_forecast)} дней",
        last_update=last_update,
        status_color=status_color,
    )
    if back_to_landing:
        _set_page("landing")
    action_click = render_action_row(
        has_applied_scenario=has_applied_scenario,
        has_saved_scenarios=bool(st.session_state.get("saved_scenarios")),
    )
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
        st.session_state["what_if_freight_change_pct"] = 0.0
        st.session_state["what_if_demand_shock_pct"] = 0.0
        st.session_state["what_if_hdays"] = int(len(r.get("as_is_forecast", pd.DataFrame())) or CONFIG["HORIZON_DAYS_DEFAULT"])
        st.session_state["what_if_calc_mode"] = str(r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
        st.session_state["what_if_use_segments"] = False
        st.toast("Форма сброшена к базовым значениям", icon="⟲")
        st.rerun()
    elif action_click == "cancel_active":
        r = reset_scenario_ui_state_to_base(r, clear_saved_slot=False)
        st.session_state.results = r
        st.toast("Активный сценарий отменён", icon="⊘")
        st.rerun()
    elif action_click == "decision":
        st.session_state.active_workspace_tab = PAGE_DECISION
        st.session_state["workspace_tab_radio"] = PAGE_DECISION
        st.rerun()
    elif action_click == "scenario":
        st.session_state.active_workspace_tab = PAGE_WHAT_IF
        st.session_state["workspace_tab_radio"] = PAGE_WHAT_IF
        st.rerun()
    elif action_click == "compare":
        st.session_state.active_workspace_tab = PAGE_COMPARE
        st.session_state["workspace_tab_radio"] = PAGE_COMPARE
        st.rerun()
    elif action_click == "export":
        st.session_state.active_workspace_tab = PAGE_REPORT
        st.session_state["workspace_tab_radio"] = PAGE_REPORT
        st.rerun()

    tabs = WORKSPACE_PAGES
    current_tab = normalize_workspace_page(st.session_state.get("active_workspace_tab", PAGE_OVERVIEW))
    radio_tab = normalize_workspace_page(st.session_state.get("workspace_tab_radio", current_tab))
    if current_tab not in tabs:
        current_tab = PAGE_OVERVIEW
    if radio_tab not in tabs:
        radio_tab = current_tab
    st.session_state.active_workspace_tab = current_tab
    st.session_state["workspace_tab_radio"] = radio_tab
    active_tab = render_tabs(current_tab, tabs, key="workspace_tab_radio")
    st.session_state.active_workspace_tab = active_tab
    if active_tab == PAGE_WHAT_IF:
        render_workspace_guide(
            active_tab=active_tab,
            has_applied_scenario=has_applied_scenario,
            has_saved_scenarios=bool(st.session_state.get("saved_scenarios")),
            has_decision_analysis=bool(
                st.session_state.get("decision_passport") or st.session_state.get("recommendation_audit_result")
            ),
        )

    current_forecast_for_compare = align_forecasts_by_scenario_dates(current_forecast, scenario_forecast)
    deltas = calculate_scenario_deltas(current_forecast_for_compare, scenario_forecast)
    base_units = float(deltas["base_units"])
    base_revenue = float(deltas["base_revenue"])
    base_profit = float(deltas["base_profit"])
    sc_units = float(deltas["scenario_units"])
    sc_revenue = float(deltas["scenario_revenue"])
    sc_profit = float(deltas["scenario_profit"])

    if active_tab == PAGE_OVERVIEW:
        render_page_header(PAGE_OVERVIEW, "Проверьте бизнес-решение до запуска")
        render_overview_hero(PRODUCT_PROMISE, PRODUCT_SUBTITLE)
        open_surface("Главная задача системы")
        st.markdown("""
Проверить бизнес-решение до запуска и показать:
1. можно ли запускать;
2. какой эффект по спросу, выручке и прибыли;
3. какой риск;
4. как безопасно проверить через пилот.
""")
        st.markdown('<div class="action-card-grid">', unsafe_allow_html=True)
        st.markdown('<div class="action-card"><div class="card-title">1. Найти лучшее решение</div><div class="muted">Система сама переберёт допустимые варианты цены, скидки, промо и логистики.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-card"><div class="card-title">2. Проверить мою идею</div><div class="muted">Вы уже знаете, что хотите сделать — система проверит эффект и риск.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-card"><div class="card-title">3. Быстрый what-if</div><div class="muted">Для ручной проверки одного сценария без полного анализа решения.</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Главное действие — «Проверить решение». What-if нужен как быстрый вспомогательный расчёт.")
        close_surface()
        mode = str(r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
        quality_raw = r.get("data_quality_label") or (r.get("data_quality_gate") or {}).get("status")
        quality_label = data_quality_ui_label(quality_raw)
        open_surface("Что уже рассчитано")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Товар/SKU", str(st.session_state.get("selected_sku_for_results", "SKU")))
        o2.metric("Горизонт прогноза", f"{len(current_forecast)} дней")
        o3.metric("Режим анализа", scenario_mode_label(mode))
        o4.metric("Качество данных", quality_label)
        close_surface()
        base_margin = (base_profit / max(base_revenue, 1e-9)) * 100 if base_revenue else float("nan")
        sc_margin = (sc_profit / max(sc_revenue, 1e-9)) * 100 if sc_revenue else float("nan")
        render_kpi_strip([
            {"label": "Спрос", "value": _fmt_units(base_units), "delta": "", "base": "Текущий план"},
            {"label": "Выручка", "value": _fmt_money(base_revenue), "delta": "", "base": "Текущий план"},
            {"label": "Прибыль", "value": _fmt_money(base_profit), "delta": "", "base": "Текущий план"},
            {"label": "Надёжность", "value": quality_label, "delta": "", "base": "Оценка"},
        ])
        open_surface("Что хотите сделать?")
        st.markdown("Система уже построила базовый прогноз. Выберите следующий шаг:")
        d1, d2, d3 = st.columns([0.5, 0.25, 0.25])
        if d1.button("Найти / проверить решение", key="overview_go_decision", type="primary", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_DECISION
            st.session_state["workspace_tab_radio"] = PAGE_DECISION
            st.rerun()
        if d2.button("Быстрый what-if", key="overview_go_what_if", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_WHAT_IF
            st.session_state["workspace_tab_radio"] = PAGE_WHAT_IF
            st.rerun()
        if d3.button("Подобрать цену", key="overview_go_price", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_PRICE
            st.session_state["workspace_tab_radio"] = PAGE_PRICE
            st.rerun()
        close_surface()
        open_surface("Что система выдаст")
        st.markdown("- вердикт: запускать, пилотировать или пересмотреть условия\n- влияние на спрос, выручку и прибыль\n- риск и ограничения\n- план пилота или проверки")
        close_surface()
        if has_applied_scenario:
            render_help_callout("Главный вывод", "Сценарий рассчитан. Ниже показаны решение, экономика, надёжность и следующий шаг.", "success")
            wr = st.session_state.what_if_result or {}
            gate_ok = bool((wr.get("validation_gate", {}) or {}).get("ok", True))
            snapshot = st.session_state.applied_scenario_snapshot or {}
            base_price = float(r.get("current_price", 0.0))
            applied = float(wr.get("applied_price_gross", snapshot.get("manual_price_applied", base_price)))
            requested = float(wr.get("requested_price", snapshot.get("manual_price_input", base_price)))
            verdict = render_decision_card(r, wr, deltas, gate_ok, base_price=base_price, applied_price=applied)
            open_surface("Что проверяем")
            guard_mode = normalize_price_guardrail_mode(wr.get("price_guardrail_mode", snapshot.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)))
            if bool(wr.get("extrapolation_applied", False)):
                st.markdown(f"**Цена:** запрошено {fmt_price(requested)}, расчёт выполнен через экстраполяцию от безопасной границы {fmt_price(applied)}. Надёжность ниже.")
            elif bool(wr.get("price_clipped", snapshot.get("price_clipped", False))) and guard_mode == PRICE_GUARDRAIL_SAFE_CLIP:
                st.markdown(f"**Цена:** запрошено {fmt_price(requested)}, для модели применено {fmt_price(applied)}. Причина: защитный режим, цена вне исторического диапазона.")
            else:
                st.markdown(f"**Цена:** {'без изменений — ' + fmt_price(base_price) if abs(applied-base_price)<1e-9 else fmt_price(base_price) + ' → ' + fmt_price(applied)}")
            st.markdown(f"**Скидка:** {float(wr.get('applied_discount', snapshot.get('discount_applied', 0.0))) * 100:.0f}%")
            promo = float(wr.get("promotion", snapshot.get("promo_applied", 0.0)))
            st.markdown(f"**Промо:** {'нет' if promo <= 0 else f'{promo * 100:.0f}%'}")
            shock_mult = float(wr.get("manual_shock_multiplier", snapshot.get("demand_mult", 1.0)))
            st.markdown(f"**Внешний спрос:** {fmt_pct_delta((shock_mult - 1.0) * 100)}")
            st.markdown(f"**Период:** {int(snapshot.get('horizon_days', len(r.get('scenario_forecast', []))))} дней")
            calc_mode = str(wr.get("scenario_calc_mode", r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)))
            close_surface()
        with st.expander("Детали прогноза и графики", expanded=False):
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=current_forecast_for_compare["date"], y=current_forecast_for_compare["actual_sales"], name="База", line=dict(color=PLOT_SUCCESS, width=2)))
            fig_d.update_layout(**_plot_layout("Спрос"))
            render_chart_card("Спрос", "Горизонт · шт", fig_d, [("Базовый прогноз", f"{base_units:,.0f}")])
            _render_chart_legend_help("Спрос", "шт")

            if has_applied_scenario and isinstance(r.get("scenario_forecast"), pd.DataFrame):
                sf = r.get("scenario_forecast").copy()
                bf = align_forecasts_by_scenario_dates(current_forecast, sf)
                def pick_demand_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
                    for col in preferred:
                        if col in df.columns:
                            return col
                    return None
                base_col = pick_demand_col(bf, ["actual_sales", "as_is_demand", "baseline_demand"])
                sc_col = pick_demand_col(sf, ["actual_sales", "scenario_demand", "predicted_sales"])
                if base_col is None or sc_col is None:
                    st.info("График сценария недоступен: не найдена колонка спроса.")
                else:
                    delta = sf[sc_col].values - bf[base_col].values
                    trend_text = "Сценарий ниже текущего плана большую часть периода" if float(np.nansum(delta)) < 0 else "Сценарий выше текущего плана большую часть периода"
                    st.caption(trend_text)
                    fig_bs = go.Figure()
                    fig_bs.add_trace(go.Scatter(x=sf["date"], y=bf[base_col], name="Текущий план", line=dict(color=PLOT_MUTED, width=2)))
                    fig_bs.add_trace(go.Scatter(x=sf["date"], y=sf[sc_col], name="Сценарий", line=dict(color=PLOT_ACCENT, width=2), fill="tonexty", fillcolor=PLOT_ACCENT_FILL))
                    fig_bs.update_layout(**_plot_layout("Сценарий vs план"))
                    st.plotly_chart(fig_bs, use_container_width=True, config={"displayModeBar": False})
                with st.expander("Показать детализацию изменения прибыли", expanded=False):
                    wf = go.Figure(go.Waterfall(measure=["absolute", "relative", "total"], x=["Текущий план", "Изменение", "Сценарий"], y=[base_profit, sc_profit - base_profit, sc_profit]))
                    wf.update_layout(**_plot_layout("Детализация прибыли"))
                    st.plotly_chart(wf, use_container_width=True, config={"displayModeBar": False})
                open_surface("Почему изменился прогноз")
                if str(wr.get("scenario_calc_mode", r.get("analysis_scenario_calc_mode", ""))) == CATBOOST_FULL_FACTOR_MODE or str(wr.get("scenario_price_effect_source", "")) == "catboost_full_factor_reprediction":
                    st.markdown(f"Факторная модель пересчитала спрос целиком по изменённым параметрам.\n\n- внешний спрос: {fmt_pct_delta((shock_mult - 1.0) * 100)}\n- цена: {'без изменений' if abs(applied-base_price)<1e-9 else 'изменилась'}\n- скидка: {float(wr.get('applied_discount',0.0))*100:.0f}%\n- промо: {'нет' if promo<=0 else f'{promo*100:.0f}%'}")
                else:
                    breakdown = (wr.get("effect_breakdown") or wr.get("effects") or {})
                    if breakdown:
                        render_human_effect_breakdown(breakdown)
                    else:
                        st.info("Подробное разложение эффектов недоступно для этого сценария.")
                close_surface()
                open_surface("Надёжность расчёта")
                render_simple_reliability_card(wr, snapshot)
                close_surface()
                with st.expander("Для аналитика / технические детали", expanded=False):
                    render_reliability_card(wr, r)
        open_surface("Следующий шаг")
        a1, a2, a3 = st.columns(3)
        if a1.button(PAGE_DECISION, key="go_decision_from_summary", type="primary", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_DECISION
            st.session_state["workspace_tab_radio"] = PAGE_DECISION
            st.rerun()
        if a2.button("Быстрый what-if", key="go_scenario_from_summary", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_WHAT_IF
            st.session_state["workspace_tab_radio"] = PAGE_WHAT_IF
            st.rerun()
        if has_applied_scenario and a3.button(PAGE_REPORT, key="go_report_from_summary", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_REPORT
            st.session_state["workspace_tab_radio"] = PAGE_REPORT
            st.rerun()
        close_surface()


    elif active_tab == PAGE_PRICE:
        render_page_header(PAGE_PRICE, "Система проверяет несколько цен через текущий what-if механизм и показывает лучший найденный вариант. Цена не применяется автоматически.")
        render_help_callout("Как работает подбор цены", "Система проверит несколько цен через текущий what-if механизм и покажет лучший найденный вариант. Цена не применяется автоматически.", "info")
        current_factor_overrides = factor_overrides if "factor_overrides" in locals() else {}
        base_opt_overrides = {
            "promotion": float(st.session_state.get("what_if_promo", r.get("_trained_bundle", {}).get("base_ctx", {}).get("promotion", 0.0))),
            "discount": float(st.session_state.get("what_if_discount", r.get("_trained_bundle", {}).get("base_ctx", {}).get("discount", 0.0))),
        }
        open_surface("Перед запуском")
        st.markdown("Система проверит несколько цен рядом с текущей и покажет, какая выглядит лучше по прибыли. Цена не применяется автоматически.")
        p1, p2, p3 = st.columns(3)
        p1.metric("Текущая цена", fmt_price(r.get("current_price", np.nan)))
        p2.metric("Период проверки", f"{int(st.session_state.get('what_if_hdays', CONFIG['HORIZON_DAYS_DEFAULT']))} дней")
        p3.metric("Условия", "текущий сценарий" if st.session_state.get("what_if_result") is not None else "базовый прогноз")
        close_surface()
        current_opt_signature = build_price_optimizer_signature(
            current_price=float(r.get("current_price", 0.0)),
            horizon_days=int(st.session_state.get("what_if_hdays", CONFIG["HORIZON_DAYS_DEFAULT"])),
            scenario_calc_mode=str(st.session_state.get("what_if_calc_mode", r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))),
            price_guardrail_mode=normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
            overrides=base_opt_overrides,
            factor_overrides=current_factor_overrides,
            freight_multiplier=float(st.session_state.get("what_if_freight_mult", 1.0)),
            demand_multiplier=float(st.session_state.get("what_if_demand_mult", 1.0)),
            candidate_count=25,
            search_pct=0.20,
        )
        st.caption("Цель подбора: найти вариант цены с лучшей ожидаемой прибылью в безопасном диапазоне рядом с текущей ценой. Цена не применяется автоматически.")
        if st.button("Подобрать цену", type="primary", use_container_width=True, key="run_price_optimizer_btn"):
            st.session_state.price_optimizer_result_base = analyze_price_optimization(
                trained_bundle=r["_trained_bundle"],
                current_price=float(r.get("current_price", 0.0)),
                runner=run_what_if_projection,
                horizon_days=int(st.session_state.get("what_if_hdays", CONFIG["HORIZON_DAYS_DEFAULT"])),
                scenario_calc_mode=str(st.session_state.get("what_if_calc_mode", r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))),
                price_guardrail_mode=normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
                overrides=base_opt_overrides,
                factor_overrides=current_factor_overrides,
                freight_multiplier=float(st.session_state.get("what_if_freight_mult", 1.0)),
                demand_multiplier=float(st.session_state.get("what_if_demand_mult", 1.0)),
                candidate_count=25,
                search_pct=0.20,
            )
            st.session_state.price_optimizer_signature_base = build_price_optimizer_signature(
                current_price=float(r.get("current_price", 0.0)),
                horizon_days=int(st.session_state.get("what_if_hdays", CONFIG["HORIZON_DAYS_DEFAULT"])),
                scenario_calc_mode=str(st.session_state.get("what_if_calc_mode", r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))),
                price_guardrail_mode=normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
                overrides=base_opt_overrides,
                factor_overrides=current_factor_overrides,
                freight_multiplier=float(st.session_state.get("what_if_freight_mult", 1.0)),
                demand_multiplier=float(st.session_state.get("what_if_demand_mult", 1.0)),
                candidate_count=25,
                search_pct=0.20,
            )
            st.session_state.price_optimizer_context = "base"
        opt = st.session_state.get("price_optimizer_result_base")
        is_price_optimizer_stale = st.session_state.get("price_optimizer_signature_base") != current_opt_signature
        if is_price_optimizer_stale and opt is not None:
            st.warning("Рекомендация по цене устарела: параметры сценария изменились. Нажмите «Подобрать цену» ещё раз.")
        render_price_optimizer_summary(opt or {})
        st.caption("Контекст: рекомендация рассчитана для параметров, указанных в разделе «What-if». Цена не применяется автоматически.")
        if isinstance(opt, dict):
            rp = opt.get("recommended_price")
            can_apply = (
                opt.get("status") in ACTIONABLE_PRICE_OPT_STATUSES
                and rp is not None
                and np.isfinite(float(rp))
                and (not is_price_optimizer_stale)
            )
            if not can_apply:
                st.caption("Недоступно: сначала проверьте цены или обновите устаревший результат.")
            if st.button("Проверить как решение", use_container_width=True, disabled=not can_apply, key="apply_price_optimizer_to_what_if"):
                st.session_state["audit_action"] = "Изменить цену"
                st.session_state["audit_target_price"] = float(rp)
                st.session_state["decision_mode"] = "audit_idea"
                st.session_state.active_workspace_tab = PAGE_DECISION
                st.session_state["workspace_tab_radio"] = PAGE_DECISION
                st.toast("Цена перенесена в проверку решения.", icon="✓")
                st.rerun()
            with st.expander("Для аналитика: все проверенные цены", expanded=False):
                render_price_optimizer_chart(opt)
                render_price_optimizer_table(opt)
    elif active_tab == PAGE_WHAT_IF:
        render_page_header(PAGE_WHAT_IF, "Быстро проверьте, как изменение цены, скидки, промо или внешнего спроса повлияет на спрос, выручку и прибыль.")
        st.info("Основной путь: 1) что меняем, 2) что получится, 3) рассчитать сценарий. Изменения не применяются автоматически." )
        base_ctx = r["_trained_bundle"]["base_ctx"]
        open_surface("От чего считаем")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Цена", fmt_price(r.get("current_price", np.nan)))
        c2.metric("Спрос", _fmt_units(base_units))
        c3.metric("Выручка", _fmt_money(base_revenue))
        c4.metric("Прибыль", _fmt_money(base_profit))
        close_surface()
        current_analysis_mode = str(r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
        analysis_mode_key = f"{st.session_state.get('selected_sku_for_results', '')}:{current_analysis_mode}"
        if st.session_state.get("what_if_calc_mode_initialized_for") != analysis_mode_key:
            st.session_state["what_if_calc_mode"] = current_analysis_mode
            st.session_state["what_if_calc_mode_initialized_for"] = analysis_mode_key

        base_form = collect_current_form_values(
            float(r.get("current_price", 0.0)),
            float(base_ctx.get("discount", 0.0)),
            float(np.clip(base_ctx.get("promotion", 0.0), 0.0, 1.0)),
            1.0,
            1.0,
            int(len(r.get("as_is_forecast", pd.DataFrame())) or CONFIG["HORIZON_DAYS_DEFAULT"]),
            current_analysis_mode,
            DEFAULT_PRICE_GUARDRAIL_MODE,
        )
        for key, val in {
            "what_if_price": base_form["manual_price"],
            "what_if_discount": base_form["discount"],
            "what_if_promo": base_form["promo_value"],
            "what_if_freight_mult": base_form["freight_mult"],
            "what_if_demand_mult": base_form["demand_mult"],
            "what_if_hdays": base_form["hdays"],
            "what_if_calc_mode": base_form["scenario_calc_mode"],
        }.items():
            if key not in st.session_state:
                st.session_state[key] = val
        st.session_state["what_if_freight_change_pct"] = float(st.session_state.get("what_if_freight_change_pct", multiplier_to_pct(st.session_state.get("what_if_freight_mult", 1.0))))
        st.session_state["what_if_demand_shock_pct"] = float(st.session_state.get("what_if_demand_shock_pct", multiplier_to_pct(st.session_state.get("what_if_demand_mult", 1.0))))
        freight_form_mult = float(pct_to_multiplier(st.session_state.get("what_if_freight_change_pct", 0.0)))
        demand_form_mult = float(pct_to_multiplier(st.session_state.get("what_if_demand_shock_pct", 0.0)))
        current_form_top = collect_current_form_values(
            float(st.session_state.get("what_if_price", base_form["manual_price"])),
            float(st.session_state.get("what_if_discount", base_form["discount"])),
            float(st.session_state.get("what_if_promo", base_form["promo_value"])),
            freight_form_mult,
            demand_form_mult,
            int(st.session_state.get("what_if_hdays", base_form["hdays"])),
            str(st.session_state.get("what_if_calc_mode", base_form["scenario_calc_mode"])),
            normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
        )
        st.session_state.form_last_values = current_form_top
        st.session_state.scenario_ui_status = get_user_scenario_status(
            current_form_top,
            base_form,
            st.session_state.applied_scenario_snapshot,
            st.session_state.get("scenario_ui_status", "as_is"),
        )
        render_scenario_status_banner(
            st.session_state.scenario_ui_status,
            st.session_state.applied_scenario_snapshot,
            st.session_state.last_saved_slot,
            selected_mode=str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
        )
        base_total_demand = float(pd.to_numeric(current_forecast.get("actual_sales", 0.0), errors="coerce").fillna(0.0).sum()) if len(current_forecast) else 0.0
        base_total_revenue = float(pd.to_numeric(current_forecast.get("revenue", 0.0), errors="coerce").fillna(0.0).sum()) if len(current_forecast) else 0.0
        base_total_profit = float(pd.to_numeric(current_forecast.get("profit", 0.0), errors="coerce").fillna(0.0).sum()) if len(current_forecast) else 0.0
        with st.expander("Детали базового прогноза", expanded=False):
            b1, b2, b3 = st.columns(3)
            b1.metric("Текущая скидка", fmt_pct_abs(float(base_ctx.get("discount", 0.0)) * 100.0))
            b2.metric("Текущее промо", fmt_pct_abs(float(base_ctx.get("promotion", 0.0)) * 100.0))
            b3.metric("Период", f"{len(current_forecast)} дней")
        with st.expander("Готовые пресеты (необязательно)", expanded=False):
            st.caption("Выберите готовый вариант, затем нажмите «Рассчитать сценарий».")
            primary_preset_buttons = [
                ("Скидка 5%", "discount_5"),
                ("Скидка 10%", "discount_10"),
                ("Цена +5%", "price_plus_5"),
                ("Цена -5%", "price_minus_5"),
                ("Промо", "promo"),
            ]
            extra_preset_buttons = [
                ("Логистика +10%", "freight_plus_10"),
                ("Спрос -10%", "demand_minus_10"),
                ("Спрос +10%", "demand_plus_10"),
            ]
            preset_cols = st.columns(5)
            preset_clicked = None
            for idx, (label, code) in enumerate(primary_preset_buttons):
                with preset_cols[idx % 5]:
                    if st.button(label, key=f"preset_{code}", use_container_width=True):
                        preset_clicked = code
            with st.expander("Ещё варианты", expanded=False):
                extra_cols = st.columns(3)
                for idx, (label, code) in enumerate(extra_preset_buttons):
                    with extra_cols[idx % 3]:
                        if st.button(label, key=f"preset_{code}", use_container_width=True):
                            preset_clicked = code
            if st.button("Сбросить к базовому сценарию", key="preset_reset_to_base", use_container_width=True):
                preset_clicked = "reset"
            if preset_clicked:
                current_price_value = float(r.get("current_price", 0.0))
                st.session_state["what_if_price"] = current_price_value
                st.session_state["what_if_discount"] = 0.0
                st.session_state["what_if_promo"] = 0.0
                st.session_state["what_if_freight_change_pct"] = 0.0
                st.session_state["what_if_demand_shock_pct"] = 0.0
                st.session_state["what_if_freight_mult"] = 1.0
                st.session_state["what_if_demand_mult"] = 1.0
                st.session_state["what_if_hdays"] = int(len(r.get("as_is_forecast", pd.DataFrame())) or CONFIG["HORIZON_DAYS_DEFAULT"])
                st.session_state["what_if_use_segments"] = False
                if preset_clicked == "discount_5":
                    st.session_state["what_if_discount"] = 0.05
                elif preset_clicked == "discount_10":
                    st.session_state["what_if_discount"] = 0.10
                elif preset_clicked == "price_plus_5":
                    st.session_state["what_if_price"] = current_price_value * 1.05
                elif preset_clicked == "price_minus_5":
                    st.session_state["what_if_price"] = current_price_value * 0.95
                elif preset_clicked == "promo":
                    st.session_state["what_if_promo"] = 0.20
                elif preset_clicked == "freight_plus_10":
                    st.session_state["what_if_freight_change_pct"] = 10.0
                    st.session_state["what_if_freight_mult"] = 1.10
                elif preset_clicked == "demand_minus_10":
                    st.session_state["what_if_demand_shock_pct"] = -10.0
                    st.session_state["what_if_demand_mult"] = 0.90
                elif preset_clicked == "demand_plus_10":
                    st.session_state["what_if_demand_shock_pct"] = 10.0
                    st.session_state["what_if_demand_mult"] = 1.10
                st.session_state.scenario_ui_status = "as_is" if preset_clicked == "reset" else "dirty"
                st.rerun()


        st.markdown('<div class="scenario-shell">', unsafe_allow_html=True)
        open_surface("Быстрый what-if", "Основной путь для бизнеса: период, цена, скидка, промо и внешний спрос. Аналитические настройки скрыты ниже.")
        if st.session_state["what_if_promo"] > 0.70:
            st.session_state["what_if_promo"] = 0.70
        if "what_if_freight_change_pct" not in st.session_state:
            st.session_state["what_if_freight_change_pct"] = float(multiplier_to_pct(st.session_state.get("what_if_freight_mult", 1.0)))
        if "what_if_demand_shock_pct" not in st.session_state:
            st.session_state["what_if_demand_shock_pct"] = float(multiplier_to_pct(st.session_state.get("what_if_demand_mult", 1.0)))
        st.markdown('<div class="scenario-grid">', unsafe_allow_html=True)
        left_col, right_col = st.columns([1.2, 0.8])
        with left_col:
            with st.form("scenario_form"):
                st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
                st.markdown('<div class="scenario-card-header"><div><div class="scenario-card-title">1. Что меняем?</div><div class="scenario-card-caption">На этот период будут пересчитаны спрос, выручка и прибыль.</div></div><div class="scenario-step">1</div></div>', unsafe_allow_html=True)
                hdays = st.slider("На сколько дней считаем сценарий", 7, 90, key="what_if_hdays", step=1)
                st.markdown('<div class="preview-spacer"></div>', unsafe_allow_html=True)
                st.markdown('<div class="scenario-card-header"><div><div class="scenario-card-title">Цена и скидка</div><div class="scenario-card-caption">Цена для расчёта спроса = новая цена × (1 − скидка).</div></div><div class="scenario-step">1</div></div>', unsafe_allow_html=True)
                st.metric("Текущая цена", fmt_price(r["current_price"]))
                manual_price = st.number_input("Новая цена до скидки, ₽", min_value=0.01, step=1.0, key="what_if_price")
                discount = percent_slider_to_share("Скидка, %", "what_if_discount", min_pct=0, max_pct=95, step=1, default_share=0.0)
                st.caption(f"Цена после скидки: {fmt_price(manual_price * (1.0 - discount))}")
                st.markdown('<div class="preview-spacer"></div>', unsafe_allow_html=True)
                st.markdown('<div class="scenario-card-header"><div><div class="scenario-card-title">Промо и внешний спрос</div><div class="scenario-card-caption">Внешний спрос — бизнес-гипотеза: реклама, сезонность, рынок или конкуренты.</div></div><div class="scenario-step">1</div></div>', unsafe_allow_html=True)
                promo_value = percent_slider_to_share("Промо-поддержка, %", "what_if_promo", min_pct=0, max_pct=70, step=1, default_share=0.0)
                demand_shock_pct = st.slider("Внешний спрос, %", -30.0, 30.0, key="what_if_demand_shock_pct", step=1.0, help="Ручная бизнес-гипотеза: реклама, сезонность, рынок, дефицит или действия конкурентов.")
                st.markdown('<div class="preview-spacer"></div>', unsafe_allow_html=True)
                st.markdown('<div class="scenario-card-header"><div><div class="scenario-card-title">2. Что получится?</div><div class="scenario-card-caption">Справа показан предварительный контекст: спрос, выручка, прибыль и риск обновятся после расчёта.</div></div><div class="scenario-step">2</div></div>', unsafe_allow_html=True)
                with st.expander("Дополнительно: логистика", expanded=False):
                    st.markdown("**Логистика**")
                    freight_change_pct = st.slider("Изменение затрат на логистику, %", -50.0, 50.0, key="what_if_freight_change_pct", step=1.0)
                st.caption(f"Допущения: логистика {freight_change_pct:+.0f}%, внешний спрос {demand_shock_pct:+.0f}%")
                freight_mult = pct_to_multiplier(freight_change_pct)
                demand_mult = pct_to_multiplier(demand_shock_pct)
                st.session_state["what_if_freight_mult"] = float(freight_mult)
                st.session_state["what_if_demand_mult"] = float(demand_mult)
                with st.expander("Для аналитика: режим расчёта и защитные настройки", expanded=False):
                    st.caption("Откройте только для проверки технических допущений: режим расчёта, защитный режим цены, дополнительные факторы или сегменты периода.")
                    st.markdown("**Текущий режим расчёта**")
                    st.caption("Режим выбран перед запуском анализа и в этом разделе не меняется.")
                    current_analysis_mode = str(r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))
                    st.session_state["what_if_calc_mode"] = str(
                        st.session_state.get("what_if_calc_mode", current_analysis_mode)
                    )
                    st.session_state["what_if_calc_mode"] = current_analysis_mode
                    st.caption(f"Текущий режим расчёта: {scenario_mode_label(current_analysis_mode)}")
                    if st.session_state["what_if_calc_mode"] == CATBOOST_FULL_FACTOR_MODE:
                        st.caption(
                            "В этом режиме факторная модель повторно прогнозирует спрос "
                            "по изменённым факторам. Ручная поправка спроса применяется отдельно."
                        )
                    else:
                        st.caption("Метод: базовый прогноз + сценарный пересчёт факторов.")
                    st.caption("Подробные технические сведения доступны на странице диагностики.")
                    scenario_calc_mode = str(st.session_state.get("what_if_calc_mode", r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE)))
                    st.markdown("**Что делать, если цена вне прошлой истории**")
                    if scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE:
                        st.radio(
                            "Что делать, если цена вне прошлой истории?",
                            options=[PRICE_GUARDRAIL_SAFE_CLIP, PRICE_GUARDRAIL_EXTRAPOLATE],
                            format_func=lambda x: {
                                PRICE_GUARDRAIL_SAFE_CLIP: "Осторожно: считать по ближайшей безопасной цене",
                                PRICE_GUARDRAIL_EXTRAPOLATE: "Проверить мою цену: оценить риск за пределами истории",
                            }[x],
                            index=0 if normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)) == PRICE_GUARDRAIL_SAFE_CLIP else 1,
                            help="Осторожный режим снижает риск ошибки, если новая цена сильно отличается от прошлых цен. Второй режим сохраняет введённую цену для экономики сценария, но оценка спроса становится менее надёжной.",
                            key="price_guardrail_mode",
                        )
                    else:
                        st.session_state["price_guardrail_mode"] = DEFAULT_PRICE_GUARDRAIL_MODE
                    factor_overrides: Dict[str, Any] = {}
                    if scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE:
                        cb_bundle_ui = ((r.get("_trained_bundle", {}) or {}).get("catboost_full_factor_bundle", {}) or {})
                        factor_catalog_ui = cb_bundle_ui.get("factor_catalog", pd.DataFrame()) if isinstance(cb_bundle_ui, dict) else pd.DataFrame()
                        st.markdown("**Дополнительные факторы из загруженных данных**")
                        if isinstance(factor_catalog_ui, pd.DataFrame) and len(factor_catalog_ui):
                            if "editable" not in factor_catalog_ui.columns:
                                factor_catalog_ui["editable"] = True
                            editable = factor_catalog_ui[
                                factor_catalog_ui["feature"].astype(str).str.startswith("factor__")
                                & factor_catalog_ui["editable"].astype(bool)
                            ].copy()
                            if len(editable) == 0:
                                st.caption("Дополнительные факторы для ручного изменения не найдены.")
                            else:
                                st.caption("Эти параметры нужны, если вы хотите вручную изменить дополнительные факторы, которые были в загруженных данных. Ниже можно задать значения для сценария.")
                                for _, fr in editable.iterrows():
                                    feature_name = str(fr.get("feature"))
                                    ui_key = f"what_if_factor_{feature_name.replace('factor__', '').replace(' ', '_')}"
                                    dtype = str(fr.get("dtype", "numeric"))
                                    if dtype == "numeric":
                                        default_val = float(pd.to_numeric(fr.get("current_value", fr.get("fill_value", 0.0)), errors="coerce"))
                                        lo = pd.to_numeric(fr.get("train_min", np.nan), errors="coerce")
                                        hi = pd.to_numeric(fr.get("train_max", np.nan), errors="coerce")
                                        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                                            val = st.slider(
                                                f"{feature_name}",
                                                min_value=float(lo),
                                                max_value=float(hi),
                                                value=float(np.clip(default_val, float(lo), float(hi))),
                                                key=ui_key,
                                            )
                                        else:
                                            val = st.number_input(f"{feature_name}", value=float(default_val), key=ui_key)
                                    else:
                                        choices = []
                                        if feature_name in r["_trained_bundle"]["daily_base"].columns:
                                            choices = sorted(r["_trained_bundle"]["daily_base"][feature_name].dropna().astype(str).unique().tolist())
                                        if not choices:
                                            choices = [str(fr.get("current_value", "unknown"))]
                                        default_choice = str(fr.get("current_value", choices[0]))
                                        idx = choices.index(default_choice) if default_choice in choices else 0
                                        val = st.selectbox(f"{feature_name}", options=choices, index=idx, key=ui_key)
                                    factor_overrides[feature_name] = val
                        else:
                            st.caption("Каталог дополнительных факторов недоступен.")
                    segments_payload: Dict[str, Any] = {}
                    segment_errors: List[str] = []
                    segment_count_ui = 0
                    st.markdown("**Периоды с разными параметрами**")
                    use_segments = st.checkbox("Использовать разные параметры по периодам", key="what_if_use_segments")
                    segments: List[Dict[str, Any]] = []
                    if use_segments and scenario_calc_mode == CATBOOST_FULL_FACTOR_MODE:
                        st.info("Для выбранного режима пока применяется единый сценарий на весь период.")
                    if use_segments and scenario_calc_mode == "enhanced_local_factors":
                        seg_count = st.selectbox("Количество сегментов", options=[1, 2, 3], index=1)
                        segment_count_ui = int(seg_count)
                        fdates = r["_trained_bundle"]["future_dates"]
                        start_default = pd.to_datetime(fdates["date"]).min().date() if len(fdates) else pd.Timestamp.utcnow().date()
                        end_default = pd.to_datetime(fdates["date"]).max().date() if len(fdates) else start_default
                        full_dates = pd.date_range(start_default, end_default, freq="D")
                        if len(full_dates) == 0:
                            full_dates = pd.DatetimeIndex([pd.Timestamp(start_default)])
                        chunk = int(np.ceil(len(full_dates) / max(int(seg_count), 1)))
                        for i in range(int(seg_count)):
                            st.markdown(f"**Сегмент {i+1}**")
                            idx_start = min(i * chunk, len(full_dates) - 1)
                            idx_end = min(((i + 1) * chunk) - 1, len(full_dates) - 1)
                            if i == int(seg_count) - 1:
                                idx_end = len(full_dates) - 1
                            default_seg_start = full_dates[idx_start].date()
                            default_seg_end = full_dates[idx_end].date()
                            start_key = f"seg_start_{i}"
                            end_key = f"seg_end_{i}"
                            if start_key not in st.session_state:
                                st.session_state[start_key] = default_seg_start
                            if end_key not in st.session_state:
                                st.session_state[end_key] = default_seg_end
                            s1, s2 = st.columns(2)
                            seg_start = s1.date_input("Дата начала", value=st.session_state[start_key], key=start_key)
                            seg_end = s2.date_input("Дата окончания", value=st.session_state[end_key], key=end_key)
                            s3, s4, s5, s8 = st.columns(4)
                            seg_price = s3.number_input("Цена до скидки, ₽", min_value=0.01, value=float(manual_price), key=f"seg_price_{i}")
                            seg_discount_pct = s4.slider("Скидка, %", 0, 95, int(round(float(discount) * 100.0)), step=1, key=f"seg_discount_pct_{i}", format="%d%%")
                            seg_promo_pct = s5.slider("Промо, %", 0, 70, int(round(float(min(promo_value, 0.70)) * 100.0)), step=1, key=f"seg_promo_pct_{i}", format="%d%%")
                            s6, s7, s9 = st.columns(3)
                            seg_freight_change_pct = s6.slider("Изменение логистики, %", -50.0, 50.0, float(freight_change_pct), step=1.0, key=f"seg_freight_pct_{i}")
                            seg_demand_shock_pct = s7.slider("Внешний шок спроса, %", -30.0, 30.0, float(demand_shock_pct), step=1.0, key=f"seg_demand_pct_{i}")
                            seg_units = s9.number_input("Дополнительные продажи, шт.", value=0.0, step=1.0, key=f"seg_units_{i}")
                            segments.append(
                                {
                                    "start_date": str(seg_start),
                                    "end_date": str(seg_end),
                                    "price": float(seg_price),
                                    "discount": float(seg_discount_pct) / 100.0,
                                    "promotion": float(seg_promo_pct) / 100.0,
                                    "freight_multiplier": float(pct_to_multiplier(seg_freight_change_pct)),
                                    "demand_multiplier": float(pct_to_multiplier(seg_demand_shock_pct)),
                                    "shock_units": float(seg_units),
                                }
                            )
                        segment_ranges = []
                        seg_validation_errors: List[str] = []
                        for s in segments:
                            start_dt = pd.to_datetime(s.get("start_date"), errors="coerce")
                            end_dt = pd.to_datetime(s.get("end_date"), errors="coerce")
                            if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
                                seg_validation_errors.append("В сегментах найдены пустые или некорректные интервалы дат.")
                                continue
                            segment_ranges.append((start_dt, end_dt))
                        for i in range(len(segment_ranges)):
                            for j in range(i + 1, len(segment_ranges)):
                                a_start, a_end = segment_ranges[i]
                                b_start, b_end = segment_ranges[j]
                                if max(a_start, b_start) <= min(a_end, b_end):
                                    seg_validation_errors.append("Сегменты не должны пересекаться по датам.")
                                    break
                        if seg_validation_errors:
                            segment_errors = sorted(set(seg_validation_errors))
                            for msg in segment_errors:
                                st.error(msg)
                        else:
                            segment_paths, segment_warnings = build_segment_paths(
                                r["_trained_bundle"]["future_dates"].head(int(hdays)),
                                {
                                    "price": float(manual_price),
                                    "promotion": float(promo_value),
                                    "discount": float(discount),
                                    "freight_multiplier": float(freight_mult),
                                    "demand_multiplier": float(demand_mult),
                                    "freight_value": float(r["_trained_bundle"]["base_ctx"].get("freight_value", 0.0)),
                                },
                                segments,
                            )
                            segments_payload = segment_paths
                            for sw in segment_warnings:
                                st.warning(sw)
                has_segment_errors = len(segment_errors) > 0
                st.markdown("**3. Рассчитать сценарий**")
                apply_btn = st.form_submit_button("Рассчитать сценарий", type="primary", use_container_width=True, disabled=has_segment_errors)
                st.caption("После расчёта сценарий будет сравнен с текущим планом по спросу, выручке, прибыли и риску.")
                if has_segment_errors:
                    st.error("Сценарий нельзя рассчитать: исправьте ошибки в периодах сегментов.")
                st.markdown("</div>", unsafe_allow_html=True)
        with right_col:
            st.caption("2. Что получится: предварительный контекст. Итоговые спрос, выручка, прибыль и риск обновятся после нажатия «Рассчитать сценарий».")
            current_form_after_widgets = collect_current_form_values(
                float(manual_price),
                float(discount),
                float(promo_value),
                float(freight_mult),
                float(demand_mult),
                int(hdays),
                str(scenario_calc_mode),
                normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
            )
            st.session_state.form_last_values = current_form_after_widgets
            st.session_state.scenario_ui_status = get_user_scenario_status(
                current_form_after_widgets,
                base_form,
                st.session_state.applied_scenario_snapshot,
                st.session_state.get("scenario_ui_status", "as_is"),
            )
            if st.session_state.scenario_ui_status == "dirty":
                render_help_callout(
                    "Есть неприменённые изменения",
                    "Нажмите «Рассчитать сценарий», чтобы обновить результат. До этого ниже может быть показан прошлый расчёт.",
                    "warning",
                )
            render_scenario_preview(
                current_price=float(r.get("current_price", 0.0)),
                manual_price=float(manual_price),
                discount=float(discount),
                promo_value=float(promo_value),
                freight_change_pct=float(freight_change_pct),
                demand_shock_pct=float(demand_shock_pct),
                hdays=int(hdays),
                segment_count=int(segment_count_ui),
                scenario_changed=st.session_state.scenario_ui_status == "dirty",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        close_surface()
        if apply_btn:
            if len(segment_errors) > 0:
                st.error("Сначала исправьте ошибки сегментов.")
                st.stop()
            overrides_payload = {"promotion": float(promo_value), "discount": float(discount)}
            demand_multiplier_for_run = float(demand_mult)
            if scenario_calc_mode == "enhanced_local_factors" and segments_payload:
                overrides_payload.update(
                    {
                        "price_path": segments_payload.get("price_path", []),
                        "discount_path": segments_payload.get("discount_path", []),
                        "promo_path": segments_payload.get("promo_path", []),
                        "freight_path": segments_payload.get("freight_path", []),
                        "demand_multiplier_path": segments_payload.get("demand_multiplier_path", []),
                        "segments": segments_payload.get("segments", []),
                        "shocks": list(segments_payload.get("segment_shocks", [])),
                    }
                )
                if len(segments_payload.get("demand_multiplier_path", [])):
                    demand_multiplier_for_run = 1.0
            st.session_state.what_if_result = run_what_if_projection(
                r["_trained_bundle"],
                manual_price=float(manual_price),
                freight_multiplier=float(freight_mult),
                demand_multiplier=float(demand_multiplier_for_run),
                horizon_days=int(hdays),
                overrides=overrides_payload,
                factor_overrides=factor_overrides,
                scenario_calc_mode=str(scenario_calc_mode),
                price_guardrail_mode=st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE),
            )
            st.session_state["last_factor_overrides"] = dict(factor_overrides or {})
            wr = st.session_state.what_if_result
            try:
                base_for_scenario = align_forecasts_by_scenario_dates(current_forecast, wr["daily"])
            except Exception as exc:
                st.error(f"Ошибка выравнивания текущего плана и сценария: {exc}")
                st.session_state.what_if_result = None
                st.session_state.scenario_ui_status = "dirty"
                st.stop()
            gate = validate_scenario_consistency(base_for_scenario, wr, expected_hdays=int(hdays))
            wr["validation_gate"] = gate
            if not gate.get("ok", False):
                for e in gate.get("errors", []):
                    st.error(f"Проверка корректности сценария: {e}")
                st.session_state.what_if_result = None
                st.session_state.scenario_ui_status = "dirty"
                st.stop()
            r["scenario_forecast"] = wr["daily"].copy()
            r["scenario_price_requested"] = float(wr.get("requested_price", manual_price))
            r["scenario_price_modeled"] = float(wr.get("model_price", wr.get("price_for_model", manual_price)))
            r["scenario_price"] = r["scenario_price_modeled"]
            base_units_local = float(base_for_scenario["actual_sales"].sum()) if len(base_for_scenario) else float("nan")
            base_revenue_local = float(base_for_scenario["revenue"].sum()) if len(base_for_scenario) else float("nan")
            base_profit_local = float(base_for_scenario["profit"].sum()) if ("profit" in base_for_scenario.columns and len(base_for_scenario)) else float("nan")
            sc_forecast_local = wr.get("daily", pd.DataFrame())
            sc_units_local = float(sc_forecast_local["actual_sales"].sum()) if len(sc_forecast_local) else float("nan")
            sc_revenue_local = float(sc_forecast_local["revenue"].sum()) if len(sc_forecast_local) else float("nan")
            sc_profit_local = float(sc_forecast_local["profit"].sum()) if ("profit" in sc_forecast_local.columns and len(sc_forecast_local)) else float("nan")
            demand_delta_pct_local = ((sc_units_local - base_units_local) / base_units_local * 100.0) if base_units_local else float("nan")
            revenue_delta_pct_local = ((sc_revenue_local - base_revenue_local) / base_revenue_local * 100.0) if base_revenue_local else float("nan")
            profit_delta_pct_local = safe_signed_pct((sc_profit_local - base_profit_local), base_profit_local)
            shape_quality_low_local = bool((r.get("summary", {}) or {}).get("shape_quality_low", False) or (r.get("diagnostics", {}) or {}).get("shape_quality_low", False))
            economic_label_local, _, _ = classify_economic_verdict(
                float(profit_delta_pct_local) if np.isfinite(profit_delta_pct_local) else float("nan"),
                float(demand_delta_pct_local) if np.isfinite(demand_delta_pct_local) else float("nan"),
                float(revenue_delta_pct_local) if np.isfinite(revenue_delta_pct_local) else float("nan"),
            )
            reliability_label_local, _, _ = classify_reliability_verdict(
                bool(wr.get("ood_flag", False)),
                list(wr.get("warnings", [])),
                str(wr.get("confidence_label", "")),
                str(wr.get("support_label", "")),
                shape_quality_low_local,
                bool((wr.get("validation_gate", {}) or {}).get("ok", True)),
            )
            wr["economic_verdict"] = economic_label_local
            wr["reliability_verdict"] = reliability_label_local
            r["manual_scenario_summary_json"], r["manual_scenario_daily_csv"] = build_manual_scenario_artifacts(r, wr)
            refresh_excel_export(r, wr)
            st.session_state.applied_scenario_snapshot = build_applied_scenario_snapshot(
                wr,
                manual_price,
                discount,
                promo_value,
                freight_mult,
                demand_mult,
                hdays,
                scenario_calc_mode,
            )
            st.session_state.scenario_ui_status = "applied"
            st.session_state.results = r
            st.rerun()

        wr = st.session_state.what_if_result
        if wr is None:
            open_surface("Итог", "Сравнение с базовым прогнозом на выбранный период.")
            render_product_empty_state("Сценарий не рассчитан", "Задайте параметры и нажмите «Рассчитать сценарий».", "До расчёта ниже может быть показан только базовый прогноз.")
            close_surface()
            demand_delta = revenue_delta = profit_delta = margin_delta = np.nan
        else:
            sc_forecast = r.get("scenario_forecast")
            base_for_scenario = align_forecasts_by_scenario_dates(current_forecast, sc_forecast)
            base_units_local = float(base_for_scenario["actual_sales"].sum()) if len(base_for_scenario) else float("nan")
            base_revenue_local = float(base_for_scenario["revenue"].sum()) if len(base_for_scenario) else float("nan")
            base_profit_local = float(base_for_scenario["profit"].sum()) if ("profit" in base_for_scenario.columns and len(base_for_scenario)) else float("nan")
            sc_units_local = float(sc_forecast["actual_sales"].sum()) if sc_forecast is not None else float("nan")
            sc_revenue_local = float(sc_forecast["revenue"].sum()) if sc_forecast is not None else float("nan")
            sc_profit_local = float(sc_forecast["profit"].sum()) if (sc_forecast is not None and "profit" in sc_forecast.columns) else float("nan")
            demand_delta = sc_units_local - base_units_local
            revenue_delta = sc_revenue_local - base_revenue_local
            profit_delta = sc_profit_local - base_profit_local
            demand_delta_pct = (demand_delta / base_units_local * 100.0) if base_units_local else float("nan")
            revenue_delta_pct = (revenue_delta / base_revenue_local * 100.0) if base_revenue_local else float("nan")
            profit_delta_pct = safe_signed_pct(profit_delta, base_profit_local)
            base_margin = (base_profit_local / base_revenue_local * 100.0) if base_revenue_local else float("nan")
            scenario_margin = (sc_profit_local / sc_revenue_local * 100.0) if sc_revenue_local else float("nan")
            margin_delta = scenario_margin - base_margin if np.isfinite(scenario_margin) and np.isfinite(base_margin) else float("nan")
            shape_quality_low = bool((r.get("summary", {}) or {}).get("shape_quality_low", False) or (r.get("diagnostics", {}) or {}).get("shape_quality_low", False))
            assessment_label, assessment_css = classify_scenario_assessment(
                float(profit_delta_pct) if np.isfinite(profit_delta_pct) else 0.0,
                list(wr.get("warnings", [])),
                bool(wr.get("ood_flag", False)),
                str(wr.get("confidence_label", "")),
                shape_quality_low,
                str(wr.get("support_label", "")),
            )
            economic_label, economic_css, economic_text = classify_economic_verdict(
                float(profit_delta_pct) if np.isfinite(profit_delta_pct) else float("nan"),
                float(demand_delta_pct) if np.isfinite(demand_delta_pct) else float("nan"),
                float(revenue_delta_pct) if np.isfinite(revenue_delta_pct) else float("nan"),
            )
            reliability_label_explicit, reliability_css_explicit, reliability_text = classify_reliability_verdict(
                bool(wr.get("ood_flag", False)),
                list(wr.get("warnings", [])),
                str(wr.get("confidence_label", "")),
                str(wr.get("support_label", "")),
                shape_quality_low,
                bool((wr.get("validation_gate", {}) or {}).get("ok", True)),
            )
            wr["economic_verdict"] = economic_label
            wr["reliability_verdict"] = reliability_label_explicit
            gate_ok_now = bool((wr.get("validation_gate", {}) or {}).get("ok", True))
            ui_decision = build_ui_decision_summary(gate_ok_now, float(profit_delta_pct) if np.isfinite(profit_delta_pct) else float("nan"), reliability_label_explicit)
            open_surface("Итог", "Сравнение с базовым прогнозом на выбранный период.")
            render_decision_summary_card(
                decision_label=ui_decision["decision_label"],
                tone=ui_decision["tone"],
                reason=ui_decision["reason"],
                metrics=[
                    {"label": "Прибыль", "value": fmt_money_total(profit_delta), "delta": fmt_pct_delta(profit_delta_pct)},
                    {"label": "Спрос", "value": fmt_units(demand_delta), "delta": fmt_pct_delta(demand_delta_pct)},
                    {"label": "Выручка", "value": fmt_money_total(revenue_delta), "delta": fmt_pct_delta(revenue_delta_pct)},
                ],
                economy_label=economic_label,
                reliability_label=reliability_label_explicit,
            )
            kpi_payload = {
                "Спрос": {"value": fmt_units(sc_units_local), "delta_text": f"{fmt_units(demand_delta)} / {fmt_pct(demand_delta_pct)}", "base_text": fmt_units(base_units_local), "delta_numeric": demand_delta},
                "Выручка": {"value": fmt_money_total(sc_revenue_local), "delta_text": f"{fmt_money_total(revenue_delta)} / {fmt_pct_delta(revenue_delta_pct)}", "base_text": fmt_money_total(base_revenue_local), "delta_numeric": revenue_delta},
                "Прибыль": {"value": fmt_money_total(sc_profit_local), "delta_text": f"{fmt_money_total(profit_delta)} / {fmt_pct_delta(profit_delta_pct)}", "base_text": fmt_money_total(base_profit_local), "delta_numeric": profit_delta},
                "Маржа": {"value": fmt_pct_abs(scenario_margin), "delta_text": fmt_pp_delta(margin_delta), "base_text": fmt_pct_abs(base_margin), "delta_numeric": margin_delta},
            }
            render_result_kpi_grid(kpi_payload)
            st.markdown(f'<div class="{economic_css}"><b>Экономический итог:</b> {economic_label}. {economic_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{reliability_css_explicit}"><b>Надёжность расчёта:</b> {reliability_label_explicit}. {reliability_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{assessment_css}"><b>Оценка сценария:</b> {assessment_label}. Используйте как ориентир для сравнения, а не как гарантию.</div>', unsafe_allow_html=True)
            reading_points = []
            if np.isfinite(profit_delta) and profit_delta > 0:
                reading_points.append("Прибыль выше базового плана. Это модельная оценка для проверки гипотезы, а не гарантия результата.")
            elif np.isfinite(profit_delta) and profit_delta < 0:
                reading_points.append("Прибыль ниже базового плана. Такой сценарий не стоит запускать без отдельного бизнес-обоснования.")
            if str(reliability_label_explicit).lower() not in {"высокая", "высокий", "high"}:
                reading_points.append("Надёжность ограничена: используйте сценарий как ориентир и проверьте на пилоте.")
            if reading_points:
                open_surface("Как читать результат")
                for point in reading_points:
                    st.markdown(f"• {point}")
                close_surface()
            if assessment_label == "Рискован":
                st.markdown(
                    '<div class="scenario-danger-inline"><b>Сценарий рискован.</b> Спрос растёт, но прибыль падает: скидка снижает маржу сильнее, чем растёт объём. Проверьте меньшую скидку, повышение цены или сценарий без промо.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("#### Краткий вывод")
            render_business_summary(revenue_delta, profit_delta, demand_delta, margin_delta)
            st.markdown("#### Почему изменилась прибыль")
            render_profit_change_explanation(
                {
                    "net_price": float(current_forecast["net_unit_price"].mean()) if "net_unit_price" in current_forecast.columns and len(current_forecast) else float("nan"),
                    "unit_margin": (base_profit_local / max(base_units_local, 1e-9)),
                    "profit": base_profit_local,
                },
                {
                    "net_price": float(sc_forecast["net_unit_price"].mean()) if ("net_unit_price" in sc_forecast.columns and len(sc_forecast)) else float("nan"),
                    "unit_margin": (sc_profit_local / max(sc_units_local, 1e-9)),
                    "profit": sc_profit_local,
                },
                float(demand_delta_pct) if np.isfinite(demand_delta_pct) else 0.0,
                profit_delta,
            )
            close_surface()
            render_applied_scenario_block(st.session_state.applied_scenario_snapshot)
            open_surface("Почему изменился прогноз")
            render_human_effect_breakdown((wr.get("effect_breakdown", {}) or {}))
            with st.expander("Техническое разложение эффектов", expanded=False):
                decomp = pd.DataFrame(wr.get("daily_effects_summary", []))
                if len(decomp):
                    st.dataframe(decomp.head(30), use_container_width=True)
                else:
                    st.caption("Нет технических данных разложения.")
            close_surface()
            open_surface("Надёжность сценария")
            render_reliability_card(wr, r)
            close_surface()

        if wr is not None:
            open_surface("Сохранить вариант", "Сохраните рассчитанный сценарий как вариант, чтобы сравнить его с другими решениями.")
            selected_slot = st.selectbox("Куда сохранить вариант", options=["Scenario A", "Scenario B", "Scenario C"], key="compare_slot", format_func=lambda x: SCENARIO_SLOT_LABELS.get(x, x))
            save_to_slot_btn = st.button("Сохранить вариант", use_container_width=True, disabled=r.get("scenario_forecast") is None)
            if save_to_slot_btn and r.get("scenario_forecast") is not None:
                st.session_state.saved_scenarios[selected_slot] = build_saved_scenario_metrics(current_forecast, r["scenario_forecast"], st.session_state.what_if_result)
                st.session_state.last_saved_slot = selected_slot
                st.session_state.scenario_ui_status = "saved"
                st.toast(f"Сценарий сохранён: {SCENARIO_SLOT_LABELS.get(selected_slot, selected_slot)}", icon="✓")
                st.rerun()
            close_surface()
        else:
            st.caption("Сначала рассчитайте сценарий.")

        st.info("Сравнить с текущим планом можно в разделе «Сравнение».")
        open_surface("Хотите подобрать цену автоматически?", "Подбор цены вынесен в отдельный раздел, чтобы не смешивать настройку сценария и поиск цены.")
        st.markdown("Откройте раздел **«Подбор цены»**: там система проверит цены рядом с текущей и покажет лучший вариант по прибыли. Цена не применяется автоматически.")
        if st.button("Открыть раздел «Подбор цены»", key="go_price_candidate_from_scenario", use_container_width=True):
            st.session_state.active_workspace_tab = PAGE_PRICE
            st.session_state["workspace_tab_radio"] = PAGE_PRICE
            st.rerun()
        close_surface()

        st.markdown("</div>", unsafe_allow_html=True)


    elif active_tab == PAGE_DECISION:
        render_page_header(
            PAGE_DECISION,
            "Покажет, стоит ли запускать изменение, какие риски есть и как безопасно протестировать.",
        )
        render_help_callout(
            "Что делает этот раздел",
            "Анализатор не меняет модель и не пересчитывает базовый прогноз. Он проверяет варианты решения на основе уже рассчитанного прогноза или сценария, объясняет риски и предлагает план пилота.",
            "info",
        )
        if not isinstance(r.get("_trained_bundle"), dict) or not r.get("_trained_bundle"):
            st.warning("Сначала выполните базовый анализ: для проверки решений нужен расчётный контекст.")
            st.stop()
        objective_map = {
            "Прибыль": "profit",
            "Выручка": "revenue",
            "Спрос": "demand",
            "Риск": "risk_reduction",
        }
        action_map = {
            "Цена": "price_change",
            "Скидка": "discount_change",
            "Промо": "promotion_change",
            "Логистика": "freight_change",
        }
        reverse_action_map = {
            "Изменить цену": "price_change",
            "Изменить скидку": "discount_change",
            "Включить промо": "promotion_change",
            "Изменить логистику": "freight_change",
            "Внешний спрос": "demand_shock",
        }

        def _show_decision_passport(passport: Dict[str, Any], table: Optional[List[Dict[str, Any]]] = None) -> None:
            rel = passport.get("reliability", {}) or {}
            eff = passport.get("expected_effect", {}) or {}
            action = passport.get("recommended_action", {}) or {}
            status_raw = str(passport.get("decision_status", "not_recommended"))
            tone = decision_status_tone(status_raw)
            status_text = decision_status_label(status_raw)
            if status_raw in {"recommended", "approve", "approved", "test_recommended", "test_only", "experimental", "experimental_only"}:
                short_reco = "Можно рассмотреть через пилот"
            elif status_raw == "blocked":
                short_reco = "Запуск заблокирован ограничениями"
            else:
                short_reco = "Не запускать без пересмотра условий"
            evidence = passport.get("evidence", []) or []
            limitations = passport.get("limitations", []) or []
            reason = str(evidence[0] if evidence else passport.get("reason", "Модельная оценка требует проверки на пилоте."))
            action_title = action.get("title") or action_type_label(action.get("action_type"))

            render_verdict_panel(
                verdict_label=status_text,
                action_title=str(action_title),
                reason=reason,
                metrics=[
                    {"label": "Прибыль", "value": fmt_pct_delta(_safe_metric_float(eff.get("conservative_profit_delta_pct", eff.get("profit_delta_pct", 0.0))))},
                    {"label": "Спрос", "value": fmt_pct_delta(_safe_metric_float(eff.get("demand_delta_pct", 0.0)))},
                    {"label": "Выручка", "value": fmt_pct_delta(_safe_metric_float(eff.get("revenue_delta_pct", 0.0)))},
                    {"label": "Риск", "value": risk_level_label(rel.get("risk_level", "n/a"))},
                ],
                reliability_label=f"{_safe_metric_float(rel.get('score', 0.0)):.0f}/100",
                next_step="Пилот 14 дней с контролем прибыли и спроса.",
                tone=tone,
            )

            open_surface("Что именно предлагается")
            info_rows = [
                ("Тип действия", action_type_label(action.get("action_type", passport.get("action_type", "")))),
                ("Текущее значение", action.get("current_value", passport.get("current_value", "—"))),
                ("Новое значение", action.get("target_value", passport.get("target_value", "—"))),
                ("Период проверки", f"{int((passport.get('validation_plan', {}) or {}).get('test_period_days', 0))} дней"),
                ("Что хотим улучшить", objective_label(action.get("objective", passport.get("objective", "profit")))),
            ]
            st.markdown('<div class="decision-section-grid">', unsafe_allow_html=True)
            for label, value in info_rows:
                st.markdown(f'<div class="decision-section-card"><div class="decision-section-label">{_html_safe(label)}</div><div class="decision-section-value">{_html_safe(value)}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            close_surface()

            open_surface("Почему такой вывод")
            if evidence:
                for item in evidence[:6]:
                    st.markdown(f"• {item}")
            else:
                st.caption("Подробные причины не переданы, используйте метрики и план проверки ниже.")
            close_surface()

            open_surface("Что может пойти не так")
            if limitations:
                for item in limitations[:8]:
                    st.markdown(f"• {item}")
            else:
                st.markdown("Критичных ограничений не найдено, но результат всё равно является модельной оценкой.")
            close_surface()

            plan = passport.get("validation_plan", {}) or {}
            open_surface("План безопасной проверки")
            p1, p2, p3 = st.columns(3)
            p1.metric("Период теста", f"{int(plan.get('test_period_days', 0))} дней")
            p2.metric("Формат теста", test_scope_label(plan.get("test_scope", "controlled_test")))
            p3.metric("Главная метрика", metric_label(plan.get("success_metric", "gross_profit")))
            secondary = plan.get("secondary_metrics", []) or []
            st.markdown(f"**Дополнительные метрики:** {', '.join(metric_label(x) for x in secondary) if secondary else '—'}")
            st.markdown(f"**Когда откатывать:** {plan.get('rollback_condition', 'Условие отката не задано.')}")
            st.markdown(f"**Как проверять:** {plan.get('control_recommendation', 'Использовать контрольную группу, если возможно.')}")
            close_surface()

            open_surface("Что делать дальше")
            next_steps = ["Запустить ограниченный пилот", "Следить за прибылью, спросом и выручкой", "Откатить, если сработает условие отката"] if tone in {"success", "warning"} else ["Не запускать в текущем виде", "Снизить риск: уменьшить изменение цены/скидки/промо", "Проверить альтернативный сценарий"]
            st.markdown('<div class="decision-next-grid">', unsafe_allow_html=True)
            for item in next_steps:
                st.markdown(f'<div class="decision-next-item">{item}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            close_surface()

            if table:
                user_table = build_user_friendly_decision_candidates_table(table)
                if len(user_table):
                    open_surface("Какие варианты были проверены")
                    st.caption("Первый вариант — лучший по выбранной цели и ограничениям.")
                    st.dataframe(user_table.head(5), use_container_width=True, hide_index=True)
                    close_surface()

            technical_failed = any(bool(((row or {}).get("full_reliability") or (row or {}).get("reliability") or {}).get("technical_error")) for row in (table or []))
            if technical_failed:
                st.warning("Часть вариантов не рассчиталась и исключена из результата. Подробности доступны в техническом блоке.")
            with st.expander("Для аналитика: технический файл решения", expanded=False):
                st.download_button(
                    "Скачать технический файл решения",
                    data=json.dumps(passport, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="decision_passport.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.caption("Файл нужен для аудита, передачи команде или сохранения параметров проверки.")
                st.markdown("**Технический JSON паспорта**")
                st.json(passport, expanded=False)
                if table:
                    st.dataframe(pd.DataFrame(table), use_container_width=True)

        context_source_label = st.radio(
            "Что взять за отправную точку",
            options=[
                "Базовый прогноз",
                "Рассчитанный сценарий",
            ],
            index=1 if st.session_state.get("what_if_result") is not None else 0,
            key="decision_context_source_label",
            help="Выберите, относительно чего проверять решение: текущий базовый план или уже рассчитанный сценарий.",
        )
        if st.session_state.get("what_if_result") is None:
            if context_source_label == "Рассчитанный сценарий":
                context_source_label = "Базовый прогноз"
            st.warning("Сценарий ещё не рассчитан. Сейчас можно проверять решения только от базового прогноза.")
        decision_context_source = "applied_scenario" if context_source_label == "Рассчитанный сценарий" else "base_ctx"
        current_decision_context = build_decision_current_context(r, decision_context_source)
        open_surface("Отправная точка анализа")
        st.markdown(f"**Цена:** {_fmt_decision_context_value(current_decision_context.get('price'), decimals=2)}")
        st.markdown(f"**Скидка:** {_fmt_decision_context_value(float(current_decision_context.get('discount', np.nan)) * 100.0, suffix='%', decimals=0)}")
        st.markdown(f"**Промо:** {_fmt_decision_context_value(float(current_decision_context.get('promotion', np.nan)) * 100.0, suffix='%', decimals=0)}")
        st.markdown(f"**Логистика:** {_fmt_decision_context_value(current_decision_context.get('freight_value'), decimals=2)}")
        st.markdown(f"**Источник:** {context_source_label}")
        close_surface()

        if "decision_mode" not in st.session_state:
            st.session_state["decision_mode"] = "find_best"
        selected_decision_mode = render_decision_mode_cards(str(st.session_state.get("decision_mode", "find_best")))
        if selected_decision_mode != st.session_state.get("decision_mode"):
            st.session_state["decision_mode"] = selected_decision_mode
            if selected_decision_mode == "quick_what_if":
                st.session_state.active_workspace_tab = PAGE_WHAT_IF
                st.session_state["workspace_tab_radio"] = PAGE_WHAT_IF
            st.rerun()
        if st.session_state.get("decision_mode") == "quick_what_if":
            st.session_state.active_workspace_tab = PAGE_WHAT_IF
            st.session_state["workspace_tab_radio"] = PAGE_WHAT_IF
            st.rerun()
        if st.session_state.get("decision_mode") == "find_best":
            st.caption("Система сама переберёт допустимые варианты: цену, скидку, промо и логистику. Затем покажет лучший вариант, риски и план проверки.")
            d_obj_label = st.selectbox("Что хотим улучшить", list(objective_map.keys()), key="decision_objective")
            d_actions = st.multiselect("Что можно менять", list(action_map.keys()), default=["Цена", "Скидка", "Промо", "Логистика"], key="decision_allowed_actions", help="Выберите, какие рычаги система имеет право менять. Если не уверены, оставьте цену, скидку, промо и логистику.")
            d_horizon = st.number_input("Период проверки, дней", min_value=7, max_value=120, value=int(st.session_state.get("what_if_hdays", CONFIG["HORIZON_DAYS_DEFAULT"])), step=1, key="decision_horizon")
            with st.expander("Ограничения бизнеса — можно оставить по умолчанию", expanded=False):
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    max_price_change_pct = st.number_input("Цена может измениться не больше чем на, %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="decision_max_price_change_pct")
                    min_margin_pct = st.number_input("Минимальная маржа, %", min_value=-100.0, max_value=100.0, value=0.0, step=1.0, key="decision_min_margin_pct")
                with bc2:
                    max_demand_drop_pct = st.number_input("Допустимое падение спроса, %", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="decision_max_demand_drop_pct")
                    max_revenue_drop_pct = st.number_input("Допустимое падение выручки, %", min_value=0.0, max_value=100.0, value=5.0, step=1.0, key="decision_max_revenue_drop_pct")
                with bc3:
                    min_expected_profit_uplift_pct = st.number_input("Минимальный прирост прибыли, %", min_value=-100.0, max_value=100.0, value=3.0, step=1.0, key="decision_min_profit_uplift_pct")
                    forbid_price_below_cost = st.checkbox("Не разрешать цену ниже себестоимости", value=True, key="decision_forbid_price_below_cost")
                    require_cost_for_profit = st.checkbox("Считать прибыль только при наличии себестоимости", value=True, key="decision_require_cost_for_profit")
            business_constraints = {
                "max_price_change_pct": float(max_price_change_pct),
                "min_margin_pct": float(min_margin_pct),
                "forbid_price_below_cost": bool(forbid_price_below_cost),
                "max_demand_drop_pct": float(max_demand_drop_pct),
                "max_revenue_drop_pct": float(max_revenue_drop_pct),
                "min_expected_profit_uplift_pct": float(min_expected_profit_uplift_pct),
                "require_cost_for_profit": bool(require_cost_for_profit),
            }
            decision_signature = json.dumps(
                {
                    "context_source": decision_context_source,
                    "current_context": current_decision_context,
                    "objective_label": d_obj_label,
                    "allowed_actions": d_actions,
                    "horizon": int(d_horizon),
                    "business_constraints": business_constraints,
                    "scenario_calc_mode": str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                    "price_guardrail_mode": normalize_price_guardrail_mode(
                        st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if st.button("Найти лучшее решение", key="run_decision_optimizer", use_container_width=True):
                objective = objective_map[d_obj_label]
                candidates = generate_decision_candidates(
                    r["_trained_bundle"],
                    current_decision_context,
                    objective=objective,
                    allowed_actions=[action_map[a] for a in d_actions],
                    horizon_days=int(d_horizon),
                )
                evaluated = evaluate_decision_candidates(
                    r,
                    r["_trained_bundle"],
                    candidates,
                    run_what_if_projection,
                    scenario_calc_mode=str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                    price_guardrail_mode=normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
                    horizon_days=int(d_horizon),
                    objective=objective,
                    current_context={**current_decision_context, "constraints": business_constraints},
                    constraints=business_constraints,
                    max_candidates=12 if str(st.session_state.get("what_if_calc_mode")) == CATBOOST_FULL_FACTOR_MODE else 24,
                    max_refinements=4 if str(st.session_state.get("what_if_calc_mode")) == CATBOOST_FULL_FACTOR_MODE else 12,
                    timeout_sec=25.0,
                )
                opt = rank_decision_candidates(evaluated, objective=objective)
                ranked_for_analysis = opt.get("ranked_candidates", evaluated) if isinstance(opt, dict) else evaluated
                decision_analysis = analyze_decision(
                    DecisionAnalysisInput(
                        baseline={},
                        trained_bundle=r.get("_trained_bundle", {}),
                        data_contract=r.get("data_contract", r.get("_trained_bundle", {}).get("data_contract", {})),
                        model_quality_gate=r.get("model_quality_gate", r.get("_trained_bundle", {}).get("model_quality_gate", {})),
                        current_context=current_decision_context,
                        allowed_actions=[action_map[a] for a in d_actions],
                        business_constraints=business_constraints,
                        objective=objective,
                        horizon=int(d_horizon),
                    ),
                    evaluated_candidates=ranked_for_analysis,
                ).to_dict()
                passport = build_decision_passport(
                    "find_best_decision",
                    opt,
                    calculation_context={
                        **current_decision_context,
                        "scenario_calc_mode": str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                        "scenario_calc_mode_label": scenario_mode_label(str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE))),
                        "price_guardrail_mode": normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
                        "business_constraints": business_constraints,
                        "decision_analysis": decision_analysis,
                    },
                )
                passport.setdefault("decision_analysis_result", decision_analysis)
                st.session_state["decision_optimizer_result"] = opt
                st.session_state["decision_analysis_result"] = decision_analysis
                st.session_state["decision_passport"] = passport
                st.session_state["decision_optimizer_signature"] = decision_signature
            if isinstance(st.session_state.get("decision_passport"), dict):
                if st.session_state.get("decision_optimizer_signature") == decision_signature:
                    _show_decision_passport(st.session_state["decision_passport"], st.session_state.get("decision_optimizer_result", {}).get("ranking_table"))
                else:
                    st.warning("Параметры изменились. Нажмите «Найти лучшее решение», чтобы пересчитать вывод.")
        if st.session_state.get("decision_mode") == "audit_idea":
            st.caption("Используйте, если у вас уже есть идея: например, поднять цену, включить скидку, запустить промо или учесть внешний рост спроса.")
            a_action_label = st.selectbox("Что хотите проверить?", list(reverse_action_map.keys()), key="audit_action")
            with st.expander("Источник и доказательства — необязательно", expanded=False):
                a_source = st.text_input("Откуда рекомендация", value="Моя гипотеза", key="audit_source")
            audit_action = reverse_action_map[a_action_label]
            base_ctx_audit = r.get("_trained_bundle", {}).get("base_ctx", {})
            if audit_action == "price_change":
                a_target = st.number_input("Новая цена", min_value=0.01, value=float(current_decision_context.get("price", r.get("current_price", base_ctx_audit.get("price", 1.0)))), step=1.0, key="audit_target_price")
                base_value = float(current_decision_context.get("price", r.get("current_price", base_ctx_audit.get("price", 1.0))))
                external_evidence = False
                evidence_comment = ""
            elif audit_action == "discount_change":
                discount_pct = st.slider("Новая скидка, %", min_value=0, max_value=95, value=int(float(current_decision_context.get("discount", base_ctx_audit.get("discount", 0.0))) * 100), step=1, key="audit_target_discount_pct")
                a_target = float(discount_pct) / 100.0
                base_value = float(current_decision_context.get("discount", base_ctx_audit.get("discount", 0.0)))
                external_evidence = False
                evidence_comment = ""
            elif audit_action == "promotion_change":
                promo_choice = st.selectbox("Промо", ["Выключить", "Включить"], key="audit_target_promo")
                a_target = 1.0 if promo_choice == "Включить" else 0.0
                base_value = float(current_decision_context.get("promotion", base_ctx_audit.get("promotion", 0.0)))
                external_evidence = False
                evidence_comment = ""
            elif audit_action == "freight_change":
                a_target = st.number_input(
                    "Новая логистика на единицу, ₽",
                    min_value=0.0,
                    value=float(current_decision_context.get("freight_value", base_ctx_audit.get("freight_value", 0.0))),
                    step=1.0,
                    key="audit_target_freight",
                )
                base_value = float(current_decision_context.get("freight_value", base_ctx_audit.get("freight_value", 0.0)))
                external_evidence = False
                evidence_comment = ""
            else:
                demand_pct = st.slider("Ожидаемое внешнее изменение спроса, %", min_value=-50, max_value=50, value=0, step=1, key="audit_target_demand_pct")
                a_target = 1.0 + float(demand_pct) / 100.0
                base_value = 1.0
                external_evidence = st.checkbox("Есть внешнее подтверждение гипотезы", value=False, key="audit_external_evidence")
                evidence_comment = st.text_input("Кратко укажите источник", value="", key="audit_evidence_comment")
                st.caption("Ручная поправка спроса — это внешняя бизнес-гипотеза, например реклама, сезонность, рынок или дефицит товара. Модель не считает это доказанным эффектом, поэтому без подтверждения рекомендация должна идти только в пилот.")
            a_obj_label = st.selectbox("Что хотим улучшить", list(objective_map.keys()), key="audit_objective")
            with st.expander("Комментарий к гипотезе — необязательно", expanded=False):
                a_comment = st.text_area("Комментарий к гипотезе", value="Внешняя рекомендация", key="audit_comment")
            a_horizon = st.number_input("Период проверки, дней", min_value=7, max_value=120, value=int(st.session_state.get("what_if_hdays", CONFIG["HORIZON_DAYS_DEFAULT"])), step=1, key="audit_horizon")
            audit_signature = json.dumps(
                {
                    "context_source": decision_context_source,
                    "current_context": current_decision_context,
                    "source": a_source,
                    "action": audit_action,
                    "target": _safe_metric_float(a_target),
                    "base_value": _safe_metric_float(base_value),
                    "objective_label": a_obj_label,
                    "comment": a_comment,
                    "horizon": int(a_horizon),
                    "external_evidence": bool(external_evidence),
                    "evidence_comment": evidence_comment,
                    "scenario_calc_mode": str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                    "price_guardrail_mode": normalize_price_guardrail_mode(
                        st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if st.button("Проверить идею", key="run_recommendation_audit", use_container_width=True):
                objective = objective_map[a_obj_label]
                recommendation = {
                    "source_name": a_source,
                    "action_type": audit_action,
                    "target_value": float(a_target),
                    "change_pct": safe_pct_delta(float(a_target), base_value) if 'safe_pct_delta' in globals() else 0.0,
                    "objective": objective,
                    "comment": a_comment,
                    "metadata": {"external_evidence": bool(external_evidence), "evidence_comment": evidence_comment},
                }
                audit = audit_and_improve_recommendation(
                    r,
                    r["_trained_bundle"],
                    recommendation,
                    run_what_if_projection,
                    scenario_calc_mode=str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                    price_guardrail_mode=normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
                    horizon_days=int(a_horizon),
                    objective=objective,
                    current_context={
                        **current_decision_context,
                        "scenario_calc_mode": str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE)),
                        "scenario_calc_mode_label": scenario_mode_label(str(st.session_state.get("what_if_calc_mode", DEFAULT_SCENARIO_CALC_MODE))),
                        "price_guardrail_mode": normalize_price_guardrail_mode(st.session_state.get("price_guardrail_mode", DEFAULT_PRICE_GUARDRAIL_MODE)),
                    },
                )
                st.session_state["recommendation_audit_result"] = audit
                st.session_state["recommendation_audit_signature"] = audit_signature
            audit = st.session_state.get("recommendation_audit_result")
            if isinstance(audit, dict):
                if st.session_state.get("recommendation_audit_signature") != audit_signature:
                    st.warning("Параметры идеи изменились. Нажмите «Проверить идею», чтобы обновить вывод.")
                    audit = None
            if isinstance(audit, dict):
                verdict = audit.get("audit_verdict", {})
                human_verdict = decision_status_label(str(verdict.get("verdict", "needs_validation")))
                open_surface("Вердикт по рекомендации")
                st.markdown(f"**Статус:** {human_verdict}")
                st.markdown(f"**Причина:** {verdict.get('reason', '—')}")
                close_surface()
                improved = audit.get("improved_solution", {}) or {}
                open_surface("Как улучшить рекомендацию")
                if isinstance(improved, dict) and improved:
                    st.markdown(f"**Предлагаемое действие:** {action_type_label(improved.get('action_type', audit_action))}")
                    st.markdown(f"**Новое значение:** {improved.get('target_value', improved.get('new_value', '—'))}")
                    st.markdown(f"**Почему лучше:** {improved.get('reason', improved.get('rationale', 'Учитывает найденные риски и ограничения.'))}")
                    st.markdown(f"**Какие ограничения учтены:** {improved.get('constraints_note', 'Бизнес-ограничения и риск пилота.')}")
                else:
                    st.markdown("Система не предложила улучшение в явном виде. Используйте вердикт и план проверки ниже.")
                close_surface()
                with st.expander("Для аналитика: технические детали", expanded=False):
                    st.json(improved, expanded=False)
                _show_decision_passport(audit.get("decision_passport", {}), audit.get("alternatives_table", []))

    elif active_tab == PAGE_COMPARE:
        render_page_header(PAGE_COMPARE, "Сравните текущий план, рассчитанный сценарий и сохранённые варианты.")
        compare_base = current_forecast
        if r.get("scenario_forecast") is not None:
            compare_base = align_forecasts_by_scenario_dates(current_forecast, r.get("scenario_forecast"))
        compare_df = build_user_friendly_comparison_table(compare_base, r.get("scenario_forecast"), st.session_state.saved_scenarios, True)
        wr_local = st.session_state.what_if_result or {}
        snap_local = st.session_state.applied_scenario_snapshot or {}
        if len(compare_df) and "Сценарий" in compare_df.columns:
            mask_current = compare_df["Сценарий"].astype(str).eq("Текущий сценарий")
            if mask_current.any():
                compare_df.loc[mask_current, "Цена до скидки"] = float(
                    snap_local.get("manual_price_applied")
                    or wr_local.get("requested_price")
                    or wr_local.get("applied_price_gross")
                    or np.nan
                )
                compare_df.loc[mask_current, "Цена после скидки"] = float(
                    snap_local.get("net_price_applied")
                    or wr_local.get("applied_price_net")
                    or np.nan
                )
                compare_df.loc[mask_current, "Надёжность"] = str(wr_local.get("confidence_label", ""))
                compare_df.loc[mask_current, "Поддержка данных"] = str(wr_local.get("support_label", ""))
        def _display_value(value: Any, formatter) -> str:
            if value is None:
                return "—"
            if isinstance(value, str):
                cleaned = value.strip()
                return cleaned if cleaned and cleaned != "—" else "—"
            num = safe_float_or_nan(value)
            if np.isfinite(num):
                try:
                    return str(formatter(num))
                except Exception:
                    return str(num)
            return "—"

        if len(compare_df):
            numeric_compare = compare_df.copy()
            for col_name in ["Прибыль", "Выручка", "Предупреждения"]:
                if col_name in numeric_compare.columns:
                    numeric_compare[col_name] = pd.to_numeric(numeric_compare[col_name], errors="coerce")
            open_surface("Краткие лидеры")
            l1, l2, l3 = st.columns(3)
            if "Прибыль" in numeric_compare.columns and numeric_compare["Прибыль"].notna().any():
                best_profit = numeric_compare.loc[numeric_compare["Прибыль"].idxmax()]
                l1.metric("Лучший по прибыли", str(best_profit.get("Сценарий", "—")), fmt_money_total(best_profit.get("Прибыль", np.nan)))
            if "Предупреждения" in numeric_compare.columns and numeric_compare["Предупреждения"].notna().any():
                riskiest = numeric_compare.loc[numeric_compare["Предупреждения"].idxmax()]
                l2.metric("Самый рискованный", str(riskiest.get("Сценарий", "—")), f"{int(riskiest.get('Предупреждения', 0))} предупрежд.")
            if "Предупреждения" in numeric_compare.columns and numeric_compare["Предупреждения"].notna().any():
                safest = numeric_compare.loc[numeric_compare["Предупреждения"].idxmin()]
                l3.metric("Самый безопасный", str(safest.get("Сценарий", "—")), f"{int(safest.get('Предупреждения', 0))} предупрежд.")
            else:
                l3.metric("Самый безопасный", "Текущий план", "База")
            close_surface()
            cols = st.columns(min(5, len(compare_df)))
            for i, (_, row) in enumerate(compare_df.head(5).iterrows()):
                with cols[i % len(cols)]:
                    scenario_name = str(row.get("Сценарий", "Сценарий"))
                    price_value = row.get("Цена", None)
                    if price_value in [None, "—", ""]:
                        price_value = row.get("Цена после скидки", row.get("Цена до скидки", "—"))
                    if scenario_name == "Текущий план":
                        price_value = fmt_price(r.get("current_price", np.nan))
                    if scenario_name == "Текущий сценарий" and price_value in [None, "—", ""]:
                        snap = st.session_state.applied_scenario_snapshot or {}
                        wr_local = st.session_state.what_if_result or {}
                        price_value = (
                            snap.get("manual_price_applied")
                            or wr_local.get("requested_price")
                            or wr_local.get("applied_price_gross")
                            or wr_local.get("model_price")
                            or price_value
                        )
                    open_surface(str(row.get("Сценарий", "Сценарий")))
                    st.markdown(f"**Цена:** {_display_value(price_value, fmt_price)}")
                    st.markdown(f"**Спрос:** {_display_value(row.get('Спрос', np.nan), fmt_units)}")
                    st.markdown(f"**Выручка:** {_display_value(row.get('Выручка', np.nan), fmt_money_total)}")
                    st.markdown(f"**Прибыль:** {_display_value(row.get('Прибыль', np.nan), fmt_money_total)}")
                    st.markdown(f"**Изм. прибыли:** {_display_value(row.get('Δ прибыли', np.nan), fmt_money_total)}")
                    delta_profit = pd.to_numeric(pd.Series([row.get("Δ прибыли", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
                    row_conf = str(row.get("Надёжность", "")).strip().lower()
                    reliability = row_conf
                    if scenario_name == "Текущий план":
                        status = "База сравнения"
                    elif delta_profit < 0:
                        status = "Хуже"
                    elif delta_profit > 0:
                        status = "Требует проверки" if ("низ" in reliability or reliability == "low") else "Лучше"
                    else:
                        status = "Без изменений"
                    reliability_text_ui = row.get("Надёжность") if str(row.get("Надёжность", "")).strip() else "н/д"
                    st.markdown(f"**Надёжность:** {reliability_text_ui}")
                    st.markdown(f"**Статус:** {status}")
                    close_surface()
            with st.expander("Показать подробную таблицу", expanded=False):
                st.dataframe(compare_df, use_container_width=True, hide_index=True)
        else:
            render_product_empty_state("Нет вариантов для сравнения", "Рассчитайте сценарий или сохраните вариант, чтобы сравнить экономику и риски.", "Перейдите в «What-if» и нажмите «Рассчитать сценарий».")

    elif active_tab == PAGE_REPORT:
        render_page_header(PAGE_REPORT, "Скачайте бизнес-отчёт, дневные данные или технический файл для аналитика.")
        wr = st.session_state.what_if_result or {}
        scenario_applied = wr != {} and r.get("scenario_forecast") is not None
        gate_ok = bool((wr.get("validation_gate", {}) or {}).get("ok", True)) if scenario_applied else False
        has_results = st.session_state.results is not None
        open_surface("Что войдёт в отчёт")
        st.markdown("""- базовый прогноз;
- проверенное решение или сценарий;
- спрос, выручка, прибыль;
- риски и ограничения;
- план проверки;
- технический паспорт для аналитика.""")
        if not scenario_applied:
            st.caption("Недоступно для сценарных файлов: сначала рассчитайте сценарий.")
        close_surface()
        excel_disabled = not has_results
        csv_disabled = (not has_results) or (not scenario_applied) or (not gate_ok)
        tech_disabled = (not has_results) or (not scenario_applied) or (not gate_ok)
        open_surface("Excel для бизнеса", "Основной отчёт для передачи команде.")
        if excel_disabled:
            st.caption("Недоступно: сначала загрузите данные и запустите анализ.")
        st.download_button("Скачать Excel для бизнеса", data=_download_blob(st.session_state.results.get("excel_buffer") if has_results else b"", b""), file_name=f"pricing_report_{st.session_state.get('selected_sku_for_results', 'report')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", disabled=excel_disabled, use_container_width=True)
        close_surface()
        open_surface("CSV с дневными данными", "Детальный прогноз по дням для проверки цифр.")
        if csv_disabled:
            reason = "сначала рассчитайте сценарий" if not scenario_applied else "проверка корректности сценария не пройдена"
            st.caption(f"Недоступно: {reason}.")
        st.download_button("Скачать CSV с дневными данными", data=_download_blob(st.session_state.results.get("manual_scenario_daily_csv", b"") if has_results else b"", b""), file_name="scenario_daily.csv", mime="text/csv", disabled=csv_disabled, use_container_width=True)
        close_surface()
        with st.expander("Для аналитика: технический файл", expanded=False):
            if tech_disabled:
                reason = "сначала рассчитайте сценарий" if not scenario_applied else "проверка корректности сценария не пройдена"
                st.caption(f"Недоступно: {reason}.")
            st.download_button("Скачать технический файл", data=_download_blob(st.session_state.results.get("manual_scenario_summary_json", b"{}") if has_results else b"{}", b"{}"), file_name="summary.json", mime="application/json", disabled=tech_disabled, use_container_width=True)

    elif active_tab == PAGE_DIAGNOSTICS:
        render_page_header(
            PAGE_DIAGNOSTICS,
            "Техническая информация для аналитика: качество данных, режим расчёта, ограничения и служебные детали.",
        )
        wr_diag = st.session_state.what_if_result or {}
        diagnostic_payload = {
            "data_contract": r.get("data_contract"),
            "model_quality_gate": r.get("model_quality_gate"),
            "scenario_contract": wr_diag.get("scenario_contract"),
            "scenario_sensitivity_diagnostics": wr_diag.get("sensitivity_diagnostics"),
            "catboost_diagnostics": r.get("catboost_diagnostics"),
            "decision_passport": st.session_state.get("decision_passport"),
            "warnings": r.get("warnings", []),
            "scenario_calc_mode": r.get("analysis_scenario_calc_mode"),
        }
        open_surface("Сводка диагностики")
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("Качество данных", data_quality_ui_label(r.get("data_quality_label") or (r.get("data_quality_gate") or {}).get("status")))
        q2.metric("Режим расчёта", scenario_mode_label(str(r.get("analysis_scenario_calc_mode", DEFAULT_SCENARIO_CALC_MODE))))
        q3.metric("Предупреждения", str(len(r.get("warnings", []) or [])))
        q4.metric("Scenario contract", "Есть" if wr_diag.get("scenario_contract") else "Нет")
        q5.metric("Decision passport", "Есть" if st.session_state.get("decision_passport") else "Нет")
        close_surface()
        with st.expander("Для аналитика / технический JSON", expanded=False):
            st.json(diagnostic_payload, expanded=False)
        with st.expander("Raw tables", expanded=False):
            if isinstance(r.get("history_daily"), pd.DataFrame):
                st.dataframe(r.get("history_daily").head(100), use_container_width=True)
            if isinstance(r.get("as_is_forecast"), pd.DataFrame):
                st.dataframe(r.get("as_is_forecast"), use_container_width=True)


    st.caption("What-if Cloud")
