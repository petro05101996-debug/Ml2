from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    class XGBRegressor(HistGradientBoostingRegressor):  # type: ignore
        def __init__(self, **kwargs):
            self._xgb_params = dict(kwargs)
            super().__init__(max_depth=kwargs.get("max_depth", 6), learning_rate=kwargs.get("learning_rate", 0.05), max_iter=kwargs.get("n_estimators", 250), random_state=kwargs.get("random_state", 42))

        def get_params(self, deep: bool = True):
            p = super().get_params(deep=deep)
            p.update(self._xgb_params)
            return p

from .config import CONFIG


def _prepare_xgb_input(X: pd.DataFrame, feature_names: List[str], cat_features: List[str]) -> pd.DataFrame:
    out = X[feature_names].copy()
    if HAS_XGBOOST:
        for c in cat_features:
            if c in out.columns:
                out[c] = out[c].astype("category")
        return out
    for c in cat_features:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def clean_feature_frame(
    df: pd.DataFrame,
    features: List[str],
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    feature_stats: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    out = df.copy()
    stats = feature_stats or {}
    num_set = set(numeric_features or [])
    cat_set = set(categorical_features or [])
    for c in features:
        if c not in out.columns:
            out[c] = np.nan
        if c in cat_set:
            out[c] = out[c].astype(object).fillna("unknown").astype("category")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fallback = float(stats.get(c, 0.0)) if isinstance(stats, dict) else 0.0
            med = float(out[c].median()) if out[c].notna().any() else fallback
            if not np.isfinite(med):
                med = fallback
            out[c] = out[c].fillna(med)
            if c in num_set:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(med)
    return out


def build_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_models: int = CONFIG["ENSEMBLE_SIZE"],
    kind: str = "demand",
    small_mode: bool = False,
    cat_features: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None,
    loss_function: str = "MAE",
    training_profile: str = "final",
) -> List[Any]:
    _ = (kind, loss_function)
    cat_features = cat_features or []
    if len(X) == 0:
        raise ValueError("Пустая обучающая выборка.")

    monotone_map = {"price": -1, "freight_value": -1, "review_score": 1}
    monotone_constraints_tuple = tuple(monotone_map.get(f, 0) for f in feature_names)

    ensemble: List[Any] = []
    n_estimators = 700 if not small_mode else 400
    if str(training_profile) == "backtest":
        n_estimators = 350 if not small_mode else 200
    for i in range(int(n_models)):
        model = XGBRegressor(
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            tree_method="hist",
            enable_categorical=True,
            n_estimators=n_estimators,
            max_depth=4 if not small_mode else 3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.65,
            min_child_weight=12,
            reg_lambda=4.0,
            reg_alpha=1.0,
            max_cat_to_onehot=8,
            random_state=42 + i,
            monotone_constraints=monotone_constraints_tuple,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = pd.to_numeric(sample_weight, errors="coerce").fillna(1.0).values
        model.fit(_prepare_xgb_input(X, feature_names, cat_features), y, **fit_kwargs)
        model.is_fallback = False
        ensemble.append(model)
    return ensemble


def build_poisson_fallback(X_num: pd.DataFrame, y: pd.Series) -> PoissonRegressor:
    model = PoissonRegressor(alpha=0.0001, max_iter=1000)
    model.fit(X_num, y)
    model.is_fallback = True
    return model


def ensemble_predict(models_local: List[Any], X_local: pd.DataFrame, feature_names: Optional[List[str]] = None, cat_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if len(models_local) == 0:
        raise ValueError("No models in ensemble")
    feature_names = feature_names or list(X_local.columns)
    preds = [m.predict(_prepare_xgb_input(X_local, feature_names, cat_features or [])) for m in models_local]
    mat = np.vstack(preds)
    return mat.mean(axis=0), mat.std(axis=0, ddof=0)
