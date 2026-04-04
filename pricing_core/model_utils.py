from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import CONFIG

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False


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
            out[c] = out[c].astype(str).replace({"nan": "unknown", "None": "unknown"}).fillna("unknown")
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


def _numeric_for_non_catboost(X: pd.DataFrame, cat_features: List[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_features:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(float)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
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
    loss_function: str = "RMSE",
) -> List[Any]:
    _ = kind
    cat_features = cat_features or []
    if len(X) == 0:
        raise ValueError("Пустая обучающая выборка.")

    ensemble: List[Any] = []
    weights = np.ones(len(X)) if sample_weight is None else pd.to_numeric(sample_weight, errors="coerce").fillna(1.0).values

    mono = {}
    for k, v in {"price": -1, "freight_value": -1, "review_score": 1}.items():
        if k in feature_names:
            mono[k] = v

    for i in range(int(n_models)):
        if CATBOOST_AVAILABLE:
            model = CatBoostRegressor(
                loss_function="MAE" if str(loss_function).upper() == "MAE" else "RMSE",
                iterations=300 if small_mode else 500,
                depth=5 if small_mode else 6,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                subsample=0.8,
                bootstrap_type="Bernoulli",
                verbose=False,
                random_seed=42 + i,
                monotone_constraints=mono if mono else None,
            )
            model.fit(X[feature_names], y, cat_features=[c for c in cat_features if c in feature_names], sample_weight=weights)
            model.is_fallback = False
            ensemble.append(model)
        else:
            criterion = "absolute_error" if str(loss_function).upper() == "MAE" else "squared_error"
            model = RandomForestRegressor(
                n_estimators=int(CONFIG["RF_TREES"]),
                max_depth=int(CONFIG["RF_DEPTH"] if not small_mode else max(4, int(CONFIG["RF_DEPTH"]) // 2)),
                random_state=42 + i,
                n_jobs=1,
                criterion=criterion,
            )
            model.fit(_numeric_for_non_catboost(X[feature_names], cat_features), y, sample_weight=weights)
            model.is_fallback = True
            ensemble.append(model)
    return ensemble


def ensemble_predict(models_local: List[Any], X_local: pd.DataFrame, feature_names: Optional[List[str]] = None, cat_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if len(models_local) == 0:
        raise ValueError("No models in ensemble")
    feature_names = feature_names or list(X_local.columns)
    cat_features = cat_features or []
    preds = []
    for m in models_local:
        if getattr(m, "is_fallback", False):
            preds.append(m.predict(_numeric_for_non_catboost(X_local[feature_names], cat_features)))
        else:
            preds.append(m.predict(X_local[feature_names]))
    mat = np.vstack(preds)
    return mat.mean(axis=0), mat.std(axis=0, ddof=0)
