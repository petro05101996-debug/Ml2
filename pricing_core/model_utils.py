from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import CONFIG


def clean_feature_frame(df: pd.DataFrame, features: List[str], feature_stats: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    out = df.copy()
    stats = feature_stats or {}
    for c in features:
        if c not in out.columns:
            out[c] = stats.get(c, 0.0) if isinstance(stats, dict) else 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")
        fallback = float(stats.get(c, 0.0)) if isinstance(stats, dict) else 0.0
        med = float(out[c].median()) if not out[c].isna().all() and np.isfinite(out[c].median()) else fallback
        out[c] = out[c].fillna(med).replace([np.inf, -np.inf], med)
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
    ensemble: List[Any] = []
    n_rows = len(X)
    if n_rows == 0:
        raise ValueError("Пустая обучающая выборка.")

    _ = kind
    eff_loss = str(loss_function).upper()
    sample_size = min(n_rows, int(CONFIG["MAX_TRAIN_ROWS_PER_MODEL"]))
    n_trees = int(CONFIG["RF_TREES"])
    if eff_loss == "MAE":
        n_trees = max(120, n_trees // 2)

    for i in range(int(n_models)):
        rng = np.random.default_rng(42 + i)
        idx = rng.choice(n_rows, size=sample_size, replace=(sample_size >= n_rows))
        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=int(CONFIG["RF_DEPTH"] if not small_mode else max(4, int(CONFIG["RF_DEPTH"]) // 2)),
            random_state=42 + i,
            n_jobs=1,
        )
        weights = None if sample_weight is None else pd.to_numeric(sample_weight.iloc[idx], errors="coerce").fillna(1.0).values
        model.fit(_numeric_for_non_catboost(X.iloc[idx][feature_names], cat_features or []), y.iloc[idx], sample_weight=weights)
        ensemble.append(model)
    return ensemble


def ensemble_predict(models_local: List[Any], X_local: pd.DataFrame, feature_names: Optional[List[str]] = None, cat_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if len(models_local) == 0:
        raise ValueError("No models in ensemble")
    feature_names = feature_names or list(X_local.columns)
    preds = np.vstack([m.predict(_numeric_for_non_catboost(X_local[feature_names], cat_features or [])) for m in models_local])
    return preds.mean(axis=0), preds.std(axis=0, ddof=0)
