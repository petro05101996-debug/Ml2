from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .model_utils import build_models, clean_feature_frame, ensemble_predict
from .v1_features import build_v1_one_step_features


def train_v1_demand_model(train_df: pd.DataFrame, feature_spec: Dict[str, Any], small_mode: bool = False) -> List[Any]:
    feats = list(feature_spec.get("demand_features", []))
    cat_feats = list(feature_spec.get("cat_features_demand", []))
    num_feats = [c for c in feature_spec.get("numeric_demand_features", feats) if c in feats]
    X = clean_feature_frame(train_df, feats, numeric_features=num_feats, categorical_features=cat_feats)[feats]
    y = pd.to_numeric(train_df["sales"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = pd.Series(np.ones(len(X)), index=train_df.index)
    return build_models(X, y, feats, kind="demand", small_mode=small_mode, cat_features=cat_feats, sample_weight=weights, loss_function="MAE")


def predict_v1_demand(X: pd.DataFrame, demand_models: List[Any], feature_spec: Dict[str, Any]) -> np.ndarray:
    feats = list(feature_spec.get("demand_features", []))
    cat_feats = list(feature_spec.get("cat_features_demand", []))
    num_feats = [c for c in feature_spec.get("numeric_demand_features", feats) if c in feats]
    xf = clean_feature_frame(X, feats, numeric_features=num_feats, categorical_features=cat_feats)[feats]
    pred, _ = ensemble_predict(demand_models, xf, feature_names=feats, cat_features=cat_feats)
    return np.clip(pred.astype(float), 0.0, None)


def recursive_v1_demand_forecast(
    demand_models: List[Any],
    base_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
    calibration_factor: float = 1.0,
) -> pd.DataFrame:
    hist = base_history.copy().sort_values("date").reset_index(drop=True)
    history_span_days = int((hist["date"].max() - hist["date"].min()).days + 1) if len(hist) else 1
    rows = []
    for dt in pd.to_datetime(future_dates_df["date"]):
        step = build_v1_one_step_features(hist, pd.Timestamp(dt), base_ctx, history_span_days, feature_spec)
        pred_sales = float(predict_v1_demand(step, demand_models, feature_spec)[0])
        pred_sales = max(0.0, pred_sales * float(calibration_factor))

        row = {"date": pd.Timestamp(dt), "pred_sales": pred_sales}
        for c in feature_spec.get("scenario_features", []):
            row[c] = float(pd.to_numeric(pd.Series([base_ctx.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        for c in feature_spec.get("categorical_demand_features", []):
            row[c] = str(base_ctx.get(c, "unknown"))
        row["cost"] = float(pd.to_numeric(pd.Series([base_ctx.get("cost", 0.0)]), errors="coerce").fillna(0.0).iloc[0])

        rows.append(row)
        hist = pd.concat([hist, pd.DataFrame([{"date": pd.Timestamp(dt), "sales": pred_sales, **{k: row[k] for k in row if k not in {"pred_sales"}}}])], ignore_index=True)
    return pd.DataFrame(rows)
