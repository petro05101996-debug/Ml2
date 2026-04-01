from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .core import build_models, clean_feature_frame
from .v1_features import build_v1_one_step_features


def train_v1_baseline_model(
    train_df: pd.DataFrame,
    feature_spec: Dict[str, Any],
    small_mode: bool = False,
) -> List[Any]:
    feats = list(feature_spec.get("baseline_features", []))
    X = clean_feature_frame(train_df, feats)[feats]
    y = pd.to_numeric(train_df.get("log_sales", np.log1p(train_df.get("sales", 0.0))), errors="coerce").fillna(0.0)
    return build_models(X, y, feats, kind="baseline", small_mode=small_mode, cat_features=feature_spec.get("cat_features_baseline", []))


def predict_v1_baseline_log(
    frame: pd.DataFrame,
    baseline_models: List[Any],
    feature_spec: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    feats = list(feature_spec.get("baseline_features", []))
    X = clean_feature_frame(frame, feats)[feats]
    if not baseline_models:
        pred = np.zeros(len(X), dtype=float)
        return pred, np.zeros(len(X), dtype=float)
    preds = np.column_stack([m.predict(X) for m in baseline_models]).astype(float)
    mean_pred = np.mean(preds, axis=1)
    std_pred = np.std(preds, axis=1)
    return mean_pred, std_pred


def recursive_v1_baseline_forecast(
    baseline_models: List[Any],
    history_df: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> pd.DataFrame:
    hist = history_df.copy().sort_values("date").reset_index(drop=True)
    history_span_days = int((hist["date"].max() - hist["date"].min()).days + 1) if len(hist) else 1
    rows = []
    for dt in pd.to_datetime(future_dates_df["date"]):
        step = build_v1_one_step_features(hist, pd.Timestamp(dt), base_ctx, history_span_days)
        pred_log, pred_std = predict_v1_baseline_log(step, baseline_models, feature_spec)
        baseline_log = float(pred_log[0])
        baseline_sales = float(max(np.expm1(baseline_log), 0.0))
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "baseline_sales": baseline_sales,
                "baseline_log": baseline_log,
                "baseline_std": float(pred_std[0]),
                "price": float(base_ctx.get("price", 0.0)),
                "cost": float(base_ctx.get("cost", 0.0)),
                "discount": float(base_ctx.get("discount", 0.0)),
                "promotion": float(base_ctx.get("promotion", 0.0)),
                "freight_value": float(base_ctx.get("freight_value", 0.0)),
                "review_score": float(base_ctx.get("review_score", 4.5)),
            }
        )
        hist = pd.concat(
            [
                hist,
                pd.DataFrame(
                    [
                        {
                            "date": pd.Timestamp(dt),
                            "sales": baseline_sales,
                            "price": float(base_ctx.get("price", 0.0)),
                            "cost": float(base_ctx.get("cost", 0.0)),
                            "discount": float(base_ctx.get("discount", 0.0)),
                            "promotion": float(base_ctx.get("promotion", 0.0)),
                            "freight_value": float(base_ctx.get("freight_value", 0.0)),
                            "review_score": float(base_ctx.get("review_score", 4.5)),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(rows)
