from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .model_utils import build_models, build_poisson_fallback, clean_feature_frame, ensemble_predict
from .v1_features import build_v1_one_step_features


def train_v1_demand_model(train_df: pd.DataFrame, feature_spec: Dict[str, Any], small_mode: bool = False) -> Dict[str, Any]:
    feats = list(feature_spec.get("demand_features", []))
    cat_feats = list(feature_spec.get("cat_features_demand", []))
    num_feats = [c for c in feature_spec.get("numeric_demand_features", feats) if c in feats]
    X = clean_feature_frame(train_df, feats, numeric_features=num_feats, categorical_features=cat_feats)[feats]
    y = pd.to_numeric(train_df["sales"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = pd.Series(np.ones(len(X)), index=train_df.index)
    main_models = build_models(X, y, feats, kind="demand", small_mode=small_mode, cat_features=cat_feats, sample_weight=weights, loss_function="MAE")
    fallback_model = build_poisson_fallback(X[num_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0), y)
    return {
        "main_models": main_models,
        "fallback_model": fallback_model,
        "feature_spec": feature_spec,
    }


def predict_v1_demand(X: pd.DataFrame, trained_models: Dict[str, Any], feature_spec: Dict[str, Any], forecast_mode: str = "strong_signal") -> pd.DataFrame:
    feats = list(feature_spec.get("demand_features", []))
    cat_feats = list(feature_spec.get("cat_features_demand", []))
    num_feats = [c for c in feature_spec.get("numeric_demand_features", feats) if c in feats]
    xf = clean_feature_frame(X, feats, numeric_features=num_feats, categorical_features=cat_feats)

    pred_main, _ = ensemble_predict(trained_models["main_models"], xf[feats], feature_names=feats, cat_features=cat_feats)
    pred_fb = trained_models["fallback_model"].predict(xf[num_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0))

    pred_main = np.clip(pred_main.astype(float), 0.0, None)
    pred_fb = np.clip(np.asarray(pred_fb, dtype=float), 0.0, None)
    if forecast_mode == "strong_signal":
        pred_final = pred_main
    else:
        pred_final = np.minimum(pred_main, pred_fb)
    return pd.DataFrame({"pred_sales": pred_final, "pred_sales_main": pred_main, "pred_sales_fallback": pred_fb})


def recursive_v1_demand_forecast(
    trained_models: Dict[str, Any],
    base_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
    calibration_factor: float = 1.0,
    forecast_mode: str = "strong_signal",
) -> pd.DataFrame:
    hist = base_history.copy().sort_values("date").reset_index(drop=True)
    history_span_days = int((hist["date"].max() - hist["date"].min()).days + 1) if len(hist) else 1
    rows = []
    for dt in pd.to_datetime(future_dates_df["date"]):
        step = build_v1_one_step_features(hist, pd.Timestamp(dt), base_ctx, history_span_days, feature_spec)
        pred_row = predict_v1_demand(step, trained_models, feature_spec, forecast_mode=forecast_mode).iloc[0].to_dict()
        pred_row = {k: max(0.0, float(v) * float(calibration_factor)) for k, v in pred_row.items()}

        row = {"date": pd.Timestamp(dt), **pred_row}
        for c in feature_spec.get("scenario_features", []):
            row[c] = float(pd.to_numeric(pd.Series([base_ctx.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        for c in feature_spec.get("categorical_demand_features", []):
            row[c] = str(base_ctx.get(c, "unknown"))
        row["cost"] = float(pd.to_numeric(pd.Series([base_ctx.get("cost", 0.0)]), errors="coerce").fillna(0.0).iloc[0])

        rows.append(row)
        hist = pd.concat([hist, pd.DataFrame([{"date": pd.Timestamp(dt), "sales": row["pred_sales"], **{k: row[k] for k in row if k not in {"pred_sales", "pred_sales_main", "pred_sales_fallback"}}}])], ignore_index=True)
    return pd.DataFrame(rows)
