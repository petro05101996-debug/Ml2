from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .model_utils import HAS_XGBOOST, build_models, build_poisson_fallback, clean_feature_frame, ensemble_predict
from .v1_features import build_v1_one_step_features


def _build_demand_sample_weights(train_df: pd.DataFrame) -> pd.Series:
    if len(train_df) == 0:
        return pd.Series(dtype=float)
    sales = pd.to_numeric(train_df.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    promo = pd.to_numeric(train_df.get("promotion", 0.0), errors="coerce").fillna(0.0)
    discount = pd.to_numeric(train_df.get("discount", 0.0), errors="coerce").fillna(0.0)

    w = pd.Series(np.ones(len(train_df), dtype=float), index=train_df.index)
    med_sales = float(sales.median()) if len(sales) else 0.0
    med_discount = float(discount.median()) if len(discount) else 0.0

    if sales.nunique(dropna=True) >= 4:
        q80 = float(sales.quantile(0.80))
        if q80 > med_sales:
            w.loc[sales > q80] += 0.8
    w.loc[promo > 0] += 0.3
    w.loc[discount > med_discount] += 0.2
    return w.clip(lower=1.0, upper=2.3)


def train_v1_demand_model(
    train_df: pd.DataFrame,
    feature_spec: Dict[str, Any],
    small_mode: bool = False,
    training_profile: str = "final",
) -> Dict[str, Any]:
    feats = list(feature_spec.get("demand_features", []))
    cat_feats = list(feature_spec.get("cat_features_demand", []))
    num_feats = [c for c in feature_spec.get("numeric_demand_features", feats) if c in feats]
    X = clean_feature_frame(train_df, feats, numeric_features=num_feats, categorical_features=cat_feats)[feats]
    y = pd.to_numeric(train_df["sales"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weights = _build_demand_sample_weights(train_df)
    main_models = build_models(
        X,
        y,
        feats,
        kind="demand",
        small_mode=small_mode,
        cat_features=cat_feats,
        sample_weight=weights,
        loss_function="MAE",
        training_profile=training_profile,
    )
    fallback_model = build_poisson_fallback(X[num_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0), y)
    model_backend = "xgboost" if HAS_XGBOOST else "hist_gradient_boosting_fallback"
    return {
        "main_models": main_models,
        "fallback_model": fallback_model,
        "feature_spec": feature_spec,
        "training_weight_min": float(weights.min()) if len(weights) else 1.0,
        "training_weight_max": float(weights.max()) if len(weights) else 1.0,
        "training_weight_mean": float(weights.mean()) if len(weights) else 1.0,
        "model_backend": model_backend,
    }


def predict_v1_demand(X: pd.DataFrame, trained_models: Dict[str, Any], feature_spec: Dict[str, Any], forecast_mode: str = "strong_signal") -> pd.DataFrame:
    feats = list(feature_spec.get("demand_features", []))
    cat_feats = list(feature_spec.get("cat_features_demand", []))
    num_feats = [c for c in feature_spec.get("numeric_demand_features", feats) if c in feats]
    xf = clean_feature_frame(X, feats, numeric_features=num_feats, categorical_features=cat_feats)

    main_models = trained_models["main_models"]
    if main_models and not hasattr(main_models[0], "get_params"):
        raw = X.copy()
        pred_main, _ = ensemble_predict(main_models, raw, feature_names=list(raw.columns), cat_features=[c for c in cat_feats if c in raw.columns])
    else:
        pred_main, _ = ensemble_predict(main_models, xf[feats], feature_names=feats, cat_features=cat_feats)

    try:
        pred_fb = trained_models["fallback_model"].predict(xf[num_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    except Exception:
        pred_fb = trained_models["fallback_model"].predict(X)

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
    weekday_factors: Dict[int, float] | None = None,
) -> pd.DataFrame:
    hist = base_history.copy().sort_values("date").reset_index(drop=True)
    history_span_days = int((hist["date"].max() - hist["date"].min()).days + 1) if len(hist) else 1
    rows = []
    for dt in pd.to_datetime(future_dates_df["date"]):
        step = build_v1_one_step_features(hist, pd.Timestamp(dt), base_ctx, history_span_days, feature_spec)
        pred_row = predict_v1_demand(step, trained_models, feature_spec, forecast_mode=forecast_mode).iloc[0].to_dict()
        wd_factor = float(weekday_factors.get(int(pd.Timestamp(dt).dayofweek), 1.0)) if weekday_factors else 1.0
        combined_factor = float(calibration_factor) * wd_factor
        pred_row = {k: max(0.0, float(v) * combined_factor) for k, v in pred_row.items()}

        row = {"date": pd.Timestamp(dt), **pred_row}
        for c in feature_spec.get("scenario_features", []):
            row[c] = float(pd.to_numeric(pd.Series([base_ctx.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        for c in feature_spec.get("categorical_demand_features", []):
            row[c] = str(base_ctx.get(c, "unknown"))
        row["cost"] = float(pd.to_numeric(pd.Series([base_ctx.get("cost", 0.0)]), errors="coerce").fillna(0.0).iloc[0])

        rows.append(row)
        hist = pd.concat([hist, pd.DataFrame([{"date": pd.Timestamp(dt), "sales": row["pred_sales"], **{k: row[k] for k in row if k not in {"pred_sales", "pred_sales_main", "pred_sales_fallback"}}}])], ignore_index=True)
    return pd.DataFrame(rows)
