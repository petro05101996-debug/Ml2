from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pricing_core.model_utils import HAS_XGBOOST, XGBRegressor
from pricing_core.weekly_forecast_features import build_recursive_weekly_features


BASE_NUMERIC_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_8",
    "lag_12",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "rolling_std_8",
    "rolling_min_4",
    "rolling_max_4",
    "price_week",
    "discount_week",
    "promo_week",
    "price_rel_to_rolling_8w_median",
    "price_rel_to_rolling_12w_mean",
    "discount_depth",
    "promo_x_discount",
    "price_x_promo",
    "is_year_start",
    "is_year_end",
]

BASE_CATEGORICAL_FEATURES = ["month", "quarter", "week_of_year"]


def _prepare_matrix(df: pd.DataFrame, features: List[str], fit_cols: List[str] | None = None) -> pd.DataFrame:
    x = df[features].copy()
    x = pd.get_dummies(x, columns=[c for c in x.columns if x[c].dtype == "object"], dummy_na=True)
    if fit_cols is not None:
        for c in fit_cols:
            if c not in x.columns:
                x[c] = 0.0
        x = x[fit_cols]
    return x.fillna(0.0)


def train_weekly_forecast_model(train_df: pd.DataFrame, feature_spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    user_num = [c for c in train_df.columns if str(c).startswith("user_factor_num__")]
    user_cat = [c for c in train_df.columns if str(c).startswith("user_factor_cat__")]
    numeric = [c for c in BASE_NUMERIC_FEATURES if c in train_df.columns] + user_num
    categorical = [c for c in BASE_CATEGORICAL_FEATURES if c in train_df.columns] + user_cat
    features = numeric + categorical

    train = train_df.dropna(subset=["sales_week"]).copy()
    y = np.log1p(pd.to_numeric(train["sales_week"], errors="coerce").fillna(0.0).clip(lower=0.0))
    x = _prepare_matrix(train, features)

    if HAS_XGBOOST:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(x, y)
    return {"model": model, "features": features, "matrix_cols": list(x.columns)}


def predict_weekly_forecast(model_bundle: Dict[str, Any], feature_df: pd.DataFrame) -> pd.Series:
    x = _prepare_matrix(feature_df, model_bundle["features"], fit_cols=model_bundle["matrix_cols"])
    pred = model_bundle["model"].predict(x)
    return pd.Series(np.expm1(pred).clip(min=0.0), index=feature_df.index)


def recursive_weekly_forecast(model_bundle: Dict[str, Any], history_df: pd.DataFrame, future_exog_df: pd.DataFrame) -> pd.DataFrame:
    hist = history_df.copy().sort_values("week_start")
    rows: List[Dict[str, Any]] = []
    for _, future_row in future_exog_df.sort_values("week_start").iterrows():
        row = {k: future_row.get(k) for k in future_exog_df.columns}
        row["sales_week"] = np.nan
        tmp = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        tmp = build_recursive_weekly_features(tmp)
        step = tmp.tail(1)
        pred = float(predict_weekly_forecast(model_bundle, step).iloc[0])
        row["sales_week"] = pred
        rows.append({"week_start": pd.Timestamp(row["week_start"]), "sales_week": pred})
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(rows)
