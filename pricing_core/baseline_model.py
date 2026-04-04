from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pricing_core.baseline_features import build_baseline_one_step_features
from pricing_core.model_utils import HAS_XGBOOST, XGBRegressor, clean_feature_frame


def _params(training_profile: str, small_mode: bool) -> Dict[str, Any]:
    if training_profile == "backtest":
        return {
            "n_estimators": 150 if small_mode else 250,
            "max_depth": 4 if small_mode else 5,
        }
    return {
        "n_estimators": 250 if small_mode else 500,
        "max_depth": 4 if small_mode else 5,
    }


def train_baseline_model(train_df: pd.DataFrame, feature_spec: Dict[str, Any], small_mode: bool = False, training_profile: str = "final") -> Dict[str, Any]:
    feats = feature_spec.get("baseline_features", [])
    cat = feature_spec.get("cat_features", [])
    X = clean_feature_frame(train_df, feats, feature_spec.get("baseline_numeric_features", []), cat)
    y = pd.to_numeric(train_df.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    p = _params(training_profile, small_mode)
    if HAS_XGBOOST:
        model = XGBRegressor(
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            tree_method="hist",
            enable_categorical=True,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=8,
            reg_lambda=3.0,
            reg_alpha=0.5,
            random_state=42,
            **p,
        )
        for c in cat:
            if c in X:
                X[c] = X[c].astype("category")
        model.fit(X[feats], y)
        backend, is_fallback = "xgboost", False
    else:
        X_num = pd.get_dummies(X[feats], columns=[c for c in cat if c in X.columns], dummy_na=True)
        model = HistGradientBoostingRegressor(max_depth=p["max_depth"], learning_rate=0.03, max_iter=p["n_estimators"], random_state=42)
        model.fit(X_num, y)
        backend, is_fallback = "hist_gradient_boosting", True

    return {"model": model, "feature_spec": feature_spec, "model_backend": backend, "training_profile": training_profile, "is_fallback": is_fallback}


def predict_baseline(X: pd.DataFrame, trained_baseline: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.Series:
    feats = feature_spec.get("baseline_features", [])
    cat = feature_spec.get("cat_features", [])
    Xc = clean_feature_frame(X, feats, feature_spec.get("baseline_numeric_features", []), cat)
    model = trained_baseline["model"]
    if trained_baseline.get("model_backend") == "hist_gradient_boosting":
        Xp = pd.get_dummies(Xc[feats], columns=[c for c in cat if c in Xc.columns], dummy_na=True)
        # align columns lazily
        if hasattr(model, "feature_names_in_"):
            for c in model.feature_names_in_:
                if c not in Xp.columns:
                    Xp[c] = 0.0
            Xp = Xp[list(model.feature_names_in_)]
        pred = model.predict(Xp)
    else:
        for c in cat:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype("category")
        pred = model.predict(Xc[feats])
    return pd.Series(np.clip(np.asarray(pred, dtype=float), 0.0, np.inf), index=X.index)


def recursive_baseline_forecast(trained_baseline: Dict[str, Any], base_history: pd.DataFrame, future_dates_df: pd.DataFrame, base_ctx: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.DataFrame:
    hist = base_history.copy().sort_values("date")
    rows: List[Dict[str, Any]] = []
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        step = build_baseline_one_step_features(hist, dt, base_ctx, feature_spec)
        pred = float(predict_baseline(step, trained_baseline, feature_spec).iloc[0])
        rows.append({"date": pd.Timestamp(dt), "baseline_pred": max(0.0, pred)})
        hist = pd.concat([hist, pd.DataFrame([{"date": pd.Timestamp(dt), "sales": max(0.0, pred), **{k: v for k, v in base_ctx.items() if k in ["product_id", "category", "region", "channel", "segment"]}}])], ignore_index=True)
    return pd.DataFrame(rows)


def run_baseline_holdout(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(actual, errors="coerce").fillna(0.0)
    p = pd.to_numeric(pred, errors="coerce").fillna(0.0)
    denom = float(np.abs(a).sum())
    wape = 100.0 if denom <= 1e-9 else float(np.abs(a - p).sum() / denom * 100.0)
    mae = float(np.abs(a - p).mean()) if len(a) else 0.0
    rmse = float(np.sqrt(np.mean((a - p) ** 2))) if len(a) else 0.0
    asum, psum = float(a.sum()), float(p.sum())
    bias_pct = 0.0 if abs(asum) <= 1e-9 else float((psum - asum) / asum)
    sum_ratio = 0.0 if abs(asum) <= 1e-9 else float(psum / asum)
    return {"forecast_wape": wape, "mae": mae, "rmse": rmse, "bias_pct": bias_pct, "sum_ratio": sum_ratio}


def run_baseline_rolling_backtest(panel_train: pd.DataFrame, target_category: str, target_sku: str, n_windows: int = 3, window_days: int = 28, min_train_days: int = 120) -> Dict[str, Any]:
    from pricing_core.baseline_features import derive_baseline_feature_spec

    target = panel_train[(panel_train["category"].astype(str) == str(target_category)) & (panel_train["product_id"].astype(str) == str(target_sku))].copy().sort_values("date")
    max_date = pd.Timestamp(target["date"].max()) if len(target) else pd.Timestamp("1970-01-01")
    starts = sorted([max_date - pd.Timedelta(days=(window_days * (i + 1) - 1)) for i in range(int(n_windows))])
    diag_rows, metric_rows = [], []
    for i, ws in enumerate(starts, start=1):
        we = ws + pd.Timedelta(days=window_days)
        panel_w = panel_train[panel_train["date"] < ws].copy()
        train_t = target[target["date"] < ws].copy()
        test_t = target[(target["date"] >= ws) & (target["date"] < we)].copy()
        if train_t["date"].nunique() < int(min_train_days) or test_t.empty:
            continue
        spec = derive_baseline_feature_spec(panel_w)
        trained = train_baseline_model(panel_w, spec, small_mode=len(panel_w) < 200, training_profile="backtest")
        base_ctx = {k: (train_t[k].dropna().astype(str).iloc[-1] if k in ["product_id", "category", "region", "channel", "segment"] and k in train_t else "unknown") for k in ["product_id", "category", "region", "channel", "segment"]}
        fut = pd.DataFrame({"date": test_t["date"].values})
        fc = recursive_baseline_forecast(trained, train_t, fut, base_ctx, spec)
        merged = test_t[["date", "sales"]].merge(fc, on="date", how="left")
        merged["window_id"] = i
        merged["window_start"] = ws
        merged["window_end"] = we
        diag_rows.append(merged)
        metric_rows.append({"window_id": i, "window_start": ws, "window_end": we, **run_baseline_holdout(merged["sales"], merged["baseline_pred"])})
    rolling_diag = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame(columns=["date", "sales", "baseline_pred", "window_id", "window_start", "window_end"])
    rolling_metrics = pd.DataFrame(metric_rows)
    summary = {
        "n_valid_windows": int(len(rolling_metrics)),
        "median_wape": float(rolling_metrics["forecast_wape"].median()) if len(rolling_metrics) else np.nan,
        "median_bias_pct": float(rolling_metrics["bias_pct"].median()) if len(rolling_metrics) else np.nan,
        "median_sum_ratio": float(rolling_metrics["sum_ratio"].median()) if len(rolling_metrics) else np.nan,
        "max_wape": float(rolling_metrics["forecast_wape"].max()) if len(rolling_metrics) else np.nan,
    }
    return {"rolling_diag": rolling_diag, "rolling_metrics": rolling_metrics, "rolling_summary": summary}


def build_baseline_oof_predictions(
    panel_train: pd.DataFrame,
    target_category: str,
    target_sku: str,
    min_train_days: int = 84,
    step_days: int = 7,
    horizon_days: int = 7,
) -> pd.DataFrame:
    from pricing_core.baseline_features import derive_baseline_feature_spec

    target = panel_train[(panel_train["category"].astype(str) == str(target_category)) & (panel_train["product_id"].astype(str) == str(target_sku))].copy().sort_values("date")
    out = target[["date", "sales"]].copy()
    out["baseline_oof"] = np.nan
    if target.empty:
        return out

    target = target.reset_index(drop=True)
    n = len(target)
    start_idx = int(max(min_train_days, 1))

    while start_idx < n:
        train_end_date = pd.Timestamp(target.iloc[start_idx - 1]["date"])
        test_slice = target.iloc[start_idx:start_idx + int(horizon_days)].copy()
        if test_slice.empty:
            break
        test_dates_df = pd.DataFrame({"date": pd.to_datetime(test_slice["date"].values)})

        panel_w = panel_train[pd.to_datetime(panel_train["date"]) <= train_end_date].copy()
        target_w = target[pd.to_datetime(target["date"]) <= train_end_date].copy()
        if target_w["date"].nunique() < int(min_train_days) or panel_w.empty:
            start_idx += int(step_days)
            continue

        spec_w = derive_baseline_feature_spec(panel_w)
        trained_w = train_baseline_model(panel_w, spec_w, small_mode=len(panel_w) < 200, training_profile="backtest")
        base_ctx_w = {
            k: (target_w[k].dropna().astype(str).iloc[-1] if k in target_w.columns and k in ["product_id", "category", "region", "channel", "segment"] else "unknown")
            for k in ["product_id", "category", "region", "channel", "segment"]
        }
        fc_w = recursive_baseline_forecast(trained_w, target_w, test_dates_df, base_ctx_w, spec_w)
        pred_map = dict(zip(pd.to_datetime(fc_w["date"]), pd.to_numeric(fc_w["baseline_pred"], errors="coerce")))

        mask = out["date"].isin(test_slice["date"]) & out["baseline_oof"].isna()
        out.loc[mask, "baseline_oof"] = pd.to_datetime(out.loc[mask, "date"]).map(pred_map)
        start_idx += int(step_days)

    return out.sort_values("date").reset_index(drop=True)
