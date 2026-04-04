from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pricing_core.model_utils import HAS_XGBOOST, XGBRegressor, clean_feature_frame


def _params(training_profile: str, small_mode: bool) -> Dict[str, Any]:
    if training_profile == "backtest":
        return {"n_estimators": 120 if small_mode else 180, "max_depth": 3 if small_mode else 4}
    return {"n_estimators": 200 if small_mode else 350, "max_depth": 3 if small_mode else 4}


def _monotone_tuple(feature_names: List[str]) -> tuple:
    m = {"price_rel_to_recent_median_28": -1, "discount_rate": 1, "promo_flag": 1, "stock": 1}
    return tuple(int(m.get(f, 0)) for f in feature_names)


def train_factor_model(train_df: pd.DataFrame, feature_spec: Dict[str, Any], small_mode: bool = False, training_profile: str = "final") -> Dict[str, Any]:
    feats = feature_spec.get("factor_features", [])
    cat = feature_spec.get("factor_categorical_features", [])
    X = clean_feature_frame(train_df, feats, feature_spec.get("factor_numeric_features", []), cat)
    y = pd.to_numeric(train_df.get("factor_target", 0.0), errors="coerce").fillna(0.0)
    p = _params(training_profile, small_mode)
    if HAS_XGBOOST:
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_lambda=4.0,
            reg_alpha=1.0,
            monotone_constraints=_monotone_tuple(feats),
            random_state=42,
            **p,
        )
        for c in cat:
            if c in X.columns:
                X[c] = X[c].astype("category")
        model.fit(X[feats], y)
        backend, is_fallback = "xgboost", False
    else:
        X_num = pd.get_dummies(X[feats], columns=[c for c in cat if c in X.columns], dummy_na=True)
        model = HistGradientBoostingRegressor(max_depth=p["max_depth"], max_iter=p["n_estimators"], learning_rate=0.03, random_state=42)
        model.fit(X_num, y)
        backend, is_fallback = "hist_gradient_boosting", True
    return {"model": model, "feature_spec": feature_spec, "model_backend": backend, "training_profile": training_profile, "is_fallback": is_fallback}


def predict_factor_effect(X: pd.DataFrame, trained_factor: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.DataFrame:
    feats = feature_spec.get("factor_features", [])
    cat = feature_spec.get("factor_categorical_features", [])
    Xc = clean_feature_frame(X, feats, feature_spec.get("factor_numeric_features", []), cat)
    m = trained_factor["model"]
    if trained_factor.get("model_backend") == "hist_gradient_boosting":
        Xp = pd.get_dummies(Xc[feats], columns=[c for c in cat if c in Xc.columns], dummy_na=True)
        if hasattr(m, "feature_names_in_"):
            for c in m.feature_names_in_:
                if c not in Xp.columns:
                    Xp[c] = 0.0
            Xp = Xp[list(m.feature_names_in_)]
        log_eff = m.predict(Xp)
    else:
        for c in cat:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype("category")
        log_eff = m.predict(Xc[feats])
    mult = np.clip(np.exp(np.asarray(log_eff, dtype=float)), 0.25, 4.0)
    return pd.DataFrame({"factor_log_effect": log_eff, "factor_multiplier": mult}, index=X.index)


def _price_sign_stability(df: pd.DataFrame, trained_factor: Dict[str, Any], feature_spec: Dict[str, Any]) -> float:
    if len(df) == 0:
        return 0.0
    sample = df.tail(min(20, len(df))).copy()
    base = predict_factor_effect(sample, trained_factor, feature_spec)["factor_multiplier"].values
    up_df = sample.copy()
    up_df["price_rel_to_recent_median_28"] = pd.to_numeric(up_df.get("price_rel_to_recent_median_28", 0.0), errors="coerce").fillna(0.0) + 0.1
    up = predict_factor_effect(up_df, trained_factor, feature_spec)["factor_multiplier"].values
    return float(np.mean(up <= base + 1e-9))


def build_factor_ood_flags(target_history: pd.DataFrame, scenario_frame: pd.DataFrame, feature_spec: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    hist = target_history.copy()
    if "price_rel_to_recent_median_28" not in hist.columns and "price" in hist.columns:
        ph = pd.to_numeric(hist.get("price"), errors="coerce")
        ref = float(ph.tail(28).median()) if ph.notna().any() else 1.0
        if not np.isfinite(ref) or ref <= 0:
            ref = 1.0
        hist["price_rel_to_recent_median_28"] = ph / ref - 1.0
    if "discount_rate" not in hist.columns and "discount" in hist.columns:
        hist["discount_rate"] = pd.to_numeric(hist.get("discount"), errors="coerce")

    numeric = ["price_rel_to_recent_median_28", "discount_rate", "stock"]
    for c in numeric:
        if c not in hist.columns or c not in scenario_frame.columns:
            continue
        h = pd.to_numeric(hist[c], errors="coerce").dropna()
        s = pd.to_numeric(scenario_frame[c], errors="coerce").dropna()
        if h.empty or s.empty:
            continue
        lo, hi = float(h.quantile(0.01)), float(h.quantile(0.99))
        if ((s < lo) | (s > hi)).any():
            flags.append(f"ood_numeric:{c}")

    cat_cols = [
        c for c in ["region", "channel", "segment"] + [x for x in feature_spec.get("factor_categorical_features", []) if str(x).startswith("user_factor_cat__")]
        if c in hist.columns and c in scenario_frame.columns
    ]
    for c in cat_cols:
        known = set(hist[c].astype(str).dropna().unique())
        incoming = set(scenario_frame[c].astype(str).dropna().unique())
        if any(v not in known for v in incoming):
            flags.append(f"ood_category:{c}")
    return sorted(set(flags))


def run_factor_rolling_backtest(factor_train_df: pd.DataFrame, feature_spec: Dict[str, Any], n_windows: int = 3, window_days: int = 28, min_train_days: int = 60) -> Dict[str, Any]:
    df = factor_train_df.copy().sort_values("date")
    if df.empty:
        return {"trained": False, "n_valid_windows": 0}
    max_date = pd.Timestamp(df["date"].max())
    starts = sorted([max_date - pd.Timedelta(days=(window_days * (i + 1) - 1)) for i in range(int(n_windows))])
    rows = []
    for ws in starts:
        we = ws + pd.Timedelta(days=window_days)
        tr = df[df["date"] < ws].copy()
        te = df[(df["date"] >= ws) & (df["date"] < we)].copy()
        if tr["date"].nunique() < int(min_train_days) or te.empty:
            continue
        model = train_factor_model(tr, feature_spec, small_mode=len(tr) < 200, training_profile="backtest")
        pred = predict_factor_effect(te, model, feature_spec)
        y = pd.to_numeric(te["factor_target"], errors="coerce").fillna(0.0)
        e = y - pred["factor_log_effect"]
        rows.append({
            "rmse": float(np.sqrt(np.mean(np.square(e)))),
            "mae": float(np.mean(np.abs(e))),
            "price_sign_stability": _price_sign_stability(te, model, feature_spec),
            "factor_multiplier_median": float(pred["factor_multiplier"].median()),
            "factor_multiplier_p95": float(pred["factor_multiplier"].quantile(0.95)),
            "factor_ood_share": float(pd.to_numeric(te.get("ood_flag", pd.Series(0, index=te.index)), errors="coerce").fillna(0).mean()),
        })
    n_valid = len(rows)
    if n_valid == 0:
        return {"trained": False, "n_valid_windows": 0}
    m = pd.DataFrame(rows)
    return {
        "trained": True,
        "n_valid_windows": n_valid,
        "median_factor_target_rmse": float(m["rmse"].median()),
        "median_factor_target_mae": float(m["mae"].median()),
        "price_sign_stability": float(m["price_sign_stability"].mean()),
        "factor_multiplier_median": float(m["factor_multiplier_median"].median()),
        "factor_multiplier_p95": float(m["factor_multiplier_p95"].median()),
        "factor_ood_share": float(m["factor_ood_share"].mean()),
    }


def run_factor_backtest(df: pd.DataFrame, trained_factor: Dict[str, Any], feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    # kept for compatibility
    pred = predict_factor_effect(df, trained_factor, feature_spec)
    y = pd.to_numeric(df.get("factor_target", 0.0), errors="coerce").fillna(0.0)
    e = y - pred["factor_log_effect"]
    return {
        "trained": True,
        "factor_target_rmse": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
        "factor_target_mae": float(np.mean(np.abs(e))) if len(e) else np.nan,
        "price_sign_stability": _price_sign_stability(df, trained_factor, feature_spec),
        "factor_multiplier_median": float(pred["factor_multiplier"].median()) if len(pred) else np.nan,
        "factor_multiplier_p95": float(pred["factor_multiplier"].quantile(0.95)) if len(pred) else np.nan,
        "factor_ood_share": float(df.get("ood_flag", pd.Series([0] * len(df))).mean()) if len(df) else 0.0,
    }


def compute_factor_contributions(target_history: pd.DataFrame, future_dates_df: pd.DataFrame, base_ctx: Dict[str, Any], scenario_ctx: Dict[str, Any], trained_factor: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.DataFrame:
    if trained_factor is None:
        return pd.DataFrame(columns=["factor_name", "contribution_abs", "contribution_pct", "confidence", "note"])
    from pricing_core.scenario_engine import build_future_factor_frame

    base_frame = build_future_factor_frame(target_history, future_dates_df, base_ctx, feature_spec)
    base_mult = float(predict_factor_effect(base_frame, trained_factor, feature_spec)["factor_multiplier"].mean())
    rows = []
    for f in feature_spec.get("controllable_features", []):
        if base_ctx.get(f) == scenario_ctx.get(f):
            continue
        c = dict(base_ctx)
        c[f] = scenario_ctx.get(f)
        frame = build_future_factor_frame(target_history, future_dates_df, c, feature_spec)
        m = float(predict_factor_effect(frame, trained_factor, feature_spec)["factor_multiplier"].mean())
        pct = 0.0 if base_mult <= 1e-9 else (m / base_mult - 1.0)
        rows.append({"factor_name": f, "contribution_abs": pct, "contribution_pct": pct, "confidence": "advisory", "note": "deterministic_delta"})
    return pd.DataFrame(rows, columns=["factor_name", "contribution_abs", "contribution_pct", "confidence", "note"])
