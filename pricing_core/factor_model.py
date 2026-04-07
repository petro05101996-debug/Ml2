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
    m = {"price_rel_to_recent_median_28": -1, "discount_rate": 1, "promo_flag": 1}
    return tuple(int(m.get(f, 0)) for f in feature_names)


def train_factor_model(train_df: pd.DataFrame, feature_spec: Dict[str, Any], small_mode: bool = False, training_profile: str = "final") -> Dict[str, Any]:
    feats = feature_spec.get("factor_features", [])
    cat = feature_spec.get("factor_categorical_features", [])
    X = clean_feature_frame(train_df, feats, feature_spec.get("factor_numeric_features", []), cat)
    y = pd.to_numeric(train_df.get("factor_target", 0.0), errors="coerce").fillna(0.0)
    if "factor_weight" in train_df.columns:
        sample_weight = pd.to_numeric(train_df["factor_weight"], errors="coerce").fillna(1.0).clip(lower=0.05)
    else:
        sample_weight = pd.Series(1.0, index=train_df.index, dtype=float)
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
        model.fit(X[feats], y, sample_weight=sample_weight)
        backend, is_fallback = "xgboost", False
    else:
        X_num = pd.get_dummies(X[feats], columns=[c for c in cat if c in X.columns], dummy_na=True)
        model = HistGradientBoostingRegressor(max_depth=p["max_depth"], max_iter=p["n_estimators"], learning_rate=0.03, random_state=42)
        model.fit(X_num, y, sample_weight=sample_weight)
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
    mult = np.clip(np.exp(np.asarray(log_eff, dtype=float)), 0.70, 1.35)
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


def build_factor_ood_flags(target_history: pd.DataFrame, factor_future_df: pd.DataFrame, feature_spec: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    hist = target_history.copy()

    if "price_rel_to_recent_median_28" not in hist.columns and "price" in hist.columns:
        ph = pd.to_numeric(hist.get("price"), errors="coerce").dropna()
        recent_ref = float(ph.tail(28).median()) if len(ph) else np.nan
        if not np.isfinite(recent_ref) or recent_ref <= 0:
            recent_ref = float(ph.median()) if len(ph) else 1.0
        if not np.isfinite(recent_ref) or recent_ref <= 0:
            recent_ref = 1.0
        hist["price_rel_to_recent_median_28"] = pd.to_numeric(hist.get("price"), errors="coerce") / recent_ref - 1.0

    if "discount_rate" not in hist.columns and "discount" in hist.columns:
        hist["discount_rate"] = pd.to_numeric(hist.get("discount"), errors="coerce")

    for c in ["price_rel_to_recent_median_28", "discount_rate"]:
        if c not in hist.columns or c not in factor_future_df.columns:
            continue
        h = pd.to_numeric(hist[c], errors="coerce").dropna()
        s = pd.to_numeric(factor_future_df[c], errors="coerce").dropna()
        if h.empty or s.empty:
            continue
        lo, hi = float(h.quantile(0.01)), float(h.quantile(0.99))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if ((s < lo) | (s > hi)).any():
            flags.append(f"ood_numeric:{c}")

    default_cats = ["series_id", "product_id", "category", "region", "channel", "segment"]
    user_cats = [c for c in feature_spec.get("factor_categorical_features", []) if str(c).startswith("user_factor_cat__")]
    for c in default_cats + user_cats:
        if c not in hist.columns or c not in factor_future_df.columns:
            continue
        known = set(hist[c].dropna().astype(str).unique())
        incoming = set(factor_future_df[c].dropna().astype(str).unique())
        if known and any(v not in known for v in incoming):
            flags.append(f"ood_category:{c}")

    return sorted(set(flags))



def _estimate_window_ood_share(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    if test_df.empty or train_df.empty:
        return 0.0
    flags = pd.Series(False, index=test_df.index)

    for col in ["price_rel_to_recent_median_28", "discount_rate"]:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        h = pd.to_numeric(train_df[col], errors="coerce").dropna()
        s = pd.to_numeric(test_df[col], errors="coerce")
        if h.empty or s.dropna().empty:
            continue
        lo, hi = float(h.quantile(0.01)), float(h.quantile(0.99))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        flags = flags | ((s < lo) | (s > hi)).fillna(False)

    for col in ["series_id", "product_id", "category", "region", "channel", "segment"] + [c for c in test_df.columns if str(c).startswith("user_factor_cat__")]:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        known = set(train_df[col].dropna().astype(str).unique())
        if not known:
            continue
        incoming = test_df[col].astype(str)
        flags = flags | (~incoming.isin(known))

    return float(flags.mean()) if len(flags) else 0.0

def run_factor_rolling_backtest(factor_train_df: pd.DataFrame, feature_spec: Dict[str, Any], n_windows: int = 3, window_days: int = 28, min_train_days: int = 60) -> Dict[str, Any]:
    df = factor_train_df.copy().sort_values("date")
    if df.empty:
        return {"trained": False, "n_valid_windows": 0, "factor_ood_share": 0.0}
    max_date = pd.Timestamp(df["date"].max())
    starts = sorted([max_date - pd.Timedelta(days=(window_days * (i + 1) - 1)) for i in range(int(n_windows))])

    rows = []
    for ws in starts:
        we = ws + pd.Timedelta(days=window_days)
        tr = df[df["date"] < ws].copy()
        te = df[(df["date"] >= ws) & (df["date"] < we)].copy()
        if tr["date"].nunique() < int(min_train_days) or te.empty:
            continue

        trained = train_factor_model(tr, feature_spec, small_mode=len(tr) < 200, training_profile="backtest")
        pred = predict_factor_effect(te, trained, feature_spec)
        y = pd.to_numeric(te["factor_target"], errors="coerce").fillna(0.0)
        e = y - pred["factor_log_effect"]
        rows.append(
            {
                "factor_target_rmse": float(np.sqrt(np.mean(np.square(e)))),
                "factor_target_mae": float(np.mean(np.abs(e))),
                "price_sign_stability": _price_sign_stability(te, trained, feature_spec),
                "factor_multiplier_median": float(pred["factor_multiplier"].median()),
                "factor_multiplier_p95": float(pred["factor_multiplier"].quantile(0.95)),
                "factor_ood_share": _estimate_window_ood_share(tr, te),
            }
        )

    if not rows:
        return {"trained": False, "n_valid_windows": 0, "factor_ood_share": 0.0}

    m = pd.DataFrame(rows)
    return {
        "trained": True,
        "n_valid_windows": int(len(m)),
        "median_factor_target_rmse": float(m["factor_target_rmse"].median()),
        "median_factor_target_mae": float(m["factor_target_mae"].median()),
        "price_sign_stability": float(m["price_sign_stability"].mean()),
        "factor_multiplier_median": float(m["factor_multiplier_median"].median()),
        "factor_multiplier_p95": float(m["factor_multiplier_p95"].median()),
        "factor_ood_share": float(m["factor_ood_share"].mean()),
    }


def run_factor_backtest(df: pd.DataFrame, trained_factor: Dict[str, Any], feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    # kept for compatibility and local sanity checks
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


def _compute_delta_contributions(
    target_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    from_ctx: Dict[str, Any],
    to_ctx: Dict[str, Any],
    trained_factor: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> pd.DataFrame:
    if trained_factor is None:
        return pd.DataFrame(columns=["factor_name", "from_value", "to_value", "multiplier_delta", "contribution_pct", "confidence", "note"])
    from pricing_core.scenario_engine import build_future_factor_frame

    base_frame = build_future_factor_frame(target_history, future_dates_df, from_ctx, feature_spec)
    base_mult = float(predict_factor_effect(base_frame, trained_factor, feature_spec)["factor_multiplier"].mean())
    rows = []
    for f in feature_spec.get("controllable_features", []):
        if from_ctx.get(f) == to_ctx.get(f):
            continue
        c = dict(from_ctx)
        c[f] = to_ctx.get(f)
        frame = build_future_factor_frame(target_history, future_dates_df, c, feature_spec)
        m = float(predict_factor_effect(frame, trained_factor, feature_spec)["factor_multiplier"].mean())
        pct = 0.0 if base_mult <= 1e-9 else (m / base_mult - 1.0)
        rows.append(
            {
                "factor_name": f,
                "from_value": from_ctx.get(f),
                "to_value": to_ctx.get(f),
                "multiplier_delta": pct,
                "contribution_pct": pct,
                "confidence": "advisory",
                "note": "deterministic_one_factor_delta_not_shap",
            }
        )
    return pd.DataFrame(rows, columns=["factor_name", "from_value", "to_value", "multiplier_delta", "contribution_pct", "confidence", "note"])


def compute_current_state_contributions(
    target_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    neutral_ctx: Dict[str, Any],
    current_ctx: Dict[str, Any],
    trained_factor: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> pd.DataFrame:
    return _compute_delta_contributions(target_history, future_dates_df, neutral_ctx, current_ctx, trained_factor, feature_spec)


def compute_scenario_delta_contributions(
    target_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    current_ctx: Dict[str, Any],
    scenario_ctx: Dict[str, Any],
    trained_factor: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> pd.DataFrame:
    return _compute_delta_contributions(target_history, future_dates_df, current_ctx, scenario_ctx, trained_factor, feature_spec)


def compute_factor_contributions(target_history: pd.DataFrame, future_dates_df: pd.DataFrame, base_ctx: Dict[str, Any], scenario_ctx: Dict[str, Any], trained_factor: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.DataFrame:
    return _compute_delta_contributions(target_history, future_dates_df, base_ctx, scenario_ctx, trained_factor, feature_spec)
