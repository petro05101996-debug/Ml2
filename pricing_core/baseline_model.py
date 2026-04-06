from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from pricing_core.baseline_features import build_baseline_one_step_features
from pricing_core.model_utils import HAS_XGBOOST, XGBRegressor, clean_feature_frame

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    HAS_STATSMODELS_ETS = True
except Exception:  # pragma: no cover - optional fallback
    HAS_STATSMODELS_ETS = False


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
        hist = pd.concat([hist, pd.DataFrame([{"date": pd.Timestamp(dt), "sales": max(0.0, pred), **{k: v for k, v in base_ctx.items() if k in ["series_id", "product_id", "category", "region", "channel", "segment"]}}])], ignore_index=True)
    return pd.DataFrame(rows)


def recursive_baseline_forecast_median7(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(base_history.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base = float(s.tail(7).median()) if len(s) else 0.0
    return pd.DataFrame({"date": pd.to_datetime(future_dates_df["date"]), "baseline_pred": max(0.0, base)})


def recursive_baseline_forecast_mean28(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(base_history.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base = float(s.tail(28).mean()) if len(s) else 0.0
    return pd.DataFrame({"date": pd.to_datetime(future_dates_df["date"]), "baseline_pred": max(0.0, base)})


def recursive_baseline_forecast_dow_median8w(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    hist = base_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist.dropna(subset=["date"]).sort_values("date")
    fallback = float(hist["sales"].tail(7).median()) if len(hist) else 0.0
    if len(hist):
        cutoff = pd.Timestamp(hist["date"].max()) - pd.Timedelta(weeks=8)
        hist_8w = hist[hist["date"] >= cutoff].copy()
    else:
        hist_8w = hist

    rows: List[Dict[str, Any]] = []
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        dow = int(pd.Timestamp(dt).dayofweek)
        pool = hist_8w[hist_8w["date"].dt.dayofweek == dow]["sales"] if len(hist_8w) else pd.Series(dtype=float)
        pred = float(pool.median()) if len(pool) else fallback
        rows.append({"date": pd.Timestamp(dt), "baseline_pred": max(0.0, pred)})
    return pd.DataFrame(rows)


def recursive_baseline_forecast_recent_level_dow_profile(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    hist = base_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist.dropna(subset=["date"]).sort_values("date")
    if hist.empty:
        return pd.DataFrame({"date": pd.to_datetime(future_dates_df["date"]), "baseline_pred": 0.0})

    recent_28 = hist.tail(28)["sales"]
    recent_7 = hist.tail(7)["sales"]
    eps = 1e-6
    recent_level = float(recent_28.mean()) if len(recent_28) else float(hist["sales"].mean())
    short_term_correction = float(recent_7.median() / max(float(recent_28.mean()) if len(recent_28) else recent_level, eps))
    base_level = max(0.0, recent_level * short_term_correction)
    dow_profile = build_weekday_profile(hist, lookback_weeks=8, smoothing=0.5)
    dow_weights = (dow_profile * 7.0).reindex(range(7), fill_value=1.0)

    rows: List[Dict[str, Any]] = []
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        dow = int(pd.Timestamp(dt).dayofweek)
        rows.append({"date": pd.Timestamp(dt), "baseline_pred": max(0.0, base_level * float(dow_weights.get(dow, 1.0)))})
    return pd.DataFrame(rows)


def recursive_baseline_forecast_recent_level_dow_trend(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    hist = base_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist.dropna(subset=["date"]).sort_values("date")
    if hist.empty:
        return pd.DataFrame({"date": pd.to_datetime(future_dates_df["date"]), "baseline_pred": 0.0})

    recent_28 = hist.tail(28)["sales"]
    recent_level = float(recent_28.mean()) if len(recent_28) else float(hist["sales"].mean())
    dow_profile = build_weekday_profile(hist, lookback_weeks=8, smoothing=0.5)
    dow_weights = (dow_profile * 7.0).reindex(range(7), fill_value=1.0)

    weekly_hist = aggregate_daily_to_weekly(hist).tail(8).copy()
    slope = 0.0
    if len(weekly_hist) >= 4:
        diffs = pd.to_numeric(weekly_hist["sales"], errors="coerce").fillna(0.0).diff().dropna()
        if len(diffs):
            weekly_delta = float(diffs.median())
            daily_delta = weekly_delta / 7.0
            slope = float(np.clip(daily_delta / max(recent_level, 1.0), -0.05, 0.05))

    start = pd.Timestamp(hist["date"].max()) + pd.Timedelta(days=1)
    rows: List[Dict[str, Any]] = []
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        t = pd.Timestamp(dt)
        step = max(0, int((t - start).days))
        trend_mult = float(np.clip(1.0 + slope * step, 0.65, 1.35))
        dow = int(t.dayofweek)
        pred = max(0.0, recent_level * trend_mult * float(dow_weights.get(dow, 1.0)))
        rows.append({"date": t, "baseline_pred": pred})
    return pd.DataFrame(rows)


def recursive_baseline_forecast_rolling_dow_regression(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    hist = base_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(hist) < 21:
        return recursive_baseline_forecast_recent_level_dow_profile(base_history, future_dates_df)

    hist["dow"] = hist["date"].dt.dayofweek.astype(int)
    hist["week_of_month"] = ((hist["date"].dt.day - 1) // 7 + 1).astype(int)
    hist["month"] = hist["date"].dt.month.astype(int)
    hist["lag7"] = hist["sales"].shift(7)
    hist["recent_mean_7"] = hist["sales"].shift(1).rolling(7, min_periods=1).mean()
    hist["recent_mean_28"] = hist["sales"].shift(1).rolling(28, min_periods=1).mean()
    train = hist.dropna(subset=["lag7", "recent_mean_7", "recent_mean_28"]).copy()
    if len(train) < 14:
        return recursive_baseline_forecast_recent_level_dow_trend(base_history, future_dates_df)

    feats = ["dow", "week_of_month", "month", "recent_mean_7", "recent_mean_28", "lag7"]
    X = pd.get_dummies(train[feats], columns=["dow", "week_of_month", "month"], drop_first=False)
    y = train["sales"].astype(float)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)

    hist_extended = hist[["date", "sales"]].copy()
    rows: List[Dict[str, Any]] = []
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        t = pd.Timestamp(dt)
        prev = hist_extended.copy().sort_values("date")
        recent_mean_7 = float(prev.tail(7)["sales"].mean()) if len(prev) else 0.0
        recent_mean_28 = float(prev.tail(28)["sales"].mean()) if len(prev) else recent_mean_7
        lag_target = t - pd.Timedelta(days=7)
        lag_pool = prev.loc[prev["date"] == lag_target, "sales"]
        lag7 = float(lag_pool.iloc[-1]) if len(lag_pool) else recent_mean_7
        frame = pd.DataFrame(
            {
                "dow": [int(t.dayofweek)],
                "week_of_month": [int((t.day - 1) // 7 + 1)],
                "month": [int(t.month)],
                "recent_mean_7": [recent_mean_7],
                "recent_mean_28": [recent_mean_28],
                "lag7": [lag7],
            }
        )
        xf = pd.get_dummies(frame, columns=["dow", "week_of_month", "month"], drop_first=False)
        xf = xf.reindex(columns=X.columns, fill_value=0.0)
        pred = max(0.0, float(model.predict(xf)[0]))
        drift_floor = max(0.0, recent_mean_28 * 0.5)
        drift_cap = max(drift_floor + 1e-9, recent_mean_28 * 1.8)
        pred = float(np.clip(pred, drift_floor, drift_cap))
        rows.append({"date": t, "baseline_pred": pred})
        hist_extended = pd.concat([hist_extended, pd.DataFrame({"date": [t], "sales": [pred]})], ignore_index=True)
    return pd.DataFrame(rows)


def recursive_baseline_forecast_ets_seasonal7(base_history: pd.DataFrame, future_dates_df: pd.DataFrame) -> pd.DataFrame:
    hist = base_history.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["sales"] = pd.to_numeric(hist.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(hist) < 35 or not HAS_STATSMODELS_ETS:
        return recursive_baseline_forecast_recent_level_dow_trend(base_history, future_dates_df)
    ts = hist.set_index("date")["sales"].asfreq("D").fillna(0.0)
    horizon = int(len(pd.to_datetime(future_dates_df["date"], errors="coerce")))
    if horizon <= 0:
        return pd.DataFrame(columns=["date", "baseline_pred"])
    try:
        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
        fitted = model.fit(optimized=True, use_brute=False)
        pred = np.asarray(fitted.forecast(horizon), dtype=float)
    except Exception:
        return recursive_baseline_forecast_recent_level_dow_profile(base_history, future_dates_df)
    out = pd.DataFrame({"date": pd.to_datetime(future_dates_df["date"], errors="coerce"), "baseline_pred": np.clip(pred, 0.0, np.inf)})
    return out


def week_start(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt - pd.to_timedelta(dt.dt.dayofweek.fillna(0).astype(int), unit="D")


def aggregate_daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["week_start", "sales"])
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["sales"] = pd.to_numeric(x.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x = x.dropna(subset=["date"])
    if x.empty:
        return pd.DataFrame(columns=["week_start", "sales"])
    x["week_start"] = week_start(x["date"])
    out = x.groupby("week_start", as_index=False)["sales"].sum().sort_values("week_start").reset_index(drop=True)
    return out


def build_weekday_profile(base_history: pd.DataFrame, lookback_weeks: int = 8, smoothing: float = 0.5) -> pd.Series:
    if base_history.empty:
        return pd.Series([1.0 / 7.0] * 7, index=range(7), dtype=float)
    x = base_history.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["sales"] = pd.to_numeric(x.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x = x.dropna(subset=["date"]).sort_values("date")
    if x.empty:
        return pd.Series([1.0 / 7.0] * 7, index=range(7), dtype=float)
    cutoff = pd.Timestamp(x["date"].max()) - pd.Timedelta(weeks=max(int(lookback_weeks), 1))
    recent = x[x["date"] > cutoff].copy()
    if recent.empty:
        recent = x.copy()
    by_dow = recent.groupby(recent["date"].dt.dayofweek)["sales"].sum().reindex(range(7), fill_value=0.0).astype(float)
    prof = by_dow + float(smoothing)
    denom = float(prof.sum())
    if denom <= 1e-12:
        return pd.Series([1.0 / 7.0] * 7, index=range(7), dtype=float)
    return (prof / denom).astype(float)


def disaggregate_weekly_to_daily(
    weekly_forecast: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    weekday_profile: pd.Series,
) -> pd.DataFrame:
    fut = future_dates_df.copy()
    fut["date"] = pd.to_datetime(fut["date"], errors="coerce")
    fut = fut.dropna(subset=["date"]).copy()
    if fut.empty:
        return pd.DataFrame(columns=["date", "baseline_pred"])

    wf = weekly_forecast.copy()
    wf["week_start"] = pd.to_datetime(wf["week_start"], errors="coerce")
    wf["baseline_pred_weekly"] = pd.to_numeric(wf.get("baseline_pred_weekly", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    week_map = wf.dropna(subset=["week_start"]).set_index("week_start")["baseline_pred_weekly"].to_dict()

    prof = pd.Series(weekday_profile, dtype=float).reindex(range(7), fill_value=(1.0 / 7.0))
    if float(prof.sum()) <= 1e-12:
        prof = pd.Series([1.0 / 7.0] * 7, index=range(7), dtype=float)
    else:
        prof = prof / float(prof.sum())

    fut["week_start"] = week_start(fut["date"])
    fut["dow"] = fut["date"].dt.dayofweek.astype(int)
    fut["weekly_total"] = fut["week_start"].map(week_map).fillna(0.0)
    fut["dow_share_raw"] = fut["dow"].map(prof).fillna(1.0 / 7.0)
    fut["week_available_share"] = fut.groupby("week_start")["dow_share_raw"].transform("sum").replace(0.0, np.nan)
    fut["dow_share"] = (fut["dow_share_raw"] / fut["week_available_share"]).fillna(0.0)
    fut["baseline_pred"] = (fut["weekly_total"] * fut["dow_share"]).clip(lower=0.0)
    return fut[["date", "baseline_pred"]].sort_values("date").reset_index(drop=True)


def recursive_weekly_baseline_forecast_median4w(
    weekly_history: pd.DataFrame,
    future_week_starts: pd.DataFrame,
) -> pd.DataFrame:
    s = pd.to_numeric(weekly_history.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base = float(s.tail(4).median()) if len(s) else 0.0
    out = future_week_starts.copy()
    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out["baseline_pred_weekly"] = max(0.0, base)
    return out[["week_start", "baseline_pred_weekly"]].sort_values("week_start").reset_index(drop=True)


def recursive_weekly_baseline_forecast_recent4_avg(
    weekly_history: pd.DataFrame,
    future_week_starts: pd.DataFrame,
) -> pd.DataFrame:
    s = pd.to_numeric(weekly_history.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base = float(s.tail(4).mean()) if len(s) else 0.0
    out = future_week_starts.copy()
    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out["baseline_pred_weekly"] = max(0.0, base)
    return out[["week_start", "baseline_pred_weekly"]].sort_values("week_start").reset_index(drop=True)


def recursive_weekly_baseline_forecast_mean8w(
    weekly_history: pd.DataFrame,
    future_week_starts: pd.DataFrame,
) -> pd.DataFrame:
    s = pd.to_numeric(weekly_history.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    base = float(s.tail(8).mean()) if len(s) else 0.0
    out = future_week_starts.copy()
    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out["baseline_pred_weekly"] = max(0.0, base)
    return out[["week_start", "baseline_pred_weekly"]].sort_values("week_start").reset_index(drop=True)


def forecast_weekly_baseline_by_strategy(
    strategy: str,
    weekly_history: pd.DataFrame,
    future_week_starts: pd.DataFrame,
) -> pd.DataFrame:
    s = str(strategy or "weekly_median4w")
    if s == "weekly_median4w":
        return recursive_weekly_baseline_forecast_median4w(weekly_history, future_week_starts)
    if s == "weekly_recent4_avg":
        return recursive_weekly_baseline_forecast_recent4_avg(weekly_history, future_week_starts)
    if s == "weekly_mean8w":
        return recursive_weekly_baseline_forecast_mean8w(weekly_history, future_week_starts)
    raise ValueError(f"Unknown weekly baseline strategy: {s}")


def forecast_baseline_by_strategy(
    strategy: str,
    trained_baseline: Dict[str, Any] | None,
    base_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    base_ctx: Dict[str, Any],
    feature_spec: Dict[str, Any],
) -> pd.DataFrame:
    s = str(strategy or "xgb_recursive")
    if s == "xgb_recursive":
        if trained_baseline is None:
            raise ValueError("trained_baseline is required for xgb_recursive strategy")
        return recursive_baseline_forecast(trained_baseline, base_history, future_dates_df, base_ctx, feature_spec)
    if s == "median7":
        return recursive_baseline_forecast_median7(base_history, future_dates_df)
    if s == "mean28":
        return recursive_baseline_forecast_mean28(base_history, future_dates_df)
    if s == "dow_median8w":
        return recursive_baseline_forecast_dow_median8w(base_history, future_dates_df)
    if s == "recent_level_dow_profile":
        return recursive_baseline_forecast_recent_level_dow_profile(base_history, future_dates_df)
    if s == "recent_level_dow_trend":
        return recursive_baseline_forecast_recent_level_dow_trend(base_history, future_dates_df)
    if s == "rolling_dow_regression":
        return recursive_baseline_forecast_rolling_dow_regression(base_history, future_dates_df)
    if s == "ets_seasonal7":
        return recursive_baseline_forecast_ets_seasonal7(base_history, future_dates_df)
    raise ValueError(f"Unknown baseline strategy: {s}")


def run_baseline_holdout(actual: pd.Series, pred: pd.Series, dates: pd.Series | None = None) -> Dict[str, float]:
    a = pd.to_numeric(actual, errors="coerce").fillna(0.0)
    p = pd.to_numeric(pred, errors="coerce").fillna(0.0)
    denom = float(np.abs(a).sum())
    wape = 100.0 if denom <= 1e-9 else float(np.abs(a - p).sum() / denom * 100.0)
    mae = float(np.abs(a - p).mean()) if len(a) else 0.0
    rmse = float(np.sqrt(np.mean((a - p) ** 2))) if len(a) else 0.0
    asum, psum = float(a.sum()), float(p.sum())
    bias_pct = 0.0 if abs(asum) <= 1e-9 else float((psum - asum) / asum)
    sum_ratio = 0.0 if abs(asum) <= 1e-9 else float(psum / asum)
    pred_std = float(p.std(ddof=0)) if len(p) else 0.0
    actual_std = float(a.std(ddof=0)) if len(a) else 0.0
    std_ratio = float(pred_std / max(actual_std, 1e-9))
    pred_nunique = int(p.nunique(dropna=True))
    actual_nunique = int(a.nunique(dropna=True))
    is_flat_forecast = bool(pred_nunique <= 2 and std_ratio < 0.35 and actual_nunique >= 5)
    weekday_shape_error = 0.0
    if len(a) and dates is not None:
        dts = pd.to_datetime(dates, errors="coerce")
        dow = dts.dt.dayofweek
        valid = dow.notna()
        dow = dow[valid].astype(int)
        a = a[valid]
        p = p[valid]
        a_share = pd.Series(a.values).groupby(dow).sum()
        p_share = pd.Series(p.values).groupby(dow).sum()
        a_share = a_share.reindex(range(7), fill_value=0.0)
        p_share = p_share.reindex(range(7), fill_value=0.0)
        if float(a_share.sum()) > 1e-9 and float(p_share.sum()) > 1e-9:
            a_share = a_share / float(a_share.sum())
            p_share = p_share / float(p_share.sum())
            weekday_shape_error = float(np.abs(a_share - p_share).mean())
    elif len(a):
        weekday_shape_error = np.nan
    return {
        "forecast_wape": wape,
        "mae": mae,
        "rmse": rmse,
        "bias_pct": bias_pct,
        "sum_ratio": sum_ratio,
        "pred_std": pred_std,
        "actual_std": actual_std,
        "std_ratio": std_ratio,
        "pred_nunique": pred_nunique,
        "actual_nunique": actual_nunique,
        "is_flat_forecast": is_flat_forecast,
        "weekday_shape_error": weekday_shape_error,
    }


def _safe_log(x: float) -> float:
    return float(np.log(max(x, 1e-9)))


def _composite_score(row: pd.Series) -> float:
    return float(
        row.get("median_wape", np.inf)
        + 25.0 * abs(float(row.get("median_bias_pct", 0.0)))
        + 15.0 * abs(_safe_log(float(row.get("median_sum_ratio", 1.0))))
        + 12.0 * float(row.get("flat_window_share", 0.0))
        + 10.0 * max(0.0, 0.60 - float(row.get("median_std_ratio", 0.0)))
        + 8.0 * float(row.get("median_weekday_shape_error", 0.0))
    )


def _apply_strategy_guardrails(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["guardrail_reject"] = False
    for c in ["median_wape", "median_bias_pct", "median_std_ratio", "flat_window_share"]:
        if c not in out.columns:
            out[c] = np.nan
    rg = out["strategy"].astype(str) == "rolling_dow_regression"
    guard_fail = (
        out["flat_window_share"].fillna(1.0) > 0.33
    ) | (
        out["median_std_ratio"].fillna(0.0) < 0.55
    ) | (
        out["median_bias_pct"].abs().fillna(np.inf) > 0.08
    )
    out.loc[rg & guard_fail, "guardrail_reject"] = True
    out["guardrail_reject"] = out["guardrail_reject"].astype(bool)
    return out


def run_baseline_rolling_backtest(
    panel_train: pd.DataFrame,
    target_category: str,
    target_sku: str,
    target_series_id: str | None = None,
    n_windows: int = 3,
    window_days: int = 28,
    min_train_days: int = 120,
    strategy: str = "xgb_recursive",
) -> Dict[str, Any]:
    from pricing_core.baseline_features import derive_baseline_feature_spec

    if target_series_id is not None and "series_id" in panel_train.columns:
        target = panel_train[panel_train["series_id"].astype(str) == str(target_series_id)].copy().sort_values("date")
    else:
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
        base_ctx = {k: (train_t[k].dropna().astype(str).iloc[-1] if k in ["series_id", "product_id", "category", "region", "channel", "segment"] and k in train_t else "unknown") for k in ["series_id", "product_id", "category", "region", "channel", "segment"]}
        fut = pd.DataFrame({"date": test_t["date"].values})
        trained = train_baseline_model(panel_w, spec, small_mode=len(panel_w) < 200, training_profile="backtest") if strategy == "xgb_recursive" else None
        fc = forecast_baseline_by_strategy(strategy, trained, train_t, fut, base_ctx, spec)
        merged = test_t[["date", "sales"]].merge(fc, on="date", how="left")
        merged["window_id"] = i
        merged["window_start"] = ws
        merged["window_end"] = we
        diag_rows.append(merged)
        metric_rows.append({"window_id": i, "window_start": ws, "window_end": we, **run_baseline_holdout(merged["sales"], merged["baseline_pred"], dates=merged["date"])})
    rolling_diag = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame(columns=["date", "sales", "baseline_pred", "window_id", "window_start", "window_end"])
    rolling_metrics = pd.DataFrame(metric_rows)
    backend_available = True
    fallback_used = False
    if str(strategy) == "ets_seasonal7":
        backend_available = bool(HAS_STATSMODELS_ETS)
        fallback_used = not backend_available

    summary = {
        "strategy": str(strategy),
        "n_valid_windows": int(len(rolling_metrics)),
        "median_wape": float(rolling_metrics["forecast_wape"].median()) if len(rolling_metrics) else np.nan,
        "median_bias_pct": float(rolling_metrics["bias_pct"].median()) if len(rolling_metrics) else np.nan,
        "median_sum_ratio": float(rolling_metrics["sum_ratio"].median()) if len(rolling_metrics) else np.nan,
        "max_wape": float(rolling_metrics["forecast_wape"].max()) if len(rolling_metrics) else np.nan,
        "median_std_ratio": float(rolling_metrics["std_ratio"].median()) if len(rolling_metrics) else np.nan,
        "flat_window_share": float(pd.to_numeric(rolling_metrics.get("is_flat_forecast", False), errors="coerce").fillna(0.0).mean()) if len(rolling_metrics) else np.nan,
        "median_weekday_shape_error": float(rolling_metrics["weekday_shape_error"].median()) if len(rolling_metrics) else np.nan,
        "backend_available": backend_available,
        "fallback_used": fallback_used,
    }
    return {"rolling_diag": rolling_diag, "rolling_metrics": rolling_metrics, "rolling_summary": summary}


def select_best_baseline_strategy(
    panel_train: pd.DataFrame,
    target_category: str,
    target_sku: str,
    target_series_id: str | None = None,
    n_windows: int = 3,
    window_days: int = 28,
    min_train_days: int = 120,
) -> Dict[str, Any]:
    strategies = ["xgb_recursive", "median7", "mean28", "dow_median8w", "recent_level_dow_profile", "recent_level_dow_trend", "rolling_dow_regression", "ets_seasonal7"]
    metrics_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for s in strategies:
        bt = run_baseline_rolling_backtest(
            panel_train=panel_train,
            target_category=target_category,
            target_sku=target_sku,
            target_series_id=target_series_id,
            n_windows=n_windows,
            window_days=window_days,
            min_train_days=min_train_days,
            strategy=s,
        )
        m = bt.get("rolling_metrics", pd.DataFrame()).copy()
        if not m.empty:
            m["strategy"] = s
            metrics_rows.append(m)
        rs = bt.get("rolling_summary", {}) or {}
        summary_rows.append(
            {
                "strategy": s,
                "n_valid_windows": int(rs.get("n_valid_windows", 0) or 0),
                "median_wape": float(rs.get("median_wape", np.nan)),
                "median_bias_pct": float(rs.get("median_bias_pct", np.nan)),
                "median_sum_ratio": float(rs.get("median_sum_ratio", np.nan)),
                "max_wape": float(rs.get("max_wape", np.nan)),
                "median_std_ratio": float(rs.get("median_std_ratio", np.nan)),
                "flat_window_share": float(rs.get("flat_window_share", np.nan)),
                "median_weekday_shape_error": float(rs.get("median_weekday_shape_error", np.nan)),
                "backend_available": bool(rs.get("backend_available", True)),
                "fallback_used": bool(rs.get("fallback_used", False)),
            }
        )

    strategy_metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    strategy_summary = _apply_strategy_guardrails(pd.DataFrame(summary_rows))
    valid = strategy_summary[strategy_summary["n_valid_windows"] > 0].copy()
    if valid.empty:
        best_strategy = "xgb_recursive"
    else:
        valid = valid[~valid["guardrail_reject"].fillna(False)].copy() if "guardrail_reject" in valid.columns else valid
        if valid.empty:
            valid = strategy_summary[strategy_summary["n_valid_windows"] > 0].copy()
        valid["score"] = valid.apply(_composite_score, axis=1)
        non_flat = valid[valid["flat_window_share"].fillna(1.0) <= 0.5].copy()
        if not non_flat.empty:
            valid = non_flat
        valid = valid.sort_values(["score", "median_wape"], ascending=[True, True])
        if str(valid.iloc[0]["strategy"]) == "xgb_recursive":
            simple = valid[valid["strategy"] != "xgb_recursive"].copy()
            if not simple.empty:
                runner = simple.sort_values(["score", "median_wape"]).iloc[0]
                xgb = valid.iloc[0]
                wape_gain = float(runner["median_wape"] - xgb["median_wape"])
                if not (
                    wape_gain >= 1.5
                    and float(xgb["flat_window_share"]) <= 0.33
                    and float(xgb["median_std_ratio"]) >= 0.55
                    and abs(float(xgb["median_bias_pct"])) <= 0.08
                ):
                    best_strategy = str(runner["strategy"])
                    return {
                        "best_strategy": best_strategy,
                        "strategy_metrics": strategy_metrics,
                        "strategy_summary": strategy_summary,
                    }
        if str(valid.iloc[0]["strategy"]) in {"recent_level_dow_trend", "rolling_dow_regression", "ets_seasonal7"}:
            incumbent = valid[~valid["strategy"].isin(["recent_level_dow_trend", "rolling_dow_regression", "ets_seasonal7"])].copy()
            if not incumbent.empty:
                incumbent = incumbent.sort_values(["score", "median_wape"]).iloc[0]
                challenger = valid.iloc[0]
                meaningful_wape_gain = float(incumbent["median_wape"]) - float(challenger["median_wape"]) >= 1.0
                meaningful_variance_gain = float(challenger["median_std_ratio"]) - float(incumbent["median_std_ratio"]) >= 0.08
                meaningful_flatness_gain = float(incumbent["flat_window_share"]) - float(challenger["flat_window_share"]) >= 0.2
                if not (meaningful_wape_gain or meaningful_variance_gain or meaningful_flatness_gain):
                    best_strategy = str(incumbent["strategy"])
                    return {
                        "best_strategy": best_strategy,
                        "strategy_metrics": strategy_metrics,
                        "strategy_summary": strategy_summary,
                    }
        best_strategy = str(valid.iloc[0]["strategy"])
    return {
        "best_strategy": best_strategy,
        "strategy_metrics": strategy_metrics,
        "strategy_summary": strategy_summary,
    }


def run_baseline_benchmark_suite(
    panel_train: pd.DataFrame,
    target_category: str,
    target_sku: str,
    target_series_id: str | None = None,
    n_windows: int = 3,
    window_days: int = 28,
    min_train_days: int = 120,
) -> pd.DataFrame:
    daily_selected = select_best_baseline_strategy(
        panel_train=panel_train,
        target_category=target_category,
        target_sku=target_sku,
        target_series_id=target_series_id,
        n_windows=n_windows,
        window_days=window_days,
        min_train_days=min_train_days,
    )
    if target_series_id is not None and "series_id" in panel_train.columns:
        target = panel_train[panel_train["series_id"].astype(str) == str(target_series_id)].copy().sort_values("date")
    else:
        target = panel_train[(panel_train["category"].astype(str) == str(target_category)) & (panel_train["product_id"].astype(str) == str(target_sku))].copy().sort_values("date")
    weekly_selected = run_weekly_baseline_rolling_backtest(
        target_daily_history=target,
        n_windows=n_windows,
        window_days=window_days,
        min_train_days=min_train_days,
    )
    daily_summary = daily_selected.get("strategy_summary", pd.DataFrame()).copy()
    weekly_summary = weekly_selected.get("strategy_summary", pd.DataFrame()).copy()
    if not daily_summary.empty:
        daily_summary["granularity"] = "daily"
    if not weekly_summary.empty:
        weekly_summary["granularity"] = "weekly"
        weekly_summary["backend_available"] = True
        weekly_summary["fallback_used"] = False
        weekly_summary["guardrail_reject"] = False
    summary = pd.concat([daily_summary, weekly_summary], ignore_index=True) if (not daily_summary.empty or not weekly_summary.empty) else pd.DataFrame()
    if summary.empty:
        return summary
    summary["composite_score"] = summary.apply(_composite_score, axis=1)
    summary["goal_wape_median_le_25"] = summary["median_wape"] <= 25.0
    summary["goal_wape_max_le_35"] = summary["max_wape"] <= 35.0
    summary["goal_abs_bias_le_7pct"] = summary["median_bias_pct"].abs() <= 0.07
    summary["goal_sum_ratio_in_range"] = summary["median_sum_ratio"].between(0.93, 1.07, inclusive="both")
    summary["goal_std_ratio_ge_055"] = summary["median_std_ratio"] >= 0.55
    summary["acceptance_pass"] = (
        summary["goal_wape_median_le_25"]
        & summary["goal_wape_max_le_35"]
        & summary["goal_abs_bias_le_7pct"]
        & summary["goal_sum_ratio_in_range"]
        & summary["goal_std_ratio_ge_055"]
    )
    return summary.sort_values(["composite_score", "median_wape"], ascending=[True, True]).reset_index(drop=True)


def run_weekly_baseline_rolling_backtest(
    target_daily_history: pd.DataFrame,
    n_windows: int = 3,
    window_days: int = 28,
    min_train_days: int = 120,
) -> Dict[str, Any]:
    target = target_daily_history.copy()
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    target["sales"] = pd.to_numeric(target.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    target = target.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    strategies = ["weekly_median4w", "weekly_recent4_avg", "weekly_mean8w"]
    max_date = pd.Timestamp(target["date"].max()) if len(target) else pd.Timestamp("1970-01-01")
    starts = sorted([max_date - pd.Timedelta(days=(window_days * (i + 1) - 1)) for i in range(int(n_windows))])

    metrics_rows: List[Dict[str, Any]] = []
    diag_by_strategy: Dict[str, List[pd.DataFrame]] = {s: [] for s in strategies}
    oof_by_strategy: Dict[str, pd.DataFrame] = {}

    for s in strategies:
        oof = target[["date", "sales"]].copy()
        oof["baseline_oof"] = np.nan
        oof_by_strategy[s] = oof

    for i, ws in enumerate(starts, start=1):
        we = ws + pd.Timedelta(days=window_days)
        train_t = target[target["date"] < ws].copy()
        test_t = target[(target["date"] >= ws) & (target["date"] < we)].copy()
        if train_t["date"].nunique() < int(min_train_days) or test_t.empty:
            continue

        weekly_hist = aggregate_daily_to_weekly(train_t)
        weekday_prof = build_weekday_profile(train_t)
        fut_weeks = pd.DataFrame({"week_start": sorted(week_start(test_t["date"]).dropna().unique())})

        for s in strategies:
            weekly_fc = forecast_weekly_baseline_by_strategy(s, weekly_hist, fut_weeks)
            daily_fc = disaggregate_weekly_to_daily(weekly_fc, test_t[["date"]], weekday_prof)
            merged = test_t[["date", "sales"]].merge(daily_fc, on="date", how="left")
            merged["window_id"] = i
            merged["window_start"] = ws
            merged["window_end"] = we
            merged["strategy"] = s
            diag_by_strategy[s].append(merged)

            metric = run_baseline_holdout(merged["sales"], merged["baseline_pred"], dates=merged["date"])
            metrics_rows.append({"strategy": s, "window_id": i, "window_start": ws, "window_end": we, **metric})

            pred_map = dict(zip(pd.to_datetime(daily_fc["date"]), pd.to_numeric(daily_fc["baseline_pred"], errors="coerce")))
            mask = oof_by_strategy[s]["date"].isin(test_t["date"]) & oof_by_strategy[s]["baseline_oof"].isna()
            oof_by_strategy[s].loc[mask, "baseline_oof"] = pd.to_datetime(oof_by_strategy[s].loc[mask, "date"]).map(pred_map)

    strategy_metrics = pd.DataFrame(metrics_rows)
    if strategy_metrics.empty:
        strategy_metrics = pd.DataFrame(columns=["strategy", "window_id", "window_start", "window_end", "forecast_wape", "mae", "rmse", "bias_pct", "sum_ratio"])
    summary_rows: List[Dict[str, Any]] = []
    for s in strategies:
        m = strategy_metrics[strategy_metrics["strategy"] == s].copy()
        summary_rows.append(
            {
                "strategy": s,
                "n_valid_windows": int(len(m)),
                "median_wape": float(m["forecast_wape"].median()) if len(m) else np.nan,
                "median_bias_pct": float(m["bias_pct"].median()) if len(m) else np.nan,
                "median_sum_ratio": float(m["sum_ratio"].median()) if len(m) else np.nan,
                "max_wape": float(m["forecast_wape"].max()) if len(m) else np.nan,
                "median_std_ratio": float(m["std_ratio"].median()) if len(m) else np.nan,
                "flat_window_share": float(pd.to_numeric(m.get("is_flat_forecast", False), errors="coerce").fillna(0.0).mean()) if len(m) else np.nan,
                "median_weekday_shape_error": float(m["weekday_shape_error"].median()) if len(m) else np.nan,
            }
        )
    strategy_summary = pd.DataFrame(summary_rows)
    valid = strategy_summary[strategy_summary["n_valid_windows"] > 0].copy()
    if valid.empty:
        best_strategy = "weekly_median4w"
    else:
        valid["score"] = valid.apply(_composite_score, axis=1)
        valid = valid.sort_values(["score", "median_wape"], ascending=[True, True])
        best_strategy = str(valid.iloc[0]["strategy"])

    best_metrics = strategy_metrics[strategy_metrics["strategy"] == best_strategy].copy()
    best_diag = pd.concat(diag_by_strategy.get(best_strategy, []), ignore_index=True) if diag_by_strategy.get(best_strategy) else pd.DataFrame(columns=["date", "sales", "baseline_pred", "window_id", "window_start", "window_end", "strategy"])
    best_summary_row = strategy_summary[strategy_summary["strategy"] == best_strategy]
    rolling_summary = {
        "strategy": best_strategy,
        "n_valid_windows": int(best_summary_row["n_valid_windows"].iloc[0]) if len(best_summary_row) else 0,
        "median_wape": float(best_summary_row["median_wape"].iloc[0]) if len(best_summary_row) else np.nan,
        "median_bias_pct": float(best_summary_row["median_bias_pct"].iloc[0]) if len(best_summary_row) else np.nan,
        "median_sum_ratio": float(best_summary_row["median_sum_ratio"].iloc[0]) if len(best_summary_row) else np.nan,
        "max_wape": float(best_summary_row["max_wape"].iloc[0]) if len(best_summary_row) else np.nan,
        "median_std_ratio": float(best_summary_row["median_std_ratio"].iloc[0]) if len(best_summary_row) else np.nan,
        "flat_window_share": float(best_summary_row["flat_window_share"].iloc[0]) if len(best_summary_row) else np.nan,
        "median_weekday_shape_error": float(best_summary_row["median_weekday_shape_error"].iloc[0]) if len(best_summary_row) else np.nan,
    }
    return {
        "best_strategy": best_strategy,
        "strategy_metrics": strategy_metrics,
        "strategy_summary": strategy_summary,
        "rolling_metrics": best_metrics,
        "rolling_diag": best_diag,
        "rolling_summary": rolling_summary,
        "strategy_oof_daily": oof_by_strategy,
        "oof_daily": oof_by_strategy.get(best_strategy, pd.DataFrame(columns=["date", "sales", "baseline_oof"])),
    }


def select_best_baseline_plan(
    panel_train: pd.DataFrame,
    target_category: str,
    target_sku: str,
    target_series_id: str | None = None,
    n_windows: int = 3,
    window_days: int = 28,
    min_train_days: int = 120,
) -> Dict[str, Any]:
    daily_selection = select_best_baseline_strategy(
        panel_train=panel_train,
        target_category=target_category,
        target_sku=target_sku,
        target_series_id=target_series_id,
        n_windows=n_windows,
        window_days=window_days,
        min_train_days=min_train_days,
    )
    if target_series_id is not None and "series_id" in panel_train.columns:
        target = panel_train[panel_train["series_id"].astype(str) == str(target_series_id)].copy().sort_values("date")
    else:
        target = panel_train[(panel_train["category"].astype(str) == str(target_category)) & (panel_train["product_id"].astype(str) == str(target_sku))].copy().sort_values("date")
    weekly_selection = run_weekly_baseline_rolling_backtest(
        target_daily_history=target,
        n_windows=n_windows,
        window_days=window_days,
        min_train_days=min_train_days,
    )

    daily_summary = daily_selection.get("strategy_summary", pd.DataFrame())
    daily_best = str(daily_selection.get("best_strategy", "xgb_recursive"))
    daily_row = daily_summary[daily_summary["strategy"] == daily_best] if not daily_summary.empty else pd.DataFrame()
    daily_wape = float(daily_row["median_wape"].iloc[0]) if len(daily_row) else np.inf
    daily_score = float(daily_row.apply(_composite_score, axis=1).iloc[0]) if len(daily_row) else np.inf

    weekly_summary = weekly_selection.get("strategy_summary", pd.DataFrame())
    weekly_best = str(weekly_selection.get("best_strategy", "weekly_median4w"))
    weekly_row = weekly_summary[weekly_summary["strategy"] == weekly_best] if not weekly_summary.empty else pd.DataFrame()
    weekly_wape = float(weekly_row["median_wape"].iloc[0]) if len(weekly_row) else np.inf
    weekly_score = float(weekly_row.apply(_composite_score, axis=1).iloc[0]) if len(weekly_row) else np.inf
    weekly_bad = False
    if len(weekly_row):
        weekly_bad = bool(
            float(weekly_row["flat_window_share"].iloc[0]) > 0.5
            or float(weekly_row["median_std_ratio"].iloc[0]) < 0.4
            or float(weekly_row["median_weekday_shape_error"].iloc[0]) > 0.2
        )
    weekly_beats_with_margin = np.isfinite(weekly_score) and np.isfinite(daily_score) and (weekly_score + 1e-9 < daily_score) and not weekly_bad
    granularity = "weekly" if weekly_beats_with_margin else "daily"
    selected_strategy = weekly_best if granularity == "weekly" else daily_best
    selected_median_wape = weekly_wape if granularity == "weekly" else daily_wape
    reason = (
        f"weekly selected: composite {weekly_score:.2f} vs daily {daily_score:.2f}"
        if granularity == "weekly"
        else f"daily selected: composite {daily_score:.2f} vs weekly {weekly_score:.2f}"
    )

    return {
        "granularity": granularity,
        "daily_selection": daily_selection,
        "weekly_selection": weekly_selection,
        "best_daily_strategy": daily_best,
        "best_weekly_strategy": weekly_best,
        "selected_strategy": selected_strategy,
        "selected_median_wape": selected_median_wape,
        "selection_margin_pp_wape": float(daily_wape - weekly_wape) if np.isfinite(daily_wape) and np.isfinite(weekly_wape) else np.nan,
        "selector_reason": reason,
    }


def build_baseline_oof_predictions(
    panel_train: pd.DataFrame,
    target_category: str,
    target_sku: str,
    target_series_id: str | None = None,
    min_train_days: int = 84,
    step_days: int = 7,
    horizon_days: int = 7,
    strategy: str = "xgb_recursive",
) -> pd.DataFrame:
    from pricing_core.baseline_features import derive_baseline_feature_spec

    if target_series_id is not None and "series_id" in panel_train.columns:
        target = panel_train[panel_train["series_id"].astype(str) == str(target_series_id)].copy().sort_values("date")
    else:
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
        base_ctx_w = {
            k: (target_w[k].dropna().astype(str).iloc[-1] if k in target_w.columns and k in ["series_id", "product_id", "category", "region", "channel", "segment"] else "unknown")
            for k in ["series_id", "product_id", "category", "region", "channel", "segment"]
        }
        trained_w = train_baseline_model(panel_w, spec_w, small_mode=len(panel_w) < 200, training_profile="backtest") if strategy == "xgb_recursive" else None
        fc_w = forecast_baseline_by_strategy(strategy, trained_w, target_w, test_dates_df, base_ctx_w, spec_w)
        pred_map = dict(zip(pd.to_datetime(fc_w["date"]), pd.to_numeric(fc_w["baseline_pred"], errors="coerce")))

        mask = out["date"].isin(test_slice["date"]) & out["baseline_oof"].isna()
        out.loc[mask, "baseline_oof"] = pd.to_datetime(out.loc[mask, "date"]).map(pred_map)
        start_idx += int(step_days)

    return out.sort_values("date").reset_index(drop=True)


def build_weekly_baseline_oof_predictions(
    target_daily_history: pd.DataFrame,
    strategy: str = "weekly_median4w",
    min_train_days: int = 84,
    step_days: int = 7,
    horizon_days: int = 7,
) -> pd.DataFrame:
    target = target_daily_history.copy()
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    target["sales"] = pd.to_numeric(target.get("sales", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    target = target.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out = target[["date", "sales"]].copy()
    out["baseline_oof"] = np.nan
    if target.empty:
        return out

    n = len(target)
    start_idx = int(max(min_train_days, 1))
    while start_idx < n:
        test_slice = target.iloc[start_idx:start_idx + int(horizon_days)].copy()
        train_t = target.iloc[:start_idx].copy()
        if test_slice.empty:
            break
        if train_t["date"].nunique() < int(min_train_days):
            start_idx += int(step_days)
            continue
        weekly_hist = aggregate_daily_to_weekly(train_t)
        weekday_prof = build_weekday_profile(train_t)
        fut_weeks = pd.DataFrame({"week_start": sorted(week_start(test_slice["date"]).dropna().unique())})
        weekly_fc = forecast_weekly_baseline_by_strategy(strategy, weekly_hist, fut_weeks)
        daily_fc = disaggregate_weekly_to_daily(weekly_fc, test_slice[["date"]], weekday_prof)
        pred_map = dict(zip(pd.to_datetime(daily_fc["date"]), pd.to_numeric(daily_fc["baseline_pred"], errors="coerce")))
        mask = out["date"].isin(test_slice["date"]) & out["baseline_oof"].isna()
        out.loc[mask, "baseline_oof"] = pd.to_datetime(out.loc[mask, "date"]).map(pred_map)
        start_idx += int(step_days)
    return out.sort_values("date").reset_index(drop=True)
