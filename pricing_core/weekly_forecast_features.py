from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data_adapter import USER_FACTOR_CAT_PREFIX, USER_FACTOR_NUM_PREFIX, build_weekly_panel_from_daily

LAG_COLS = [1, 2, 3, 4, 8, 12]


def _safe_mode(series: pd.Series, default: str = "unknown") -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return default
    m = s.mode(dropna=True)
    return str(m.iloc[0]) if not m.empty else default


def _weekly_agg(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date")
    x["week_start"] = x["date"].dt.to_period("W-SUN").dt.start_time
    if "discount" not in x.columns and "discount_rate" in x.columns:
        x["discount"] = pd.to_numeric(x["discount_rate"], errors="coerce")
    if "quantity" not in x.columns:
        x["quantity"] = pd.to_numeric(x.get("sales", 0.0), errors="coerce").fillna(0.0)

    num_user = [c for c in x.columns if str(c).startswith(USER_FACTOR_NUM_PREFIX)]
    cat_user = [c for c in x.columns if str(c).startswith(USER_FACTOR_CAT_PREFIX)]

    g = x.groupby("week_start", dropna=False)
    out = g.agg(
        sales_week=("sales", "sum"),
        quantity_week=("quantity", "sum"),
        revenue_week=("revenue", "sum"),
        discount_week=("discount", "mean"),
        promo_week=("promotion", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).mean())),
        cost_week=("cost", "mean"),
        freight_week=("freight_value", "mean"),
    ).reset_index()

    price_weighted = g.apply(lambda z: np.average(pd.to_numeric(z["price"], errors="coerce").fillna(0.0), weights=np.maximum(pd.to_numeric(z.get("quantity", 1.0), errors="coerce").fillna(1.0), 0.0)) if len(z) else np.nan)
    out["price_week"] = price_weighted.values

    for c in ["series_id", "product_id", "category", "region", "channel", "segment"]:
        if c in x.columns:
            out[c] = g[c].apply(_safe_mode).values

    for c in num_user:
        out[c] = g[c].mean().values
    for c in cat_user:
        out[c] = g[c].apply(_safe_mode).values

    return out.sort_values("week_start").reset_index(drop=True)


def _add_time_and_lag_features(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy().sort_values("week_start").reset_index(drop=True)
    sales = pd.to_numeric(out["sales_week"], errors="coerce").fillna(0.0)
    for lag in LAG_COLS:
        out[f"lag_{lag}"] = sales.shift(lag)

    out["rolling_mean_4"] = sales.shift(1).rolling(4, min_periods=1).mean()
    out["rolling_mean_8"] = sales.shift(1).rolling(8, min_periods=1).mean()
    out["rolling_std_4"] = sales.shift(1).rolling(4, min_periods=1).std().fillna(0.0)
    out["rolling_std_8"] = sales.shift(1).rolling(8, min_periods=1).std().fillna(0.0)
    out["rolling_min_4"] = sales.shift(1).rolling(4, min_periods=1).min()
    out["rolling_max_4"] = sales.shift(1).rolling(4, min_periods=1).max()

    p = pd.to_numeric(out["price_week"], errors="coerce")
    p_med8 = p.shift(1).rolling(8, min_periods=2).median()
    p_mean12 = p.shift(1).rolling(12, min_periods=2).mean()
    out["price_rel_to_rolling_8w_median"] = (p / p_med8.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["price_rel_to_rolling_12w_mean"] = (p / p_mean12.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["discount_depth"] = pd.to_numeric(out["discount_week"], errors="coerce").fillna(0.0)
    out["promo_x_discount"] = pd.to_numeric(out["promo_week"], errors="coerce").fillna(0.0) * out["discount_depth"]
    out["price_x_promo"] = p.fillna(0.0) * pd.to_numeric(out["promo_week"], errors="coerce").fillna(0.0)

    ws = pd.to_datetime(out["week_start"], errors="coerce")
    out["week_of_year"] = ws.dt.isocalendar().week.astype(int)
    out["month"] = ws.dt.month.astype(int)
    out["quarter"] = ws.dt.quarter.astype(int)
    out["is_year_start"] = ws.dt.is_year_start.astype(int)
    out["is_year_end"] = ws.dt.is_year_end.astype(int)
    return out


def build_weekly_train_frame(daily_history: pd.DataFrame) -> pd.DataFrame:
    return _add_time_and_lag_features(_weekly_agg(daily_history))


def build_future_weekly_frame(last_week_start: pd.Timestamp, horizon_weeks: int, context: Dict[str, Any]) -> pd.DataFrame:
    weeks = pd.date_range(pd.Timestamp(last_week_start) + pd.Timedelta(days=7), periods=int(horizon_weeks), freq="7D")
    rows: List[Dict[str, Any]] = []
    for ws in weeks:
        r: Dict[str, Any] = {
            "week_start": pd.Timestamp(ws),
            "price_week": float(pd.to_numeric(pd.Series([context.get("price", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "discount_week": float(pd.to_numeric(pd.Series([context.get("discount", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "promo_week": float(pd.to_numeric(pd.Series([context.get("promotion", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "cost_week": float(pd.to_numeric(pd.Series([context.get("cost", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            "freight_week": float(pd.to_numeric(pd.Series([context.get("freight_value", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
        }
        for k, v in context.items():
            if str(k).startswith(USER_FACTOR_NUM_PREFIX):
                r[k] = float(pd.to_numeric(pd.Series([v]), errors="coerce").fillna(0.0).iloc[0])
            if str(k).startswith(USER_FACTOR_CAT_PREFIX):
                r[k] = str(v)
        rows.append(r)
    return pd.DataFrame(rows)


def build_recursive_weekly_features(history_plus_predictions: pd.DataFrame) -> pd.DataFrame:
    return _add_time_and_lag_features(history_plus_predictions)
