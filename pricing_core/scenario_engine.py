from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pricing_core.baseline_model import recursive_baseline_forecast
from pricing_core.factor_model import build_factor_ood_flags, predict_factor_effect, predict_weekly_factor_effect
from pricing_core.shock_engine import build_default_no_shock_profile, build_shock_profile

SCENARIO_NUMERIC_KEYS = ["price", "discount", "promotion", "cost", "freight_value"]
SCENARIO_CATEGORICAL_KEYS: List[str] = []
CTX_KEYS = ["series_id", "product_id", "category", "region", "channel", "segment"]


def _last_valid_numeric(df: pd.DataFrame, c: str, default: float = 0.0) -> float:
    if c not in df.columns:
        return default
    s = pd.to_numeric(df[c], errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def _last_valid_cat(df: pd.DataFrame, c: str, default: str = "unknown") -> str:
    if c not in df.columns:
        return default
    s = df[c].dropna().astype(str)
    return str(s.iloc[-1]) if len(s) else default


def _recent_mode_or_last(df: pd.DataFrame, c: str, lookback: int = 28, default: str = "unknown") -> str:
    if c not in df.columns:
        return default
    s = df[c].dropna()
    if s.empty:
        return default
    tail = s.tail(lookback)
    mode = tail.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(s.astype(str).iloc[-1])


def build_current_state_context(target_history: pd.DataFrame, factor_feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if target_history.empty:
        return out
    user_num = [c for c in target_history.columns if c.startswith("user_factor_num__")]
    user_cat = [c for c in target_history.columns if c.startswith("user_factor_cat__")]

    for c in SCENARIO_NUMERIC_KEYS + user_num:
        out[c] = _last_valid_numeric(target_history, c, 0.0)
    for c in CTX_KEYS + SCENARIO_CATEGORICAL_KEYS + user_cat:
        out[c] = _last_valid_cat(target_history, c, "unknown")

    out["stock_total_horizon"] = np.nan
    out["use_stock_cap"] = False
    out["daily_stock_cap"] = np.nan
    return out


def build_neutral_context(target_history: pd.DataFrame, factor_feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    out = build_current_state_context(target_history, factor_feature_spec)
    if target_history.empty:
        return out
    price_hist = pd.to_numeric(target_history.get("price", np.nan), errors="coerce").dropna()
    out["price"] = float(price_hist.tail(28).median()) if len(price_hist.tail(28)) else (float(price_hist.median()) if len(price_hist) else 0.0)
    discount_hist = pd.to_numeric(target_history.get("discount", np.nan), errors="coerce").dropna()
    promo_hist = pd.to_numeric(target_history.get("promotion", np.nan), errors="coerce").dropna()
    out["discount"] = float(discount_hist.tail(28).median()) if len(discount_hist.tail(28)) else _last_valid_numeric(target_history, "discount", 0.0)
    out["promotion"] = float(promo_hist.tail(28).median()) if len(promo_hist.tail(28)) else _last_valid_numeric(target_history, "promotion", 0.0)

    controllable = set((factor_feature_spec or {}).get("controllable_features", []))
    for c in [col for col in target_history.columns if col.startswith("user_factor_num__")]:
        s = pd.to_numeric(target_history.get(c, np.nan), errors="coerce").dropna()
        if c in controllable:
            out[c] = float(s.tail(28).median()) if len(s.tail(28)) else (_last_valid_numeric(target_history, c, 0.0))
        else:
            out[c] = _last_valid_numeric(target_history, c, 0.0)
    for c in [col for col in target_history.columns if col.startswith("user_factor_cat__")]:
        out[c] = _recent_mode_or_last(target_history, c, lookback=28, default="unknown")
    return out


def build_base_scenario_context(target_history: pd.DataFrame, factor_feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    # backward compatibility alias
    return build_current_state_context(target_history, factor_feature_spec)


def apply_user_overrides(base_ctx: Dict[str, Any], scenario_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    scenario = dict(base_ctx)
    warnings = []
    for k, v in (scenario_overrides or {}).items():
        if k not in scenario and not (str(k).startswith("user_factor_num__") or str(k).startswith("user_factor_cat__")):
            warnings.append(f"unknown_override_ignored:{k}")
            continue
        if k == "use_stock_cap":
            scenario[k] = bool(v)
            continue
        if isinstance(scenario.get(k), str):
            scenario[k] = str(v)
        else:
            scenario[k] = float(pd.to_numeric(pd.Series([v]), errors="coerce").fillna(scenario.get(k, 0.0)).iloc[0])
    scenario["_warnings"] = warnings
    return scenario


def build_future_factor_frame(target_history: pd.DataFrame, future_dates_df: pd.DataFrame, scenario_ctx: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    hist = target_history.copy()
    hist["date"] = pd.to_datetime(hist.get("date"), errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")

    price_hist = pd.to_numeric(hist.get("price", np.nan), errors="coerce").dropna()
    recent_price_ref = float(price_hist.tail(28).median()) if len(price_hist.tail(28)) else (float(price_hist.median()) if len(price_hist) else 1.0)
    recent_price_ref = recent_price_ref if np.isfinite(recent_price_ref) and recent_price_ref > 0 else 1.0

    sales_hist = pd.to_numeric(hist.get("sales", np.nan), errors="coerce").dropna()
    recent_sales_7 = float(sales_hist.tail(7).mean()) if len(sales_hist.tail(7)) else float(sales_hist.mean()) if len(sales_hist) else 0.0
    recent_sales_28 = float(sales_hist.tail(28).mean()) if len(sales_hist.tail(28)) else float(sales_hist.mean()) if len(sales_hist) else 0.0
    sales_ratio = recent_sales_7 / max(recent_sales_28, 1e-9)

    recent_8w = hist[hist["date"] > (hist["date"].max() - pd.Timedelta(days=56))] if len(hist) else hist
    dow_share = pd.Series([1 / 7.0] * 7, index=range(7), dtype=float)
    if len(recent_8w):
        by_dow = pd.to_numeric(recent_8w.get("sales", 0.0), errors="coerce").fillna(0.0)
        by_dow = by_dow.groupby(recent_8w["date"].dt.dayofweek).sum().reindex(range(7), fill_value=0.0).astype(float)
        if float(by_dow.sum()) > 1e-9:
            dow_share = by_dow / float(by_dow.sum())

    promo_dates = pd.to_datetime(hist.loc[pd.to_numeric(hist.get("promotion", 0.0), errors="coerce").fillna(0.0) > 0, "date"], errors="coerce") if len(hist) else pd.Series([], dtype="datetime64[ns]")
    last_promo_date = pd.Timestamp(promo_dates.max()) if len(promo_dates) else None

    price_56 = price_hist.tail(56)
    price_rank = float((price_56 <= float(scenario_ctx.get("price", recent_price_ref))).mean()) if len(price_56) else 0.5

    last_vals = hist.tail(1).to_dict("records")[0] if len(hist) else {}
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        t = pd.Timestamp(dt)
        r: Dict[str, Any] = {"date": t}
        for c in feature_spec.get("context_features", []):
            r[c] = scenario_ctx.get(c, last_vals.get(c, "unknown"))
        for c in feature_spec.get("factor_numeric_features", []):
            r[c] = scenario_ctx.get(c, last_vals.get(c, 0.0))
        for c in feature_spec.get("factor_categorical_features", []):
            r[c] = scenario_ctx.get(c, last_vals.get(c, "unknown"))
        r["price"] = max(float(r.get("price", 0.0)), 1e-6)
        r["discount"] = float(r.get("discount", 0.0))
        r["promotion"] = float(r.get("promotion", 0.0))
        r["discount_rate"] = r["discount"]
        r["promo_flag"] = 1.0 if r["promotion"] > 0 else 0.0
        r["is_weekend"] = float(t.dayofweek >= 5)
        r["dow"] = int(t.dayofweek)
        r["week_of_month"] = int((t.day - 1) // 7 + 1)
        r["month"] = int(t.month)
        r["is_month_start"] = float(t.is_month_start)
        r["is_month_end"] = float(t.is_month_end)
        r["recent_sales_level_7"] = recent_sales_7
        r["recent_sales_level_28"] = recent_sales_28
        r["sales_level_ratio_7_to_28"] = sales_ratio
        r["weekday_profile_share"] = float(dow_share.get(int(t.dayofweek), 1.0 / 7.0))
        r["days_since_last_promo"] = float((t - last_promo_date).days) if last_promo_date is not None else 999.0
        r["price_rank_vs_last_8_weeks"] = price_rank
        r["price_rel_to_recent_median_28"] = r["price"] / recent_price_ref - 1.0
        r["price_rel_to_recent_median_28_x_promo_flag"] = r["price_rel_to_recent_median_28"] * r["promo_flag"]
        r["price_rel_to_recent_median_28_x_is_weekend"] = r["price_rel_to_recent_median_28"] * r["is_weekend"]
        rows.append(r)
    out = pd.DataFrame(rows)
    for c in feature_spec.get("factor_features", []):
        if c not in out.columns:
            out[c] = last_vals.get(c, 0.0 if c in feature_spec.get("factor_numeric_features", []) else "unknown")
    return out


def apply_total_stock_cap(raw_series: pd.Series, total_stock: float) -> pd.DataFrame:
    remaining = max(0.0, float(total_stock))
    actual_list, lost_list, rem_list = [], [], []
    for raw in pd.to_numeric(raw_series, errors="coerce").fillna(0.0).clip(lower=0.0):
        actual = min(float(raw), remaining)
        lost = float(raw) - actual
        remaining -= actual
        actual_list.append(actual)
        lost_list.append(max(0.0, lost))
        rem_list.append(max(0.0, remaining))
    return pd.DataFrame({"actual_sales": actual_list, "lost_sales": lost_list, "remaining_stock": rem_list})


def _fallback_multiplier(base_history: pd.DataFrame, ctx: Dict[str, Any], n_rows: int) -> np.ndarray:
    hist = base_history.copy()
    hist["price"] = pd.to_numeric(hist.get("price", np.nan), errors="coerce")
    hist["sales"] = pd.to_numeric(hist.get("sales", np.nan), errors="coerce")
    if "stockout_flag" in hist.columns:
        hist = hist[pd.to_numeric(hist["stockout_flag"], errors="coerce").fillna(0.0) <= 0].copy()
    hist = hist.dropna(subset=["price", "sales"])
    price_ref = float(hist["price"].tail(28).median()) if len(hist) else 1.0
    price_ref = price_ref if np.isfinite(price_ref) and price_ref > 0 else 1.0
    price_new = float(pd.to_numeric(pd.Series([ctx.get("price", price_ref)]), errors="coerce").fillna(price_ref).iloc[0])
    discount_new = float(pd.to_numeric(pd.Series([ctx.get("discount", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    promotion_new = float(pd.to_numeric(pd.Series([ctx.get("promotion", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    elasticity = -1.1
    if len(hist) >= 20 and hist["price"].nunique() > 2:
        beta = np.polyfit(np.log(hist["price"].clip(lower=1e-6)), np.log(hist["sales"].clip(lower=1e-6)), 1)[0]
        if np.isfinite(beta):
            elasticity = float(np.clip(beta, -3.0, -0.05))
    price_mult = float(np.clip((max(price_new, 1e-6) / price_ref) ** elasticity, 0.25, 4.0))
    promo_mult = 1.1 if promotion_new > 0 else 1.0
    discount_mult = float(np.clip(1.0 + max(discount_new, 0.0) * 0.25, 1.0, 1.3))
    return np.full(n_rows, float(np.clip(price_mult * promo_mult * discount_mult, 0.2, 5.0)))


def predict_context_multiplier(target_history: pd.DataFrame, future_dates_df: pd.DataFrame, ctx: Dict[str, Any], feature_spec: Dict[str, Any], trained_factor: Dict[str, Any] | None) -> Dict[str, Any]:
    factor_future_df = build_future_factor_frame(target_history, future_dates_df, ctx, feature_spec)
    ood_flags = build_factor_ood_flags(target_history, factor_future_df, feature_spec)
    if trained_factor is not None:
        factor_pred_df = predict_factor_effect(factor_future_df, trained_factor, feature_spec)
        mode = "baseline_plus_scenario"
        source = "ml_uplift"
        factor_multiplier_series = pd.to_numeric(factor_pred_df["factor_multiplier"], errors="coerce").fillna(1.0)
    else:
        factor_pred_df = pd.DataFrame({"factor_multiplier": _fallback_multiplier(target_history, ctx, len(factor_future_df))}, index=factor_future_df.index)
        mode = "fallback_elasticity"
        source = "bounded_rules"
        factor_multiplier_series = pd.to_numeric(factor_pred_df["factor_multiplier"], errors="coerce").fillna(1.0)
    return {
        "factor_future_df": factor_future_df,
        "factor_pred_df": factor_pred_df,
        "factor_multiplier_series": factor_multiplier_series,
        "ood_flags": ood_flags,
        "mode": mode,
        "scenario_effect_source": source,
    }


def _bounded_effect(series: pd.Series, confidence_level: str, ood_count: int) -> pd.Series:
    bounds = {
        "high": (0.65, 1.40),
        "medium": (0.75, 1.25),
        "low": (0.85, 1.15),
    }
    lo, hi = bounds.get(str(confidence_level).lower(), bounds["medium"])
    if ood_count > 0:
        lo = max(lo, 0.9)
        hi = min(hi, 1.1)
    return pd.to_numeric(series, errors="coerce").fillna(1.0).clip(lower=lo, upper=hi)


def _contexts_equal_on_controls(a: Dict[str, Any], b: Dict[str, Any], controls: List[str]) -> bool:
    for c in controls:
        va, vb = a.get(c), b.get(c)
        if isinstance(va, str) or isinstance(vb, str):
            if str(va) != str(vb):
                return False
        else:
            if abs(float(pd.to_numeric(pd.Series([va]), errors="coerce").fillna(0.0).iloc[0]) - float(pd.to_numeric(pd.Series([vb]), errors="coerce").fillna(0.0).iloc[0])) > 1e-9:
                return False
    return True


def run_scenario_forecast(
    trained_baseline: Dict[str, Any] | None,
    trained_factor: Dict[str, Any] | None,
    base_history: pd.DataFrame,
    future_dates_df: pd.DataFrame,
    baseline_feature_spec: Dict[str, Any],
    factor_feature_spec: Dict[str, Any] | None,
    scenario_overrides: Dict[str, Any] | None = None,
    shocks: List[Dict[str, Any]] | None = None,
    baseline_override_df: pd.DataFrame | None = None,
    demand_multiplier: float = 1.0,
    factor_backtest_summary: Dict[str, Any] | None = None,
    confidence_level: str = "medium",
) -> Dict[str, Any]:
    base_ctx = {}
    for c in baseline_feature_spec.get("baseline_context_features", []):
        base_ctx[c] = _last_valid_cat(base_history, c, "unknown")

    if baseline_override_df is not None:
        neutral_baseline_df = baseline_override_df[["date", "baseline_pred"]].copy()
    else:
        if trained_baseline is None:
            raise ValueError("trained_baseline is required when baseline_override_df is not provided")
        neutral_baseline_df = recursive_baseline_forecast(trained_baseline, base_history, future_dates_df, base_ctx, baseline_feature_spec)

    feature_spec = factor_feature_spec or {
        "controllable_features": ["price", "discount", "promotion"],
        "context_features": CTX_KEYS,
        "factor_numeric_features": ["price", "discount", "promotion", "price_rel_to_recent_median_28", "discount_rate", "promo_flag"],
        "factor_features": ["price", "discount", "promotion", "price_rel_to_recent_median_28", "discount_rate", "promo_flag", "price_rel_to_recent_median_28_x_promo_flag", "price_rel_to_recent_median_28_x_is_weekend"],
        "factor_categorical_features": CTX_KEYS,
    }

    neutral_ctx = build_neutral_context(base_history, feature_spec)
    current_ctx = build_current_state_context(base_history, feature_spec)
    scenario_ctx = apply_user_overrides(current_ctx, scenario_overrides)

    current_mult = predict_context_multiplier(base_history, future_dates_df, current_ctx, feature_spec, trained_factor if factor_feature_spec is not None else None)
    scenario_mult = predict_context_multiplier(base_history, future_dates_df, scenario_ctx, feature_spec, trained_factor if factor_feature_spec is not None else None)

    shock_df = build_shock_profile(shocks, future_dates_df) if shocks else build_default_no_shock_profile(future_dates_df)
    out = neutral_baseline_df.merge(shock_df, on="date", how="left")
    out["current_multiplier"] = _bounded_effect(
        current_mult["factor_multiplier_series"],
        confidence_level=confidence_level,
        ood_count=len(current_mult["ood_flags"]),
    )
    out["scenario_multiplier"] = _bounded_effect(
        scenario_mult["factor_multiplier_series"],
        confidence_level=confidence_level,
        ood_count=len(scenario_mult["ood_flags"]),
    )
    out["shock_multiplier"] = pd.to_numeric(out["shock_multiplier"], errors="coerce").fillna(1.0).clip(0.2, 5.0)
    out["neutral_baseline_pred"] = pd.to_numeric(out["baseline_pred"], errors="coerce").fillna(0.0)
    out["as_is_demand_raw"] = (out["neutral_baseline_pred"] * out["current_multiplier"] * out["shock_multiplier"] * float(demand_multiplier)).clip(lower=0.0)
    out["scenario_demand_raw"] = (out["neutral_baseline_pred"] * out["scenario_multiplier"] * out["shock_multiplier"] * float(demand_multiplier)).clip(lower=0.0)

    def _apply_cap(raw: pd.Series, ctx: Dict[str, Any], prefix: str) -> pd.DataFrame:
        use_stock_cap = bool(ctx.get("use_stock_cap", False))
        total_stock = float(pd.to_numeric(pd.Series([ctx.get("stock_total_horizon", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
        if use_stock_cap and np.isfinite(total_stock) and total_stock >= 0:
            cap_df = apply_total_stock_cap(raw, total_stock)
        else:
            cap_df = pd.DataFrame({"actual_sales": raw.values, "lost_sales": np.zeros(len(raw)), "remaining_stock": [np.nan] * len(raw)})
        cap_df = cap_df.rename(columns={"actual_sales": f"{prefix}_actual_sales", "lost_sales": f"{prefix}_lost_sales", "remaining_stock": f"{prefix}_remaining_stock"})
        return cap_df

    out = pd.concat([out.reset_index(drop=True), _apply_cap(out["as_is_demand_raw"], current_ctx, "as_is")], axis=1)
    out = pd.concat([out.reset_index(drop=True), _apply_cap(out["scenario_demand_raw"], scenario_ctx, "scenario")], axis=1)

    warnings = list(scenario_ctx.get("_warnings", []))
    controls = list(feature_spec.get("controllable_features", []))
    if _contexts_equal_on_controls(current_ctx, scenario_ctx, controls):
        as_is_total = float(out["as_is_actual_sales"].sum())
        scn_total = float(out["scenario_actual_sales"].sum())
        delta_pct = 0.0 if abs(as_is_total) < 1e-9 else abs(scn_total - as_is_total) / abs(as_is_total)
        if delta_pct > 0.02:
            warnings.append("scenario_equals_current_but_delta_nonzero")

    p95 = float((factor_backtest_summary or {}).get("factor_multiplier_p95", np.nan))
    scn_mult_mean = float(out["scenario_multiplier"].mean()) if len(out) else 1.0
    if np.isfinite(p95) and p95 > 0 and scn_mult_mean > p95 * 1.03:
        warnings.append("scenario_outside_factor_backtest_range")

    neutral_baseline_forecast = out[["date", "baseline_pred"]].copy()
    neutral_baseline_forecast["baseline_lower"] = np.nan
    neutral_baseline_forecast["baseline_upper"] = np.nan

    as_is_forecast = out[["date", "baseline_pred", "current_multiplier", "shock_multiplier", "as_is_demand_raw", "as_is_actual_sales", "as_is_lost_sales", "as_is_remaining_stock"]].copy()
    as_is_forecast = as_is_forecast.rename(columns={"current_multiplier": "factor_multiplier", "as_is_actual_sales": "actual_sales", "as_is_lost_sales": "lost_sales", "as_is_remaining_stock": "remaining_stock", "as_is_demand_raw": "demand_raw"})
    as_is_forecast["final_demand"] = as_is_forecast["actual_sales"]
    as_is_forecast["scenario_demand_raw"] = as_is_forecast["demand_raw"]  # backward compatibility
    as_is_forecast["baseline_component"] = as_is_forecast["baseline_pred"]
    as_is_forecast["factor_effect"] = as_is_forecast["factor_multiplier"]
    as_is_forecast["shock_effect"] = as_is_forecast["shock_multiplier"]
    as_is_forecast["final_forecast"] = as_is_forecast["actual_sales"]

    scenario_forecast = out[["date", "baseline_pred", "scenario_multiplier", "shock_multiplier", "scenario_demand_raw", "scenario_actual_sales", "scenario_lost_sales", "scenario_remaining_stock"]].copy()
    scenario_forecast = scenario_forecast.rename(columns={"scenario_multiplier": "factor_multiplier", "scenario_actual_sales": "actual_sales", "scenario_lost_sales": "lost_sales", "scenario_remaining_stock": "remaining_stock"})
    scenario_forecast["demand_raw"] = scenario_forecast["scenario_demand_raw"]
    scenario_forecast["final_demand"] = scenario_forecast["actual_sales"]
    scenario_forecast["baseline_component"] = scenario_forecast["baseline_pred"]
    scenario_forecast["factor_effect"] = scenario_forecast["factor_multiplier"]
    scenario_forecast["shock_effect"] = scenario_forecast["shock_multiplier"]
    scenario_forecast["final_forecast"] = scenario_forecast["actual_sales"]
    scenario_forecast["scenario_lower"] = np.nan
    scenario_forecast["scenario_upper"] = np.nan

    mode = current_mult["mode"]
    scenario_mode = scenario_mult["mode"]
    factor_source = scenario_mult["scenario_effect_source"]
    source = f"{factor_source}+shock" if shocks else factor_source
    if "scenario_outside_factor_backtest_range" in warnings:
        scenario_mode = "advisory_only"

    return {
        "neutral_baseline_forecast": neutral_baseline_forecast,
        "as_is_forecast": as_is_forecast,
        "scenario_forecast": scenario_forecast,
        "neutral_ctx": neutral_ctx,
        "current_ctx": current_ctx,
        "scenario_ctx": scenario_ctx,
        "current_multiplier_series": out["current_multiplier"].copy(),
        "scenario_multiplier_series": out["scenario_multiplier"].copy(),
        "current_ood_flags": current_mult["ood_flags"],
        "scenario_ood_flags": scenario_mult["ood_flags"],
        "warnings": warnings,
        "mode": mode,
        "scenario_mode": scenario_mode,
        "scenario_effect_source": source,
        "factor_effect_source": factor_source,
        "shock_applied": bool(shocks),
        # backward-compatible keys
        "base_ctx": current_ctx,
        "ood_flags": sorted(set(current_mult["ood_flags"] + scenario_mult["ood_flags"])),
    }


def compute_scenario_delta(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    b = float(pd.to_numeric(baseline_df.get("baseline_pred", 0.0), errors="coerce").fillna(0.0).sum())
    s = float(pd.to_numeric(scenario_df.get("actual_sales", scenario_df.get("final_demand", 0.0)), errors="coerce").fillna(0.0).sum())
    return pd.DataFrame([{"baseline_total_demand": b, "scenario_total_demand": s, "demand_delta_abs": s - b, "demand_delta_pct": 0.0 if abs(b) < 1e-9 else (s - b) / b}])


def run_weekly_scenario_forecast(
    weekly_baseline_forecast: pd.DataFrame,
    trained_weekly_factor: Dict[str, Any] | None,
    current_ctx: Dict[str, Any],
    scenario_ctx: Dict[str, Any],
    shocks: List[Dict[str, Any]] | None = None,
    confidence_level: str = "medium",
) -> Dict[str, Any]:
    wk = weekly_baseline_forecast.copy()
    wk["week_start"] = pd.to_datetime(wk["week_start"], errors="coerce")
    wk = wk.sort_values("week_start").reset_index(drop=True)
    if "baseline_pred_weekly" not in wk.columns and "baseline_pred" in wk.columns:
        wk["baseline_pred_weekly"] = pd.to_numeric(wk["baseline_pred"], errors="coerce").fillna(0.0)

    def _weekly_ctx_features(ctx: Dict[str, Any]) -> pd.DataFrame:
        out = wk[["week_start"]].copy()
        out["avg_price_week"] = float(pd.to_numeric(pd.Series([ctx.get("price", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        out["avg_discount_week"] = float(pd.to_numeric(pd.Series([ctx.get("discount", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        out["promo_share_week"] = 1.0 if float(pd.to_numeric(pd.Series([ctx.get("promotion", 0.0)]), errors="coerce").fillna(0.0).iloc[0]) > 0 else 0.0
        out["avg_freight_week"] = float(pd.to_numeric(pd.Series([ctx.get("freight_value", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        out["avg_cost_week"] = float(pd.to_numeric(pd.Series([ctx.get("cost", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        return out

    cur_feat = _weekly_ctx_features(current_ctx)
    scn_feat = _weekly_ctx_features(scenario_ctx)
    cur_mult = predict_weekly_factor_effect(cur_feat, trained_weekly_factor)
    scn_mult = predict_weekly_factor_effect(scn_feat, trained_weekly_factor)
    cur_mult = _bounded_effect(cur_mult, confidence_level, ood_count=0)
    scn_mult = _bounded_effect(scn_mult, confidence_level, ood_count=0)

    shock_weekly = pd.Series(1.0, index=wk.index, dtype=float)
    if shocks:
        for sh in shocks:
            ws = pd.Timestamp(sh.get("start_date"))
            we = pd.Timestamp(sh.get("end_date"))
            intensity = float(pd.to_numeric(pd.Series([sh.get("intensity", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            direction = str(sh.get("direction", "positive")).lower()
            mult = 1.0 + abs(intensity) if direction == "positive" else max(0.0, 1.0 - abs(intensity))
            mask = (wk["week_start"] >= ws) & (wk["week_start"] <= we)
            shock_weekly.loc[mask] = shock_weekly.loc[mask] * mult
    shock_weekly = shock_weekly.clip(lower=0.5, upper=1.5)

    out = wk.copy()
    out["baseline_component"] = pd.to_numeric(out["baseline_pred_weekly"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["factor_effect_as_is"] = cur_mult.values
    out["factor_effect_scenario"] = scn_mult.values
    out["shock_effect"] = shock_weekly.values
    out["final_as_is"] = (out["baseline_component"] * out["factor_effect_as_is"] * out["shock_effect"]).clip(lower=0.0)
    out["final_scenario"] = (out["baseline_component"] * out["factor_effect_scenario"] * out["shock_effect"]).clip(lower=0.0)
    return {
        "weekly_neutral_baseline_forecast": out[["week_start", "baseline_component"]].rename(columns={"baseline_component": "baseline_pred_weekly"}),
        "weekly_as_is_forecast": out[["week_start", "baseline_component", "factor_effect_as_is", "shock_effect", "final_as_is"]],
        "weekly_scenario_forecast": out[["week_start", "baseline_component", "factor_effect_scenario", "shock_effect", "final_scenario"]],
        "weekly_factor_effect_as_is": out[["week_start", "factor_effect_as_is"]],
        "weekly_factor_effect_scenario": out[["week_start", "factor_effect_scenario"]],
        "factor_effect_source": "ml_uplift" if trained_weekly_factor is not None else "bounded_rules",
        "shock_applied": bool(shocks),
        "ood_flags": [],
        "warnings": [],
    }
