from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pricing_core.baseline_model import recursive_baseline_forecast
from pricing_core.factor_model import build_factor_ood_flags, predict_factor_effect
from pricing_core.shock_engine import build_default_no_shock_profile, build_shock_profile

SCENARIO_NUMERIC_KEYS = ["price", "discount", "promotion", "cost", "freight_value"]
SCENARIO_CATEGORICAL_KEYS: List[str] = []


def build_base_scenario_context(target_history: pd.DataFrame, factor_feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if target_history.empty:
        return out
    last = target_history.tail(1).iloc[0]

    user_num = [c for c in target_history.columns if c.startswith("user_factor_num__")]
    user_cat = [c for c in target_history.columns if c.startswith("user_factor_cat__")]

    for c in SCENARIO_NUMERIC_KEYS + user_num:
        if c in target_history.columns:
            out[c] = float(pd.to_numeric(target_history[c], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(target_history[c], errors="coerce").dropna().size else 0.0
        else:
            out[c] = 0.0

    for c in ["product_id", "category"] + SCENARIO_CATEGORICAL_KEYS + user_cat:
        if c in target_history.columns:
            s = target_history[c].dropna().astype(str)
            out[c] = str(s.iloc[-1]) if len(s) else "unknown"
        else:
            out[c] = "unknown"

    out["stock_total_horizon"] = float(pd.to_numeric(pd.Series([last.get("stock", np.nan)]), errors="coerce").fillna(0.0).iloc[0])
    out["daily_stock_cap"] = np.nan
    return out


def apply_user_overrides(base_ctx: Dict[str, Any], scenario_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    scenario = dict(base_ctx)
    warnings = []
    for k, v in (scenario_overrides or {}).items():
        if k not in scenario:
            warnings.append(f"unknown_override_ignored:{k}")
            continue
        if isinstance(scenario.get(k), str):
            scenario[k] = str(v)
        else:
            scenario[k] = float(pd.to_numeric(pd.Series([v]), errors="coerce").fillna(scenario[k]).iloc[0])
    scenario["_warnings"] = warnings
    return scenario


def build_future_factor_frame(target_history: pd.DataFrame, future_dates_df: pd.DataFrame, scenario_ctx: Dict[str, Any], feature_spec: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    price_hist = pd.to_numeric(target_history.get("price", np.nan), errors="coerce").dropna()
    recent_price_ref = float(price_hist.tail(28).median()) if len(price_hist.tail(28)) else np.nan
    if not np.isfinite(recent_price_ref) or recent_price_ref <= 0:
        recent_price_ref = float(price_hist.median()) if len(price_hist) else 1.0
    if not np.isfinite(recent_price_ref) or recent_price_ref <= 0:
        recent_price_ref = 1.0

    last_vals = target_history.tail(1).to_dict("records")[0] if len(target_history) else {}
    for dt in pd.to_datetime(future_dates_df["date"], errors="coerce"):
        r: Dict[str, Any] = {"date": pd.Timestamp(dt)}
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
        r["is_weekend"] = float(pd.Timestamp(dt).dayofweek >= 5)
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
) -> Dict[str, Any]:
    base_ctx = {c: (base_history[c].dropna().astype(str).iloc[-1] if c in ["product_id", "category"] and c in base_history else 0.0) for c in baseline_feature_spec.get("baseline_context_features", [])}

    if baseline_override_df is not None:
        baseline_df = baseline_override_df[["date", "baseline_pred"]].copy()
    else:
        if trained_baseline is None:
            raise ValueError("trained_baseline is required when baseline_override_df is not provided")
        baseline_df = recursive_baseline_forecast(trained_baseline, base_history, future_dates_df, base_ctx, baseline_feature_spec)

    feature_spec = factor_feature_spec or {
        "controllable_features": ["price", "discount", "promotion"],
        "context_features": ["product_id", "category"],
        "factor_numeric_features": ["price", "discount", "promotion", "price_rel_to_recent_median_28", "discount_rate", "promo_flag"],
        "factor_features": ["price", "discount", "promotion", "price_rel_to_recent_median_28", "discount_rate", "promo_flag", "price_rel_to_recent_median_28_x_promo_flag", "price_rel_to_recent_median_28_x_is_weekend"],
        "factor_categorical_features": ["product_id", "category"],
    }
    base_scn_ctx = build_base_scenario_context(base_history, feature_spec)
    scenario_ctx = apply_user_overrides(base_scn_ctx, scenario_overrides)
    factor_future = build_future_factor_frame(base_history, future_dates_df, scenario_ctx, feature_spec)
    ood_flags = build_factor_ood_flags(base_history, factor_future, feature_spec)

    if trained_factor is not None and factor_feature_spec is not None:
        factor_pred = predict_factor_effect(factor_future, trained_factor, factor_feature_spec)
        factor_mult = factor_pred["factor_multiplier"].values
        mode = "baseline_plus_scenario"
    else:
        factor_mult = np.ones(len(baseline_df))
        mode = "baseline_only"

    shock_df = build_shock_profile(shocks, future_dates_df) if shocks else build_default_no_shock_profile(future_dates_df)
    out = baseline_df.merge(shock_df, on="date", how="left")
    out["factor_multiplier"] = factor_mult
    out["shock_multiplier"] = pd.to_numeric(out["shock_multiplier"], errors="coerce").fillna(1.0).clip(0.2, 5.0)
    out["scenario_demand_raw"] = (out["baseline_pred"] * out["factor_multiplier"] * out["shock_multiplier"] * float(demand_multiplier)).clip(lower=0.0)

    total_stock = float(pd.to_numeric(pd.Series([scenario_ctx.get("stock_total_horizon", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
    if np.isfinite(total_stock) and total_stock > 0:
        cap_df = apply_total_stock_cap(out["scenario_demand_raw"], total_stock)
        out = pd.concat([out.reset_index(drop=True), cap_df], axis=1)
    else:
        out["actual_sales"] = out["scenario_demand_raw"]
        out["lost_sales"] = 0.0
        out["remaining_stock"] = np.nan

    out["final_demand"] = out["actual_sales"]
    out["scenario_lower"] = np.nan
    out["scenario_upper"] = np.nan
    return {
        "scenario_forecast": out[["date", "baseline_pred", "factor_multiplier", "shock_multiplier", "scenario_demand_raw", "actual_sales", "lost_sales", "remaining_stock", "final_demand", "scenario_lower", "scenario_upper"]],
        "base_ctx": base_scn_ctx,
        "scenario_ctx": scenario_ctx,
        "warnings": scenario_ctx.get("_warnings", []),
        "ood_flags": ood_flags,
        "mode": mode,
    }


def compute_scenario_delta(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    b = float(pd.to_numeric(baseline_df.get("baseline_pred", 0.0), errors="coerce").fillna(0.0).sum())
    s = float(pd.to_numeric(scenario_df.get("actual_sales", scenario_df.get("final_demand", 0.0)), errors="coerce").fillna(0.0).sum())
    return pd.DataFrame([{"baseline_total_demand": b, "scenario_total_demand": s, "demand_delta_abs": s - b, "demand_delta_pct": 0.0 if abs(b) < 1e-9 else (s - b) / b}])
