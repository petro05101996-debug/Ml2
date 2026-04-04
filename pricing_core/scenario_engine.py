from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pricing_core.baseline_model import recursive_baseline_forecast
from pricing_core.factor_model import build_factor_ood_flags, predict_factor_effect
from pricing_core.shock_engine import build_default_no_shock_profile, build_shock_profile


def build_base_scenario_context(target_history: pd.DataFrame, factor_feature_spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cols = set(factor_feature_spec.get("controllable_features", []) + factor_feature_spec.get("context_features", []))
    for c in cols:
        if c not in target_history.columns:
            out[c] = 0.0 if c in factor_feature_spec.get("factor_numeric_features", []) else "unknown"
            continue
        s = target_history[c]
        if c in factor_feature_spec.get("factor_categorical_features", []) or s.dtype == object:
            v = s.dropna().astype(str)
            out[c] = str(v.iloc[-1]) if len(v) else "unknown"
        else:
            v = pd.to_numeric(s, errors="coerce").dropna()
            out[c] = float(v.iloc[-1]) if len(v) else 0.0
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
            if k == "promotion":
                scenario[k] = float(bool(v)) if isinstance(v, bool) else float(pd.to_numeric(pd.Series([v]), errors="coerce").fillna(0.0).iloc[0] > 0)
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
        r["price"] = max(float(r.get("price", 0.0)), 1e-6)
        r["discount"] = float(r.get("discount", 0.0))
        r["promotion"] = float(r.get("promotion", 0.0))
        r["stock"] = float(r.get("stock", 0.0))
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
) -> Dict[str, Any]:
    base_ctx = {c: (base_history[c].dropna().astype(str).iloc[-1] if c in ["product_id", "category", "region", "channel", "segment"] and c in base_history else 0.0) for c in baseline_feature_spec.get("baseline_context_features", [])}

    if baseline_override_df is not None:
        baseline_df = baseline_override_df[["date", "baseline_pred"]].copy()
    else:
        if trained_baseline is None:
            raise ValueError("trained_baseline is required when baseline_override_df is not provided")
        baseline_df = recursive_baseline_forecast(trained_baseline, base_history, future_dates_df, base_ctx, baseline_feature_spec)

    feature_spec = factor_feature_spec or {
        "controllable_features": ["price", "discount", "promotion", "stock"],
        "context_features": ["category", "region", "channel", "segment"],
        "factor_numeric_features": ["price", "discount", "promotion", "stock", "price_rel_to_recent_median_28", "discount_rate", "promo_flag"],
        "factor_features": ["price", "discount", "promotion", "stock", "price_rel_to_recent_median_28", "discount_rate", "promo_flag", "price_rel_to_recent_median_28_x_promo_flag", "price_rel_to_recent_median_28_x_is_weekend"],
        "factor_categorical_features": ["category", "region", "channel", "segment"],
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
    out["scenario_demand_raw"] = (out["baseline_pred"] * out["factor_multiplier"] * out["shock_multiplier"]).clip(lower=0.0)
    out["final_demand"] = out["scenario_demand_raw"]
    if np.isfinite(float(scenario_ctx.get("stock", np.nan))):
        stock = max(0.0, float(scenario_ctx.get("stock", 0.0)))
        out["final_demand"] = np.minimum(out["final_demand"], stock)
    out["final_demand"] = out["final_demand"].clip(lower=0.0)
    out["scenario_lower"] = out["final_demand"]
    out["scenario_upper"] = out["final_demand"]
    return {
        "scenario_forecast": out[["date", "baseline_pred", "factor_multiplier", "shock_multiplier", "scenario_demand_raw", "final_demand", "scenario_lower", "scenario_upper"]],
        "base_ctx": base_scn_ctx,
        "scenario_ctx": scenario_ctx,
        "warnings": scenario_ctx.get("_warnings", []),
        "ood_flags": ood_flags,
        "mode": mode,
    }


def compute_scenario_delta(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    b = float(pd.to_numeric(baseline_df.get("baseline_pred", 0.0), errors="coerce").fillna(0.0).sum())
    s = float(pd.to_numeric(scenario_df.get("final_demand", 0.0), errors="coerce").fillna(0.0).sum())
    return pd.DataFrame([{"baseline_total_demand": b, "scenario_total_demand": s, "demand_delta_abs": s - b, "demand_delta_pct": 0.0 if abs(b) < 1e-9 else (s - b) / b}])
