from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from confidence_engine import (
    build_confidence_summary,
    compute_freight_confidence,
    compute_price_confidence,
    compute_promo_confidence,
)
from scenario_effects import (
    apply_stock_constraint_vector,
    combine_standard_effects_vector,
    compute_freight_effect_vector,
    compute_price_effect_vector,
    compute_promo_effect_vector,
)
from scenario_engine import DEFAULTS, _blend_price_elasticity
from shock_engine import compute_shock_multiplier, compute_shock_units, validate_shocks


def _resolve_path_vector(
    dates: pd.Series,
    path_value: Any,
    default_value: float,
) -> np.ndarray:
    n = len(dates)
    if path_value is None:
        return np.repeat(float(default_value), n)
    if isinstance(path_value, dict):
        path_map = {str(k): float(v) for k, v in path_value.items()}
        out = [path_map.get(str(pd.Timestamp(d).date()), default_value) for d in pd.to_datetime(dates)]
        return np.asarray(out, dtype=float)
    if isinstance(path_value, list):
        if len(path_value) and isinstance(path_value[0], dict):
            path_map = {str(pd.to_datetime(x.get("date")).date()): float(x.get("value", default_value)) for x in path_value if x.get("date") is not None}
            out = [path_map.get(str(pd.Timestamp(d).date()), default_value) for d in pd.to_datetime(dates)]
            return np.asarray(out, dtype=float)
        if len(path_value) == n and len(path_value):
            return np.asarray(path_value, dtype=float)
    return np.repeat(float(default_value), n)


def run_enhanced_scenario(
    baseline_daily: pd.DataFrame,
    current_ctx: Dict[str, Any],
    future_dates: pd.DataFrame,
    scenario_overrides: Dict[str, Any],
    pooled_elasticity: float,
    small_mode_info: Dict[str, Any],
    requested_price: float,
    model_price: float,
    baseline_discount: float,
    scenario_discount: float,
    baseline_cost: float,
    scenario_cost: float,
    baseline_freight: float,
    scenario_freight: float,
    shocks: List[Dict[str, Any]],
    financial_price: float | None = None,
) -> Dict[str, Any]:
    daily = baseline_daily.copy().reset_index(drop=True)
    if "date" not in daily.columns and len(future_dates):
        daily["date"] = pd.to_datetime(future_dates["date"])
    daily["date"] = pd.to_datetime(daily.get("date"), errors="coerce")

    baseline_units = pd.to_numeric(daily.get("base_pred_sales", daily.get("actual_sales", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    available_stock = pd.to_numeric(daily.get("stock", pd.Series(np.repeat(np.inf, len(daily)))), errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    if float(scenario_overrides.get("stock_cap", 0.0)) > 0:
        available_stock = np.minimum(available_stock, float(scenario_overrides["stock_cap"]))

    baseline_price_gross = np.repeat(float(current_ctx.get("price", model_price)), len(daily))
    scenario_requested_price_gross = _resolve_path_vector(daily["date"], scenario_overrides.get("requested_price_path"), float(requested_price))
    scenario_model_price_gross = _resolve_path_vector(daily["date"], scenario_overrides.get("model_price_path"), float(model_price))
    scenario_financial_price_gross = _resolve_path_vector(
        daily["date"],
        scenario_overrides.get("price_path"),
        float(model_price if financial_price is None else financial_price),
    )
    baseline_discount_vec = np.repeat(float(np.clip(baseline_discount, 0.0, 0.95)), len(daily))
    scenario_discount_vec = _resolve_path_vector(daily["date"], scenario_overrides.get("discount_path"), float(np.clip(scenario_discount, 0.0, 0.95)))
    scenario_discount_vec = np.clip(scenario_discount_vec, 0.0, 0.95)

    baseline_price_net = np.maximum(0.01, baseline_price_gross * (1.0 - baseline_discount_vec))
    scenario_model_price_net = np.maximum(0.01, scenario_model_price_gross * (1.0 - scenario_discount_vec))
    scenario_price_net = np.maximum(0.01, scenario_financial_price_gross * (1.0 - scenario_discount_vec))

    baseline_promo = np.repeat(float(current_ctx.get("promotion", 0.0)), len(daily))
    scenario_promo = _resolve_path_vector(
        daily["date"],
        scenario_overrides.get("promo_path"),
        float(scenario_overrides.get("promotion", current_ctx.get("promotion", 0.0))),
    )
    baseline_freight_vec = np.repeat(float(baseline_freight), len(daily))
    scenario_freight_vec = _resolve_path_vector(daily["date"], scenario_overrides.get("freight_path"), float(scenario_freight))
    baseline_cost_vec = np.repeat(float(baseline_cost), len(daily))
    scenario_cost_vec = _resolve_path_vector(daily["date"], scenario_overrides.get("cost_path"), float(scenario_cost))

    price_conf = compute_price_confidence(
        int(small_mode_info.get("price_changes", 0)),
        float(small_mode_info.get("price_span", 0.0)),
        float(small_mode_info.get("price_stability", 0.5)),
    )
    beta_final = _blend_price_elasticity(
        float(pooled_elasticity),
        float(DEFAULTS["price_elasticity_prior"]),
        float(price_conf["score"]),
    )

    tail_multiplier = _resolve_path_vector(daily["date"], scenario_overrides.get("extrapolation_tail_multiplier_path"), float(scenario_overrides.get("extrapolation_tail_multiplier", 1.0)))
    tail_multiplier = np.clip(np.nan_to_num(tail_multiplier, nan=1.0, posinf=1.0, neginf=1.0), 0.05, 20.0)
    extrapolation_ratio = _resolve_path_vector(daily["date"], scenario_overrides.get("extrapolation_price_ratio_path"), float(scenario_overrides.get("extrapolation_price_ratio", 1.0)))
    extrapolation_ratio = np.nan_to_num(extrapolation_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    price_effect = compute_price_effect_vector(baseline_price_net, scenario_model_price_net, price_elasticity=beta_final, cap=float(DEFAULTS["price_cap"]))
    price_effect = price_effect * tail_multiplier
    promo_effect = compute_promo_effect_vector(
        promo_flag_baseline=(baseline_promo > 0).astype(float),
        promo_flag_scenario=(scenario_promo > 0).astype(float),
        promo_share_baseline=baseline_promo,
        promo_share_scenario=scenario_promo,
        alpha_flag=float(DEFAULTS["promo_alpha_flag"]),
        alpha_share=float(DEFAULTS["promo_alpha_share"]),
        cap=float(DEFAULTS["promo_cap"]),
    )
    freight_effect = compute_freight_effect_vector(
        freight_ref=baseline_freight_vec,
        freight_scenario=scenario_freight_vec,
        beta_freight=float(DEFAULTS["freight_beta"]),
        cap=float(DEFAULTS["freight_cap"]),
    )
    standard_multiplier = combine_standard_effects_vector({"price": price_effect, "promo": promo_effect, "freight": freight_effect})

    valid_shocks, shock_warnings = validate_shocks(shocks or [])
    shock_multiplier = compute_shock_multiplier(pd.to_datetime(daily["date"]), valid_shocks)
    shock_units = compute_shock_units(pd.to_datetime(daily["date"]), valid_shocks)
    demand_multiplier_path = _resolve_path_vector(daily["date"], scenario_overrides.get("demand_multiplier_path"), 1.0)
    shock_multiplier = shock_multiplier * demand_multiplier_path

    scenario_demand_raw = baseline_units * standard_multiplier * shock_multiplier + shock_units
    scenario_demand_raw = np.clip(np.nan_to_num(scenario_demand_raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    actual_sales = apply_stock_constraint_vector(scenario_demand_raw, available_stock)
    lost_sales = np.clip(scenario_demand_raw - actual_sales, 0.0, None)

    revenue = actual_sales * scenario_price_net
    profit = actual_sales * (scenario_price_net - scenario_cost_vec - scenario_freight_vec)
    baseline_revenue = baseline_units * baseline_price_net
    baseline_profit = baseline_units * (baseline_price_net - baseline_cost_vec - baseline_freight_vec)

    promo_conf = compute_promo_confidence(int(small_mode_info.get("promo_weeks", 0)), float(small_mode_info.get("promo_variability", 0.0)))
    freight_conf = compute_freight_confidence(int(small_mode_info.get("freight_changes", 0)), float(small_mode_info.get("freight_variation", 0.0)))
    conf = build_confidence_summary(price_conf, promo_conf, freight_conf, shocks_present=bool(valid_shocks))

    scenario_profile = pd.DataFrame(
        {
            "date": daily["date"],
            "baseline_units": baseline_units,
            "baseline_price_gross": baseline_price_gross,
            "requested_price_gross": scenario_requested_price_gross,
            "safe_price_gross": scenario_model_price_gross,
            "model_price_gross": scenario_model_price_gross,
            "price_for_model": scenario_model_price_gross,
            "applied_price_gross": scenario_financial_price_gross,
            "scenario_price_gross": scenario_financial_price_gross,
            "baseline_discount": baseline_discount_vec,
            "scenario_discount": scenario_discount_vec,
            "baseline_price_net": baseline_price_net,
            "model_price_net": scenario_model_price_net,
            "applied_price_net": scenario_price_net,
            "scenario_price_net": scenario_price_net,
            "extrapolation_tail_multiplier": tail_multiplier,
            "extrapolation_price_ratio": extrapolation_ratio,
            "baseline_promotion": baseline_promo,
            "scenario_promotion": scenario_promo,
            "baseline_freight_value": baseline_freight_vec,
            "scenario_freight_value": scenario_freight_vec,
            "baseline_cost": baseline_cost_vec,
            "scenario_cost": scenario_cost_vec,
            "available_stock": available_stock,
            "price_effect": price_effect,
            "promo_effect": promo_effect,
            "freight_effect": freight_effect,
            "standard_multiplier": standard_multiplier,
            "shock_multiplier": shock_multiplier,
            "shock_units": shock_units,
            "scenario_demand_raw": scenario_demand_raw,
            "actual_sales": actual_sales,
            "lost_sales": lost_sales,
            "revenue": revenue,
            "profit": profit,
            "baseline_revenue": baseline_revenue,
            "baseline_profit": baseline_profit,
        }
    )

    warnings: List[str] = list(shock_warnings)
    if float(np.min([price_conf["score"], promo_conf["score"], freight_conf["score"]])) < 0.45:
        warnings.append("Low support detected: conservative interpretation is recommended.")
    if len(valid_shocks):
        warnings.append("Manual shock assumptions are applied.")

    return {
        "scenario_profile": scenario_profile,
        "price_effect_vector": price_effect,
        "promo_effect_vector": promo_effect,
        "freight_effect_vector": freight_effect,
        "standard_multiplier": standard_multiplier,
        "shock_multiplier": shock_multiplier,
        "shock_units": shock_units,
        "final_units": actual_sales,
        "unconstrained_units": scenario_demand_raw,
        "lost_sales": lost_sales,
        "revenue": revenue,
        "profit": profit,
        "baseline_revenue": baseline_revenue,
        "baseline_profit": baseline_profit,
        "confidence": conf,
        "warnings": warnings,
        "effect_breakdown": {
            "price_effect_mean": float(np.mean(price_effect)) if len(price_effect) else 1.0,
            "promo_effect_mean": float(np.mean(promo_effect)) if len(promo_effect) else 1.0,
            "freight_effect_mean": float(np.mean(freight_effect)) if len(freight_effect) else 1.0,
            "shock_multiplier_mean": float(np.mean(shock_multiplier)) if len(shock_multiplier) else 1.0,
            "trajectory_inputs_active": bool(
                any(
                    k in scenario_overrides
                    for k in ["price_path", "discount_path", "promo_path", "freight_path", "cost_path", "demand_multiplier_path"]
                )
            ),
            "stock_constraint_active": bool(np.any(lost_sales > 1e-9)),
            "shocks_applied": bool(len(valid_shocks)),
        },
    }
