from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from confidence_engine import (
    build_confidence_summary,
    compute_freight_confidence,
    compute_price_confidence,
    compute_promo_confidence,
)
from scenario_effects import (
    apply_stock_constraint,
    combine_standard_effects,
    compute_freight_effect,
    compute_price_effect,
    compute_promo_effect,
)
from shock_engine import compute_shock_multiplier, compute_shock_units, validate_shocks


DEFAULTS = {
    "price_elasticity": -1.10,
    "price_elasticity_prior": -1.10,
    "price_cap": 0.35,
    "promo_alpha_flag": 0.08,
    "promo_alpha_share": 0.20,
    "promo_cap": 0.40,
    "freight_beta": -0.03,
    "freight_cap": 0.20,
}


def _blend_price_elasticity(local_beta: float, prior_beta: float, support_weight: float) -> float:
    w = float(np.clip(support_weight, 0.0, 1.0))
    return float(w * local_beta + (1.0 - w) * prior_beta)


def run_scenario(
    baseline_output: pd.DataFrame,
    scenario_inputs: Dict[str, Any],
    shocks: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _finite_array(values: Any, default: float = 0.0) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)

    def _finite_scalar(value: Any, default: float = 0.0) -> float:
        parsed = float(value)
        return float(parsed if np.isfinite(parsed) else default)

    frame = baseline_output.copy()
    metadata = metadata or {}

    baseline_units = pd.to_numeric(frame.get("baseline_units", frame.get("base_pred_sales", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"]) if "date" in frame.columns else pd.Series(pd.date_range("2000-01-01", periods=len(frame), freq="D"))

    price_conf = compute_price_confidence(
        int(metadata.get("price_changes", 0)),
        float(metadata.get("price_span", 0.0)),
        float(metadata.get("price_stability", 0.5)),
    )
    reference_price = float(
        scenario_inputs.get(
            "demand_price_baseline",
            scenario_inputs.get("baseline_price_ref", scenario_inputs.get("base_price", baseline_units.mean())),
        )
    )
    scenario_price = float(
        scenario_inputs.get(
            "demand_price_scenario",
            scenario_inputs.get("scenario_price", scenario_inputs.get("gross_price_scenario", reference_price)),
        )
    )
    beta_local = float(scenario_inputs.get("price_elasticity", DEFAULTS["price_elasticity"]))
    beta_prior = float(scenario_inputs.get("price_elasticity_prior", DEFAULTS["price_elasticity_prior"]))
    beta_final = _blend_price_elasticity(beta_local, beta_prior, float(price_conf["score"]))
    price_effect = compute_price_effect(
        reference_price=reference_price,
        scenario_price=scenario_price,
        price_elasticity=beta_final,
        cap=float(scenario_inputs.get("price_cap", DEFAULTS["price_cap"])),
    )
    price_effect = float(price_effect) * float(scenario_inputs.get("extrapolation_tail_multiplier", 1.0))

    promo_flag_baseline = float(scenario_inputs.get("promo_flag_baseline", scenario_inputs.get("promo_baseline", 0.0)))
    promo_flag_scenario = float(scenario_inputs.get("promo_flag_scenario", scenario_inputs.get("promo_scenario", promo_flag_baseline)))
    promo_intensity_baseline = float(scenario_inputs.get("promo_intensity_baseline", scenario_inputs.get("promo_baseline", 0.0)))
    promo_intensity_scenario = float(scenario_inputs.get("promo_intensity_scenario", scenario_inputs.get("promo_scenario", promo_intensity_baseline)))
    promo_effect = compute_promo_effect(
        promo_flag=promo_flag_scenario - promo_flag_baseline,
        promo_share=promo_intensity_scenario - promo_intensity_baseline,
        alpha_flag=float(scenario_inputs.get("promo_alpha_flag", DEFAULTS["promo_alpha_flag"])),
        alpha_share=float(scenario_inputs.get("promo_alpha_share", DEFAULTS["promo_alpha_share"])),
        cap=float(scenario_inputs.get("promo_cap", DEFAULTS["promo_cap"])),
    )

    freight_effect = compute_freight_effect(
        freight_ref=float(scenario_inputs.get("freight_ref", 0.0)),
        freight_scenario=float(scenario_inputs.get("freight_scenario", scenario_inputs.get("freight_ref", 0.0))),
        beta_freight=float(scenario_inputs.get("freight_beta", DEFAULTS["freight_beta"])),
        cap=float(scenario_inputs.get("freight_cap", DEFAULTS["freight_cap"])),
    )

    stock_effect = 1.0

    standard_multiplier = combine_standard_effects(
        {
            "price": price_effect,
            "promo": promo_effect,
            "freight": freight_effect,
        }
    )

    valid_shocks, shock_warnings = validate_shocks(shocks or [])
    shock_multiplier = compute_shock_multiplier(dates, valid_shocks)
    shock_units = compute_shock_units(dates, valid_shocks)

    scenario_demand = baseline_units * standard_multiplier * shock_multiplier + shock_units
    scenario_demand = np.clip(_finite_array(scenario_demand), 0.0, None)
    available_stock = scenario_inputs.get("available_stock")
    realized = apply_stock_constraint(scenario_demand, None if available_stock is None else np.asarray(available_stock, dtype=float))

    baseline_net_price = float(scenario_inputs.get("baseline_net_price", scenario_inputs.get("scenario_net_price", scenario_price)))
    scenario_net_price = float(scenario_inputs.get("scenario_net_price", scenario_price))
    baseline_unit_cost = float(scenario_inputs.get("baseline_unit_cost", scenario_inputs.get("unit_cost", scenario_price * 0.65)))
    unit_cost = float(scenario_inputs.get("unit_cost", baseline_unit_cost))
    baseline_freight_value = float(scenario_inputs.get("baseline_freight_value", scenario_inputs.get("freight_value", 0.0)))
    freight_value = float(scenario_inputs.get("freight_value", baseline_freight_value))

    baseline_revenue = _finite_array(baseline_units * baseline_net_price)
    final_revenue = _finite_array(realized * scenario_net_price)
    baseline_margin = _finite_array(baseline_units * (baseline_net_price - baseline_unit_cost - baseline_freight_value))
    final_margin = _finite_array(realized * (scenario_net_price - unit_cost - freight_value))

    promo_conf = compute_promo_confidence(int(metadata.get("promo_weeks", 0)), float(metadata.get("promo_variability", 0.0)))
    freight_conf = compute_freight_confidence(int(metadata.get("freight_changes", 0)), float(metadata.get("freight_variation", 0.0)))

    warnings: List[str] = list(shock_warnings)
    if min(price_conf["score"], promo_conf["score"], freight_conf["score"]) < 0.45:
        warnings.append("Low support detected: conservative interpretation is recommended.")
    if len(valid_shocks):
        warnings.append("Manual shock assumptions are applied.")

    return build_scenario_result(
        baseline_units=baseline_units,
        price_effect=_finite_scalar(price_effect, default=1.0),
        promo_effect=_finite_scalar(promo_effect, default=1.0),
        freight_effect=_finite_scalar(freight_effect, default=1.0),
        stock_effect=_finite_scalar(stock_effect, default=1.0),
        shock_multiplier=_finite_array(shock_multiplier, default=1.0),
        shock_units=_finite_array(shock_units, default=0.0),
        final_units=_finite_array(np.clip(realized, 0.0, None), default=0.0),
        baseline_revenue=baseline_revenue,
        final_revenue=final_revenue,
        baseline_margin=baseline_margin,
        final_margin=final_margin,
        confidence=build_confidence_summary(price_conf, promo_conf, freight_conf, shocks_present=bool(valid_shocks)),
        warnings=warnings,
    )


def build_scenario_result(**kwargs: Any) -> Dict[str, Any]:
    return kwargs
