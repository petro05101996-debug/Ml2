from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_price_effect(
    reference_price: float,
    scenario_price: float,
    price_elasticity: float,
    cap: float = 0.35,
) -> float:
    ref = max(float(reference_price), 1e-6)
    scen = max(float(scenario_price), 1e-6)
    beta = float(price_elasticity)
    raw = beta * np.log(scen / ref)
    return float(np.exp(np.clip(raw, -abs(cap), abs(cap))))


def compute_promo_effect(
    promo_flag: float,
    promo_share: float,
    alpha_flag: float = 0.08,
    alpha_share: float = 0.20,
    cap: float = 0.40,
) -> float:
    promo_term = alpha_share * float(promo_share) + alpha_flag * float(promo_flag)
    clipped = float(np.clip(promo_term, -abs(cap), abs(cap)))
    effect = float(np.exp(clipped))
    return float(np.clip(effect, 0.25, 4.0))


def compute_freight_effect(
    freight_ref: float,
    freight_scenario: float,
    beta_freight: float = -0.03,
    cap: float = 0.20,
) -> float:
    delta = float(freight_scenario) - float(freight_ref)
    raw = float(beta_freight) * delta
    eff = float(np.exp(np.clip(raw, -abs(cap), abs(cap))))
    return min(eff, 1.0) if freight_scenario >= freight_ref else eff


def combine_standard_effects(effects: Dict[str, float], floor: float = 0.05, ceil: float = 5.0) -> float:
    value = 1.0
    for v in effects.values():
        value *= max(float(v), 0.0)
    return float(np.clip(value, floor, ceil))


def apply_stock_constraint(units: np.ndarray, available_stock: Optional[np.ndarray]) -> np.ndarray:
    if available_stock is None:
        return np.clip(units, 0.0, None)
    return np.minimum(np.clip(units, 0.0, None), np.clip(available_stock, 0.0, None))
