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
    if not np.isfinite(promo_term):
        promo_term = 0.0
    clipped = float(np.clip(promo_term, -abs(cap), abs(cap)))
    effect = 1.0 + clipped
    return float(np.clip(effect, 0.05, 1.0 + abs(cap)))


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


def compute_price_effect_vector(
    reference_price: np.ndarray,
    scenario_price: np.ndarray,
    price_elasticity: float,
    cap: float = 0.35,
) -> np.ndarray:
    ref = np.clip(np.asarray(reference_price, dtype=float), 1e-6, None)
    scen = np.clip(np.asarray(scenario_price, dtype=float), 1e-6, None)
    raw = float(price_elasticity) * np.log(scen / ref)
    return np.exp(np.clip(raw, -abs(cap), abs(cap)))


def compute_promo_effect_vector(
    promo_flag_baseline: np.ndarray,
    promo_flag_scenario: np.ndarray,
    promo_share_baseline: np.ndarray,
    promo_share_scenario: np.ndarray,
    alpha_flag: float = 0.08,
    alpha_share: float = 0.20,
    cap: float = 0.40,
) -> np.ndarray:
    promo_term = (
        float(alpha_share) * (np.asarray(promo_share_scenario, dtype=float) - np.asarray(promo_share_baseline, dtype=float))
        + float(alpha_flag) * (np.asarray(promo_flag_scenario, dtype=float) - np.asarray(promo_flag_baseline, dtype=float))
    )
    promo_term = np.nan_to_num(promo_term, nan=0.0, posinf=0.0, neginf=0.0)
    effect = 1.0 + np.clip(promo_term, -abs(cap), abs(cap))
    return np.clip(effect, 0.05, 1.0 + abs(cap))


def compute_freight_effect_vector(
    freight_ref: np.ndarray,
    freight_scenario: np.ndarray,
    beta_freight: float = -0.03,
    cap: float = 0.20,
) -> np.ndarray:
    delta = np.asarray(freight_scenario, dtype=float) - np.asarray(freight_ref, dtype=float)
    raw = float(beta_freight) * delta
    eff = np.exp(np.clip(raw, -abs(cap), abs(cap)))
    return np.where(np.asarray(freight_scenario, dtype=float) >= np.asarray(freight_ref, dtype=float), np.minimum(eff, 1.0), eff)


def combine_standard_effects_vector(
    effects: Dict[str, np.ndarray],
    floor: float = 0.05,
    ceil: float = 5.0,
) -> np.ndarray:
    if not effects:
        return np.ones(0, dtype=float)
    length = len(next(iter(effects.values())))
    value = np.ones(length, dtype=float)
    for arr in effects.values():
        value *= np.clip(np.asarray(arr, dtype=float), 0.0, None)
    return np.clip(value, floor, ceil)


def apply_stock_constraint_vector(units: np.ndarray, available_stock: Optional[np.ndarray]) -> np.ndarray:
    return apply_stock_constraint(units, available_stock)
