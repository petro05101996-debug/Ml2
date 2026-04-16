from __future__ import annotations

from typing import Dict


def _tier(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def compute_price_confidence(price_changes: int, price_span: float, stability: float) -> Dict[str, float | str]:
    score = min(1.0, max(0.0, 0.45 * min(price_changes / 8.0, 1.0) + 0.35 * min(price_span / 0.12, 1.0) + 0.20 * max(0.0, min(stability, 1.0))))
    return {"score": float(score), "label": _tier(score)}


def compute_promo_confidence(promo_weeks: int, promo_variability: float) -> Dict[str, float | str]:
    score = min(1.0, max(0.0, 0.65 * min(promo_weeks / 8.0, 1.0) + 0.35 * min(promo_variability, 1.0)))
    return {"score": float(score), "label": _tier(score)}


def compute_freight_confidence(freight_changes: int, freight_variation: float) -> Dict[str, float | str]:
    score = min(1.0, max(0.0, 0.65 * min(freight_changes / 10.0, 1.0) + 0.35 * min(freight_variation, 1.0)))
    return {"score": float(score), "label": _tier(score)}


def build_confidence_summary(price: Dict[str, float | str], promo: Dict[str, float | str], freight: Dict[str, float | str], shocks_present: bool) -> Dict[str, Dict[str, float | str]]:
    return {
        "price": price,
        "promo": promo,
        "freight": freight,
        "shocks": {"label": "manual assumption" if shocks_present else "none", "score": 0.3 if shocks_present else 1.0},
    }
