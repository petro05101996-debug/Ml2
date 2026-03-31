from __future__ import annotations

from typing import Any, Dict, List

PRESET_KEEP_CURRENT = "Keep current price"
PRESET_CAUTIOUS_INCREASE = "Cautious increase"
PRESET_AGGRESSIVE_INCREASE = "Aggressive increase"
PRESET_LOWER_FOR_VOLUME = "Lower price for volume"
PRESET_PROMO_PUSH = "Promo push"
PRESET_COST_FREIGHT_STRESS = "Higher cost / freight stress"
PRESET_LIMITED_STOCK = "Limited stock scenario"

REQUIRED_PRESET_KEYS = [
    PRESET_KEEP_CURRENT,
    PRESET_CAUTIOUS_INCREASE,
    PRESET_AGGRESSIVE_INCREASE,
    PRESET_LOWER_FOR_VOLUME,
    PRESET_PROMO_PUSH,
    PRESET_COST_FREIGHT_STRESS,
    PRESET_LIMITED_STOCK,
]


def build_default_scenario_inputs(base_price: float, horizon_days: int, base_ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    default_names = ["Baseline", "Scenario A", "Scenario B", "Scenario C"]
    default_price_mult = [1.0, 1.03, 0.97, 1.08]
    rating_default = float(base_ctx.get("review_score", base_ctx.get("rating", 4.5)))
    reviews_default = float(base_ctx.get("reviews_count", 0.0))
    return [
        {
            "name": default_names[i],
            "price": base_price * default_price_mult[i],
            "demand_multiplier": 1.0,
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "discount_multiplier": 1.0,
            "stock_cap": 0.0,
            "promotion": 0.0,
            "rating": rating_default,
            "reviews_count": reviews_default,
            "horizon_days": int(horizon_days),
        }
        for i in range(4)
    ]


def build_seller_scenario_presets(base_price: float, base_ctx: Dict[str, Any], horizon_days: int) -> Dict[str, Dict[str, float]]:
    base_stock = float(base_ctx.get("stock", 0.0) or 0.0)
    return {
        PRESET_KEEP_CURRENT: {
            "price": float(base_price),
            "discount": float(base_ctx.get("discount", 0.0) or 0.0),
            "demand_multiplier": 1.00,
            "promotion": float(base_ctx.get("promotion", 0.0) or 0.0),
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "stock_cap": max(base_stock, 0.0),
            "horizon_days": int(horizon_days),
        },
        PRESET_CAUTIOUS_INCREASE: {
            "price": float(base_price) * 1.02,
            "discount": 0.02,
            "demand_multiplier": 0.99,
            "promotion": float(base_ctx.get("promotion", 0.0) or 0.0),
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "stock_cap": max(base_stock, 0.0),
            "horizon_days": int(horizon_days),
        },
        PRESET_AGGRESSIVE_INCREASE: {
            "price": float(base_price) * 1.07,
            "discount": 0.03,
            "demand_multiplier": 0.95,
            "promotion": float(base_ctx.get("promotion", 0.0) or 0.0),
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "stock_cap": max(base_stock, 0.0),
            "horizon_days": int(horizon_days),
        },
        PRESET_LOWER_FOR_VOLUME: {
            "price": float(base_price) * 0.96,
            "discount": 0.05,
            "demand_multiplier": 1.05,
            "promotion": max(0.05, float(base_ctx.get("promotion", 0.0) or 0.0)),
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "stock_cap": max(base_stock, 0.0),
            "horizon_days": int(horizon_days),
        },
        PRESET_PROMO_PUSH: {
            "price": float(base_price),
            "discount": 0.08,
            "demand_multiplier": 1.06,
            "promotion": max(0.15, float(base_ctx.get("promotion", 0.0) or 0.0)),
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "stock_cap": max(base_stock, 0.0),
            "horizon_days": int(horizon_days),
        },
        PRESET_COST_FREIGHT_STRESS: {
            "price": float(base_price) * 1.03,
            "discount": float(base_ctx.get("discount", 0.0) or 0.0),
            "demand_multiplier": 0.98,
            "promotion": float(base_ctx.get("promotion", 0.0) or 0.0),
            "freight_multiplier": 1.20,
            "cost_multiplier": 1.12,
            "stock_cap": max(base_stock, 0.0),
            "horizon_days": int(horizon_days),
        },
        PRESET_LIMITED_STOCK: {
            "price": float(base_price) * 1.01,
            "discount": float(base_ctx.get("discount", 0.0) or 0.0),
            "demand_multiplier": 1.0,
            "promotion": float(base_ctx.get("promotion", 0.0) or 0.0),
            "freight_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "stock_cap": max(1.0, base_stock * 0.75) if base_stock > 0 else 20.0,
            "horizon_days": int(horizon_days),
        },
    }
