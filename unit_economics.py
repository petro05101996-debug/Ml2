from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class UnitEconomicsResult:
    gross_price: float
    discount_rate: float
    net_unit_price: float
    realized_units: float
    unit_cost: Optional[float]
    unit_freight: float
    revenue: float
    unit_margin: Optional[float]
    profit: Optional[float]
    profit_is_reliable: bool
    cost_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_discount_rate(value: Any, *, price: Any = None, semantic: str = "discount_rate") -> float:
    try:
        raw = float(value)
    except Exception:
        raw = 0.0
    if semantic in {"discount_pct", "pct", "percent"}:
        rate = raw / 100.0
    elif semantic in {"discount_amount", "discount_value", "amount"}:
        try:
            p = float(price)
        except Exception:
            p = 0.0
        rate = raw / p if p > 0 else 0.0
    else:
        rate = raw
        if raw > 1.0 and raw <= 100.0:
            # Explicitly treat untyped values above one as percent, not as 95% clipping.
            rate = raw / 100.0
    return max(0.0, min(0.95, rate))


def calculate_unit_economics(
    *,
    gross_price: Any,
    realized_units: Any,
    discount_rate: Any = 0.0,
    unit_cost: Any = None,
    unit_freight: Any = 0.0,
    cost_source: str = "missing",
) -> UnitEconomicsResult:
    price = max(0.0, float(gross_price or 0.0))
    units = max(0.0, float(realized_units or 0.0))
    discount = normalize_discount_rate(discount_rate, price=price)
    freight = max(0.0, float(unit_freight or 0.0))
    net_unit_price = price * (1.0 - discount)
    revenue = units * net_unit_price
    cost_val: Optional[float]
    try:
        cost_val = float(unit_cost)
    except Exception:
        cost_val = None
    if cost_val is None:
        unit_margin = None
        profit = None
    else:
        cost_val = max(0.0, cost_val)
        unit_margin = net_unit_price - cost_val - freight
        profit = units * unit_margin
    return UnitEconomicsResult(
        gross_price=price,
        discount_rate=discount,
        net_unit_price=net_unit_price,
        realized_units=units,
        unit_cost=cost_val,
        unit_freight=freight,
        revenue=revenue,
        unit_margin=unit_margin,
        profit=profit,
        profit_is_reliable=str(cost_source) == "provided",
        cost_source=str(cost_source or "missing"),
    )


def reconcile_price_revenue_quantity(price: Any, revenue: Any, quantity: Any, discount_rate: Any = 0.0) -> Dict[str, Any]:
    try:
        p = float(price)
        r = float(revenue)
        q = float(quantity)
    except Exception:
        return {"status": "unknown", "diff_pct": None, "warnings": [], "blockers": []}
    if q <= 0 or r <= 0:
        return {"status": "unknown", "diff_pct": None, "warnings": [], "blockers": []}
    observed_net_price = r / q
    expected_net_price = p * (1.0 - normalize_discount_rate(discount_rate, price=p))
    diff_pct = abs(observed_net_price - expected_net_price) / max(abs(observed_net_price), 1e-9) * 100.0
    warnings = []
    blockers = []
    status = "ok"
    if diff_pct > 15.0:
        status = "blocked"
        blockers.append("price_revenue_quantity_discount_mismatch_gt_15pct")
    elif diff_pct > 5.0:
        status = "warning"
        warnings.append("price_revenue_quantity_discount_mismatch_5_15pct")
    return {"status": status, "diff_pct": diff_pct, "warnings": warnings, "blockers": blockers}
