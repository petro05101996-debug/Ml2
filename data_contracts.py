from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


FIELD_NAMES = [
    "date",
    "product_id",
    "price",
    "quantity",
    "revenue",
    "cost",
    "discount",
    "freight_value",
    "stock",
    "promotion",
    "category",
    "region",
    "channel",
    "segment",
]


@dataclass
class FieldContract:
    field: str
    present_in_input: bool = False
    source: str = "missing"
    confidence: str = "low"
    used_in_model: bool = False
    used_in_scenario: bool = False
    used_in_economics: bool = False
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetSemantics:
    observed_sales: bool = True
    clean_demand: bool = False
    stock_observed: bool = False
    stockout_share: Optional[float] = None
    demand_censoring_risk: str = "unknown"
    statement: str = "Model forecasts observed sales, not guaranteed uncensored market demand."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataContract:
    fields: Dict[str, FieldContract]
    source_contract: Dict[str, str]
    target_semantics: TargetSemantics
    input_grain: str = "unknown"
    missing_dates_policy: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "source_contract": dict(self.source_contract),
            "target_semantics": self.target_semantics.to_dict(),
            "input_grain": self.input_grain,
            "missing_dates_policy": self.missing_dates_policy,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
        }


def build_field_contracts(mapping: Dict[str, Optional[str]], columns: List[str]) -> Dict[str, FieldContract]:
    present = {str(c) for c in columns}
    out: Dict[str, FieldContract] = {}
    economics = {"price", "quantity", "revenue", "cost", "discount", "freight_value"}
    scenario = {"price", "discount", "freight_value", "stock", "promotion", "cost"}
    model = {"price", "discount", "freight_value", "promotion"}
    for field_name in FIELD_NAMES:
        mapped = mapping.get(field_name)
        is_present = bool(mapped is not None and str(mapped) in present)
        out[field_name] = FieldContract(
            field=field_name,
            present_in_input=is_present,
            source="provided" if is_present else "missing",
            confidence="high" if is_present else "low",
            used_in_model=field_name in model,
            used_in_scenario=field_name in scenario,
            used_in_economics=field_name in economics,
        )
    return out


def infer_stockout_share(stock_present: bool, stock_values: Any, sales_values: Any) -> Optional[float]:
    if not stock_present:
        return None
    try:
        import numpy as np
        import pandas as pd

        stock = pd.to_numeric(stock_values, errors="coerce")
        sales = pd.to_numeric(sales_values, errors="coerce")
        observed = stock.notna() & sales.notna() & (sales > 0)
        if not bool(observed.any()):
            return 0.0
        return float(((stock[observed] <= sales[observed]) | (stock[observed] <= 0)).mean())
    except Exception:
        return None


def censoring_risk(stock_present: bool, stockout_share: Optional[float]) -> str:
    if not stock_present:
        return "unknown"
    if stockout_share is None:
        return "unknown"
    if stockout_share > 0.30:
        return "high"
    if stockout_share > 0.15:
        return "medium_high"
    if stockout_share > 0.05:
        return "medium"
    return "low"
