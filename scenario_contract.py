from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ScenarioContract:
    scenario_id: str
    scenario_mode: str
    baseline_source: str
    effect_source: str
    requested_inputs: Dict[str, Any] = field(default_factory=dict)
    applied_inputs: Dict[str, Any] = field(default_factory=dict)
    price_policy: Dict[str, Any] = field(default_factory=dict)
    cost_policy: Dict[str, Any] = field(default_factory=dict)
    stock_policy: Dict[str, Any] = field(default_factory=dict)
    factor_policy: Dict[str, Any] = field(default_factory=dict)
    model_quality_gate: Dict[str, Any] = field(default_factory=dict)
    recommendation_gate: Dict[str, Any] = field(default_factory=dict)
    calculation_gate: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
