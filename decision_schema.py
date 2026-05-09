from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionCandidate:
    candidate_id: str
    mode: str
    action_type: str
    title: str
    source: str
    current_value: Optional[float]
    target_value: Optional[float]
    change_pct: Optional[float]
    objective: str
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionEvaluation:
    candidate_id: str
    scenario_result: Dict[str, Any]
    baseline_result: Dict[str, Any]
    expected_effect: Dict[str, Any]
    reliability: Dict[str, Any]
    economic_checks: Dict[str, Any]
    statistical_support: Dict[str, Any]
    validation_plan: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationInput:
    source_name: str
    action_type: str
    target_value: Optional[float]
    change_pct: Optional[float]
    objective: str
    comment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionPassport:
    mode: str
    decision_title: str
    decision_status: str
    recommended_action: Dict[str, Any]
    expected_effect: Dict[str, Any]
    reliability: Dict[str, Any]
    evidence: List[str]
    limitations: List[str]
    validation_plan: Dict[str, Any]
    alternatives: Dict[str, Any]
    input_recommendation: Optional[Dict[str, Any]] = None
    audit_verdict: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
