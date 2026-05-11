from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BusinessConstraints:
    min_margin_pct: float = 0.0
    max_price_change_pct: float = 10.0
    max_discount_pct: float = 50.0
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    forbid_price_below_cost: bool = True
    max_demand_drop_pct: float = 10.0
    max_revenue_drop_pct: float = 5.0
    max_stockout_risk: float = 0.05
    min_expected_profit_uplift_pct: float = 3.0
    allowed_actions: List[str] = field(default_factory=lambda: ["do_nothing", "price_change", "discount_change", "promotion_change", "freight_change", "manual_demand_shock", "user_custom_scenario"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionCandidate:
    candidate_id: str
    action_type: str
    objective: str
    changed_factors: List[str] = field(default_factory=list)
    current_values: Dict[str, Any] = field(default_factory=dict)
    target_values: Dict[str, Any] = field(default_factory=dict)
    source: str = "generated"
    requires_real_cost: bool = False
    requires_stock_data: bool = False
    requires_factor_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionPassport:
    decision_title: str
    decision_status: str
    recommended_action: Dict[str, Any]
    expected_effect: Dict[str, Any]
    conservative_effect: Dict[str, Any]
    risk_level: str
    why_this_action: List[str]
    why_not_other_actions: List[str]
    assumptions: List[str]
    data_quality: Dict[str, Any]
    model_quality: Dict[str, Any]
    factor_support: Dict[str, Any]
    unit_economics: Dict[str, Any]
    scenario_contract: Dict[str, Any]
    blockers: List[str]
    warnings: List[str]
    test_plan: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    monitoring_metrics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionAnalysisInput:
    baseline: Dict[str, Any]
    trained_bundle: Dict[str, Any]
    data_contract: Dict[str, Any]
    model_quality_gate: Dict[str, Any]
    current_context: Dict[str, Any]
    allowed_actions: List[str]
    business_constraints: Dict[str, Any]
    objective: str
    horizon: int


@dataclass
class DecisionAnalysisResult:
    decision_status: str
    best_action: Optional[Dict[str, Any]]
    safe_option: Optional[Dict[str, Any]]
    balanced_option: Optional[Dict[str, Any]]
    aggressive_option: Optional[Dict[str, Any]]
    rejected_options: List[Dict[str, Any]]
    ranking_table: List[Dict[str, Any]]
    decision_passport: Dict[str, Any]
    test_plan: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    assumptions: List[str]
    blockers: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_test_plan(objective: str = "profit") -> Dict[str, Any]:
    return {
        "duration_days": 14,
        "scope": "selected SKU / segment",
        "primary_metric": objective,
        "secondary_metrics": ["revenue", "demand", "conversion", "stockout"],
        "success_condition": f"conservative_{objective}_delta > 3%",
        "rollback_condition": "profit_delta < -2% or demand_delta < -10%",
    }


def default_rollback_plan() -> Dict[str, Any]:
    return {
        "rollback_trigger": ["profit_delta < -2%", "demand_delta < -10%", "stockout risk above threshold"],
        "rollback_action": "return_to_previous_price_or_factor_values",
        "monitoring_frequency": "daily",
        "owner_note": "Assign a business owner before launch.",
    }


def analyze_decision(payload: DecisionAnalysisInput, evaluated_candidates: Optional[List[Dict[str, Any]]] = None) -> DecisionAnalysisResult:
    constraints = payload.business_constraints or BusinessConstraints().to_dict()
    candidates = evaluated_candidates or []
    acceptable = []
    rejected = []
    for c in candidates:
        rel = c.get("reliability", {}) if isinstance(c, dict) else {}
        status = str(rel.get("decision_status") or c.get("decision_status") or "experimental_only")
        blockers = c.get("blockers") or rel.get("blockers") or []
        if status in {"recommended", "test_recommended", "controlled_test_only"} and not blockers:
            acceptable.append(c)
        else:
            rejected.append(c)
    acceptable.sort(key=lambda x: float(x.get("decision_rank_score", 0.0) or 0.0), reverse=True)
    best = acceptable[0] if acceptable else None
    aggressive = next((c for c in acceptable if str((c.get("reliability", {}) or {}).get("decision_status", "")) != "not_recommended"), None)
    status = "recommended" if best and str((best.get("reliability", {}) or {}).get("decision_status")) == "recommended" else "controlled_test_only" if best else "not_recommended"
    blockers = [] if best else ["no_gated_candidate_available"]
    warnings = [] if best else ["Лучшее решение — не менять параметры без controlled test."]
    test_plan = default_test_plan(payload.objective)
    rollback_plan = default_rollback_plan()
    action = best or {"candidate_id": "do_nothing", "action_type": "do_nothing", "objective": "risk_reduction"}
    passport = DecisionPassport(
        decision_title="Decision Analyst result",
        decision_status=status,
        recommended_action=action,
        expected_effect=(best or {}).get("expected_effect", {}),
        conservative_effect={"uses_uncertainty_haircut": True, **((best or {}).get("expected_effect", {}) if isinstance((best or {}).get("expected_effect", {}), dict) else {})},
        risk_level=str(((best or {}).get("reliability", {}) or {}).get("risk_level", "medium" if best else "low")),
        why_this_action=["Selected only from gated candidates."] if best else ["No candidate passed gate; do nothing is safest."],
        why_not_other_actions=["Rejected candidates failed blockers, reliability, cost, model, or uncertainty gates."],
        assumptions=["Forecast/what-if/decision permissions are separated.", "Do Nothing is always a baseline decision."],
        data_quality=payload.data_contract,
        model_quality=payload.model_quality_gate,
        factor_support={},
        unit_economics={},
        scenario_contract=(best or {}).get("scenario_contract", {}),
        blockers=blockers,
        warnings=warnings,
        test_plan=test_plan,
        rollback_plan=rollback_plan,
        monitoring_metrics=["run_id", "dataset_hash", "model_version", "decision_status", "warnings", "blockers"],
    ).to_dict()
    return DecisionAnalysisResult(status, best, best, best, aggressive, rejected, candidates, passport, test_plan, rollback_plan, passport["assumptions"], blockers, warnings)
