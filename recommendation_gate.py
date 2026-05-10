from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import math


_GATE_SEVERITY = {
    "eligible": 0,
    "recommended": 0,
    "test_only": 1,
    "test_recommended": 1,
    "experimental_only": 2,
    "not_recommended": 3,
}
_STATUS_BY_SEVERITY = {
    0: "recommended",
    1: "test_recommended",
    2: "experimental_only",
    3: "not_recommended",
}
_CONTRACT_BY_SEVERITY = {
    0: "eligible",
    1: "test_only",
    2: "experimental_only",
    3: "not_recommended",
}
PRODUCTION_STABLE_VERDICTS = {"stable", "moderately_stable"}
TEST_ONLY_VERDICTS = {"test_only_unstable", "unstable_test_only"}
EXPERIMENTAL_VERDICTS = {"experimental_unstable", "unstable_experimental_only"}


@dataclass
class RecommendationGateInput:
    data_quality: Dict[str, Any] = field(default_factory=dict)
    model_quality: Dict[str, Any] = field(default_factory=dict)
    factor_support: Dict[str, Any] = field(default_factory=dict)
    scenario_support: Dict[str, Any] = field(default_factory=dict)
    economic_significance: Dict[str, Any] = field(default_factory=dict)
    price_policy: Dict[str, Any] = field(default_factory=dict)
    cost_policy: Dict[str, Any] = field(default_factory=dict)
    monotonicity_policy: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationGateResult:
    calculation_gate: str
    recommendation_gate: str
    decision_status: str
    reasons: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    severity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calculation_gate": self.calculation_gate,
            "recommendation_gate": self.recommendation_gate,
            "decision_status": self.decision_status,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "severity": int(self.severity),
        }


def evaluate_price_monotonic_sanity(
    current_price: Any,
    scenario_price: Any,
    current_demand: Any,
    scenario_demand: Any,
) -> Dict[str, Any]:
    """Conservative economic sanity check for price-change scenarios.

    Flags cases where the model implies demand increases after a price increase
    or demand decreases after a price decrease with other factors held constant.
    Such output can be correlation rather than causal price response.
    """
    cp = _finite_float(current_price, None)
    sp = _finite_float(scenario_price, None)
    cd = _finite_float(current_demand, None)
    sd = _finite_float(scenario_demand, None)
    result = {
        "ok": True,
        "status": "passed",
        "max_recommendation_status": "recommended",
        "warnings": [],
        "blockers": [],
        "price_delta_pct": None,
        "demand_delta_pct": None,
    }
    if cp is None or sp is None or cd is None or sd is None or cp <= 0 or abs(sp - cp) <= 1e-9:
        return result
    price_delta = (sp - cp) / cp * 100.0
    demand_delta = ((sd - cd) / max(abs(cd), 1e-9) * 100.0) if cd is not None else 0.0
    result["price_delta_pct"] = price_delta
    result["demand_delta_pct"] = demand_delta
    violates = (price_delta > 0 and demand_delta > 0) or (price_delta < 0 and demand_delta < 0)
    if violates:
        msg = (
            "Economic sanity check failed: the model shows demand moving in the same direction as price. "
            "This can be correlation rather than a causal price effect; use a controlled test before rollout."
        )
        result.update({
            "ok": False,
            "status": "failed",
            "max_recommendation_status": "test_recommended",
            "warnings": [msg],
            "blockers": ["price_monotonicity_failed"],
        })
    return result


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _push_unique(items: List[str], message: str) -> None:
    if message and message not in items:
        items.append(message)


def _severity_from_status(status: Any, default: int = 0) -> int:
    return _GATE_SEVERITY.get(str(status or "").strip(), default)


def resolve_recommendation_gate(
    price_policy: Dict[str, Any] | None = None,
    data_quality: Dict[str, Any] | None = None,
    model_quality: Dict[str, Any] | None = None,
    factor_policy: Dict[str, Any] | None = None,
    economic_significance: Dict[str, Any] | None = None,
    cost_policy: Dict[str, Any] | None = None,
    decision_reliability: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve one production recommendation gate from scenario, model and economic signals.

    The function is intentionally deterministic and conservative. It can be used for a
    scenario contract (``eligible``/``test_only``/``experimental_only``) or for decision
    reliability (``recommended``/``test_recommended``/``experimental_only``) by passing
    ``decision_reliability={"status_namespace": "decision", "base_status": ...}``.
    """
    price_policy = _as_dict(price_policy)
    data_quality = _as_dict(data_quality)
    model_quality = _as_dict(model_quality)
    factor_policy = _as_dict(factor_policy)
    economic_significance = _as_dict(economic_significance)
    cost_policy = _as_dict(cost_policy)
    decision_reliability = _as_dict(decision_reliability)

    warnings_out: List[str] = list(dict.fromkeys(str(x) for x in decision_reliability.get("warnings", []) if x))
    blockers_out: List[str] = list(dict.fromkeys(str(x) for x in decision_reliability.get("blockers", []) if x))
    reasons: List[str] = []

    namespace = str(decision_reliability.get("status_namespace") or "contract")
    base_status = decision_reliability.get("base_status")
    severity = _severity_from_status(base_status, default=0)

    if blockers_out or bool(decision_reliability.get("has_blockers")):
        severity = max(severity, 3)
        _push_unique(reasons, "blockers_present")

    price_out_of_range = bool(price_policy.get("price_out_of_range") or price_policy.get("clip_applied"))
    extrapolation = bool(price_policy.get("extrapolation_applied"))
    elasticity_source = str(price_policy.get("elasticity_source") or "")
    if price_out_of_range or extrapolation:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Price is outside historical support; recommendation is limited to controlled testing.")
        _push_unique(reasons, "price_guardrail_limited_to_test")
    if extrapolation and elasticity_source.startswith("fallback_prior"):
        severity = max(severity, 2)
        _push_unique(warnings_out, "Extrapolation uses fallback prior elasticity; production recommendation is experimental only.")
        _push_unique(reasons, "fallback_elasticity_extrapolation")

    cost_proxied = bool(cost_policy.get("cost_proxied") or economic_significance.get("cost_proxied"))
    cost_missing = bool(cost_policy.get("cost_missing") or economic_significance.get("cost_missing"))
    profit_action = bool(cost_policy.get("profit_action") or economic_significance.get("profit_action") or economic_significance.get("profit_delta_pct") is not None)
    if cost_missing and profit_action:
        severity = max(severity, 3)
        _push_unique(blockers_out, "Cost is missing; profit-based price recommendation is blocked.")
        _push_unique(reasons, "cost_missing_blocks_profit_recommendation")
    elif cost_proxied and profit_action:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Cost is proxied; financial recommendation is limited to test-only.")
        _push_unique(reasons, "cost_proxied")

    if bool(data_quality.get("flat_sales") or data_quality.get("flat_history")):
        severity = max(severity, 1)
        _push_unique(warnings_out, "Flat history limits recommendation confidence.")
        _push_unique(reasons, "flat_history")

    unique_price_count = _finite_float(factor_policy.get("unique_price_count"), None)
    if unique_price_count is not None and unique_price_count < 4:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Too few unique historical prices for an automatic price recommendation.")
        _push_unique(reasons, "low_unique_price_count")

    stockout_share = _finite_float(model_quality.get("stockout_share"), None)
    if stockout_share is not None and stockout_share > 0.30:
        severity = max(severity, 3)
        _push_unique(blockers_out, "Stockout share exceeds 30%; observed sales cannot support a price/demand recommendation.")
        _push_unique(reasons, "stockout_share_above_30_not_recommended")
    elif stockout_share is not None and stockout_share > 0.15:
        severity = max(severity, 2)
        _push_unique(warnings_out, "Stockout share is 15–30%; sales model cannot be treated as clean demand; experimental only.")
        _push_unique(reasons, "stockout_share_15_30_experimental")
    elif stockout_share is not None and stockout_share > 0.05:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Stockout share is 5–15%; validate demand recommendations with a controlled test.")
        _push_unique(reasons, "stockout_share_5_15_test_only")

    naive_improvement_pct = _finite_float(model_quality.get("naive_improvement_pct"), None)
    if naive_improvement_pct is not None and naive_improvement_pct < 5.0:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Model does not materially beat naive baselines; recommendation is limited to testing.")
        _push_unique(reasons, "model_not_better_than_naive_baseline")

    rolling = _as_dict(model_quality.get("rolling_retrain_backtest"))
    rolling_verdict = str(rolling.get("verdict", ""))
    if rolling_verdict in EXPERIMENTAL_VERDICTS or rolling_verdict.startswith("experimental"):
        severity = max(severity, 2)
        _push_unique(warnings_out, "Rolling retrain backtest is unstable; recommendation is experimental only.")
        _push_unique(reasons, "rolling_retrain_unstable")
    elif rolling_verdict in TEST_ONLY_VERDICTS or rolling_verdict.startswith("test_only"):
        severity = max(severity, 1)
        _push_unique(warnings_out, "Rolling retrain backtest is not stable enough for automatic recommendation.")
        _push_unique(reasons, "rolling_retrain_test_only")

    wape_raw = model_quality.get("wape")
    wape = _finite_float(wape_raw, None)
    if wape is None:
        if bool(decision_reliability.get("allow_unknown_wape_for_test_recommendation")):
            severity = max(severity, 1)
            _push_unique(warnings_out, "Model WAPE is unknown; decision-support recommendation is limited to controlled testing.")
            _push_unique(reasons, "wape_unknown_test_only")
        else:
            severity = max(severity, 3)
            _push_unique(blockers_out, "Model WAPE is unknown; production recommendation is blocked.")
            _push_unique(reasons, "wape_unknown_not_recommended")
    elif wape > 40:
        severity = max(severity, 3)
        _push_unique(blockers_out, "Model WAPE exceeds 40%; use simulation only, not a recommendation.")
        _push_unique(reasons, "wape_above_40_not_recommended")
    elif wape > 30:
        severity = max(severity, 2)
        _push_unique(warnings_out, "Model WAPE is 30–40%; recommendation is experimental only.")
        _push_unique(reasons, "wape_30_40_experimental_only")
    elif wape > 15:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Model WAPE is 15–30%; validate via controlled test before rollout.")
        _push_unique(reasons, "wape_15_30_test_only")

    conservative_profit = _finite_float(economic_significance.get("conservative_profit_delta_pct"), None)
    if conservative_profit is not None and conservative_profit <= 0:
        severity = max(severity, 2)
        _push_unique(warnings_out, "Conservative profit uplift is non-positive.")
        _push_unique(reasons, "non_positive_conservative_profit")

    profit_delta = abs(_finite_float(economic_significance.get("profit_delta_pct"), 0.0) or 0.0)
    uncertainty = _as_dict(economic_significance.get("uncertainty"))
    expected_error = _finite_float(economic_significance.get("expected_model_error_pct", uncertainty.get("expected_model_error_pct")), None)
    if expected_error is not None and profit_delta < expected_error:
        severity = max(severity, 2)
        _push_unique(warnings_out, "Effect is smaller than expected model error.")
        _push_unique(reasons, "effect_smaller_than_model_error")

    ood_importance_score = _finite_float(factor_policy.get("ood_importance_score"), 0.0) or 0.0
    if ood_importance_score >= 10.0 or bool(factor_policy.get("important_factor_out_of_range")):
        severity = max(severity, 2)
        _push_unique(warnings_out, "Important factor is outside historical range; recommendation is experimental only.")
        _push_unique(reasons, "importance_weighted_ood")
    elif ood_importance_score > 0.0:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Some changed factors are outside historical range; validate with a controlled test.")
        _push_unique(reasons, "minor_importance_weighted_ood")

    monotonicity_policy = _as_dict(decision_reliability.get("monotonicity_policy") or price_policy.get("monotonicity_policy") or economic_significance.get("monotonicity_policy"))
    if str(monotonicity_policy.get("status", "")).lower() == "failed" or bool(monotonicity_policy.get("failed")):
        severity = max(severity, 2)
        _push_unique(warnings_out, "Economic sanity check failed: clean price-only demand moved in the same direction as price; price action is experimental only.")
        _push_unique(reasons, "price_monotonicity_failed")

    if bool(factor_policy.get("manual_demand_shock_main_driver")):
        severity = max(severity, 1)
        _push_unique(warnings_out, "Manual demand shock is a hypothesis; use controlled tests only.")
        _push_unique(reasons, "manual_demand_shock")

    if bool(model_quality.get("catboost_advanced_insufficient_data")):
        severity = max(severity, 1)
        _push_unique(warnings_out, "Advanced CatBoost mode has insufficient support data.")
        _push_unique(reasons, "advanced_mode_insufficient_data")

    severity = max(0, min(3, severity))
    recommendation_gate = _STATUS_BY_SEVERITY[severity] if namespace == "decision" else _CONTRACT_BY_SEVERITY[severity]
    calculation_blocked = bool(
        decision_reliability.get("calculation_blocked")
        or data_quality.get("invalid_input")
        or data_quality.get("empty_baseline")
        or price_policy.get("invalid_price")
        or price_policy.get("calculation_failed")
    )
    calculation_gate = "blocked" if calculation_blocked else "degraded" if severity >= 1 or warnings_out or blockers_out else "ok"
    usage_policy = {
        "can_show_forecast": not calculation_blocked,
        "can_show_what_if": not calculation_blocked,
        "can_recommend_action": _STATUS_BY_SEVERITY[severity] == "recommended",
    }
    result = RecommendationGateResult(
        calculation_gate=calculation_gate,
        recommendation_gate=recommendation_gate,
        decision_status=_STATUS_BY_SEVERITY[severity],
        reasons=reasons,
        blockers=blockers_out,
        warnings=warnings_out,
        severity=severity,
    ).to_dict()
    result["calculation_status"] = calculation_gate
    result["recommendation_status"] = _STATUS_BY_SEVERITY[severity]
    result["usage_policy"] = usage_policy
    return result
