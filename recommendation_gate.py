from __future__ import annotations

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
    if cost_proxied:
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

    wape = _finite_float(model_quality.get("wape"), None)
    if wape is not None and wape > 60:
        severity = max(severity, 2)
        _push_unique(warnings_out, "Model WAPE is very high; recommendation is experimental only.")
        _push_unique(reasons, "very_high_wape")
    elif wape is not None and wape > 40:
        severity = max(severity, 1)
        _push_unique(warnings_out, "Model WAPE is high; recommendation is limited to test-only.")
        _push_unique(reasons, "high_wape")

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
    calculation_gate = "blocked" if severity >= 3 else "degraded" if severity >= 1 or warnings_out else "ok"
    return {
        "calculation_gate": calculation_gate,
        "recommendation_gate": recommendation_gate,
        "decision_status": _STATUS_BY_SEVERITY[severity],
        "reasons": reasons,
        "blockers": blockers_out,
        "warnings": warnings_out,
        "severity": severity,
    }
