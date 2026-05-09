from __future__ import annotations

from typing import Any, Dict, Optional

from decision_math import safe_float


def build_validation_plan(candidate: dict, reliability: dict, objective: str, volatility_info=None) -> dict:
    risk = (reliability or {}).get("risk_level", "high")
    label = (reliability or {}).get("label", "low")
    if risk == "low" and label == "high":
        days = 14
        scope = "gradual_rollout"
    elif risk == "high":
        days = 28
        scope = "limited_pilot_only"
    else:
        days = 21
        scope = "controlled_test"
    if objective == "revenue":
        success = "revenue"; secondary = ["profit", "demand"]
    elif objective == "demand":
        success = "demand"; secondary = ["revenue", "profit"]
    else:
        success = "gross_profit"; secondary = ["demand", "revenue", "margin"]
    return {"test_period_days": days, "test_scope": scope, "success_metric": success, "secondary_metrics": secondary, "rollback_condition": "profit_not_above_baseline_or_demand_drop_more_than_10_pct_or_margin_drop_more_than_3pp", "control_recommendation": "Использовать контрольную группу SKU/регионов/периодов, если возможно", "causal_note": "Результат является модельной оценкой, а не доказанной причинной связью."}


def _option_action(option: Optional[dict]) -> dict:
    if not option:
        return {"action_type": "no_change_or_test", "title": "Не менять параметр без теста", "reason": "Нет сценария с достаточной надёжностью и экономической значимостью"}
    return {"candidate_id": option.get("candidate_id"), "title": option.get("title"), "action_type": option.get("action_type"), "current_value": option.get("current_value"), "target_value": option.get("target_value"), "change_pct": option.get("change_pct")}


def build_decision_passport(mode: str, optimizer_result: dict, input_recommendation: dict | None = None, audit_result: dict | None = None) -> dict:
    best = (optimizer_result or {}).get("best_action") or (optimizer_result or {}).get("improved_solution")
    if best and best.get("action_type") == "no_change_or_test":
        option = None
    else:
        option = best
    rel = (option or {}).get("reliability") or {}
    objective = (input_recommendation or {}).get("objective") or ((option or {}).get("objective")) or "profit"
    action = _option_action(option)
    title = action.get("title") or "Не менять параметр без теста"
    expected = (option or {}).get("expected_effect") or {"demand_delta_pct": 0.0, "revenue_delta_pct": 0.0, "profit_delta_pct": 0.0, "conservative_profit_delta_pct": 0.0}
    limitations = []
    warnings = (option or {}).get("warnings") or []
    blockers = (option or {}).get("blockers") or []
    limitations.extend(warnings[:6]); limitations.extend(blockers[:4])
    limitations.append("Результат является модельной оценкой, а не доказанной причинной связью.")
    if any("guardrail" in str(x).lower() or "applied value" in str(x).lower() for x in limitations):
        limitations.append("Расчёт выполнен для ограниченного значения, а не для исходной рекомендации.")
    evidence = []
    full_rel = (option or {}).get("full_reliability") or {}
    evidence.extend(full_rel.get("reasons_positive", []) if isinstance(full_rel, dict) else [])
    if not evidence and option:
        evidence = ["Главная рекомендация выбрана по ожидаемой ценности с учётом риска."]
    if not option:
        evidence = ["Надёжного сценария для автоматического внедрения не найдено."]
    validation_plan = build_validation_plan(action, rel, objective)
    return {"mode": mode, "decision_title": title, "decision_status": rel.get("decision_status", "not_recommended" if not option else "test_recommended"), "recommended_action": action, "expected_effect": expected, "reliability": {"score": safe_float(rel.get("score"), 0.0), "label": rel.get("label", "low"), "risk_level": rel.get("risk_level", "high"), "statistical_support": rel.get("statistical_support", rel.get("support", "insufficient"))}, "evidence": evidence, "limitations": list(dict.fromkeys(limitations)), "validation_plan": validation_plan, "alternatives": {"safe": (optimizer_result or {}).get("safe_option"), "balanced": (optimizer_result or {}).get("balanced_option"), "aggressive": (optimizer_result or {}).get("aggressive_option")}, "input_recommendation": input_recommendation, "audit_verdict": (audit_result or {}).get("audit_verdict") if audit_result else None}
