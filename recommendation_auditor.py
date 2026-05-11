from __future__ import annotations

from typing import Any, Callable, Dict, List

from decision_candidate_engine import evaluate_decision_candidates, run_decision_candidate
from decision_math import clamp, safe_float, safe_pct_delta
from decision_optimizer import rank_decision_candidates
from recommendation_gate import resolve_recommendation_gate
from decision_passport import build_decision_passport



def classify_recommendation_evidence(recommendation: dict) -> dict:
    source_type = str(recommendation.get("source_type") or (recommendation.get("metadata") or {}).get("source_type") or "management_opinion")
    comment = str(recommendation.get("comment") or recommendation.get("rationale") or recommendation.get("reason") or "").strip()
    evidence = recommendation.get("evidence") or (recommendation.get("metadata") or {}).get("evidence") or {}
    has_numbers = any(recommendation.get(k) is not None for k in ["target_value", "relative_change_pct", "change_pct", "expected_uplift_pct"]) or any(ch.isdigit() for ch in comment)
    has_period = bool(recommendation.get("period") or recommendation.get("date_range") or (isinstance(evidence, dict) and evidence.get("period")))
    has_source = bool(recommendation.get("source_name") or source_type in {"experiment_result", "external_market_data", "historical_analog"})
    if source_type == "experiment_result" and has_numbers and has_period:
        quality = "strong"
    elif has_numbers and has_period and has_source:
        quality = "medium"
    elif comment or has_numbers:
        quality = "weak"
    else:
        quality = "none"
    return {"source_type": source_type, "evidence_quality": quality, "has_numbers": has_numbers, "has_period": has_period, "has_source": has_source}

def _ctx(trained_bundle, current_context=None):
    base = dict((trained_bundle or {}).get("base_ctx") or {})
    if current_context: base.update(current_context)
    return base


def build_candidate_from_recommendation(recommendation: dict, trained_bundle: dict, current_context: dict | None, horizon_days: int) -> dict:
    ctx = _ctx(trained_bundle, current_context)
    action = recommendation.get("action_type", "price_change")
    current_price = safe_float(ctx.get("price"), 0.0); current_discount = clamp(ctx.get("discount", 0.0), 0, 0.95); current_promo = clamp(ctx.get("promotion", 0.0), 0, 1); current_freight = safe_float(ctx.get("freight_value"), 0.0)
    factor_overrides = dict(ctx.get("factor_overrides") or {})
    target = recommendation.get("target_value")
    target_missing = target is None
    if target is None and recommendation.get("absolute_delta_pp") is not None and action in {"discount_change", "promotion_change"}:
        base = current_discount if action == "discount_change" else current_promo
        target = round(base + safe_float(recommendation.get("absolute_delta_pp"), 0.0) / 100.0, 10)
        target_missing = False
    if target is None and recommendation.get("relative_change_pct") is not None:
        base = current_price if action == "price_change" else current_discount if action == "discount_change" else current_promo if action == "promotion_change" else current_freight if action == "freight_change" else 1.0
        target = base * (1.0 + safe_float(recommendation.get("relative_change_pct"), 0.0) / 100.0)
        target_missing = False
    if target is None and recommendation.get("change_pct") is not None:
        base = current_price if action == "price_change" else current_discount if action == "discount_change" else current_promo if action == "promotion_change" else current_freight if action == "freight_change" else 1.0
        target = base * (1.0 + safe_float(recommendation.get("change_pct"), 0.0) / 100.0)
        target_missing = False
    current = current_price if action == "price_change" else current_discount if action == "discount_change" else current_promo if action == "promotion_change" else current_freight if action == "freight_change" else 1.0
    if target_missing:
        target = current
    else:
        target = safe_float(target, current)
    params = {"manual_price": current_price, "freight_multiplier": 1.0, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "factor_overrides": dict(factor_overrides), "overrides": {"discount": current_discount, "promotion": current_promo, "freight_value": current_freight}}
    if action == "price_change": params["manual_price"] = target
    elif action == "discount_change": params["overrides"]["discount"] = clamp(target, 0, 0.95)
    elif action == "promotion_change": params["overrides"]["promotion"] = clamp(target, 0, 1)
    elif action == "freight_change":
        params["freight_multiplier"] = 1.0
        params["overrides"]["freight_value"] = safe_float(target, current_freight)
    elif action == "demand_shock": params["demand_multiplier"] = max(0.01, target)
    metadata = {"source_name": recommendation.get("source_name"), "horizon_days": horizon_days, **(recommendation.get("metadata") or {})}
    blockers = []
    if target_missing:
        blockers.append("target_value_required")
        metadata["invalid_recommendation"] = True
        metadata["blockers"] = blockers
    relative_change_pct = safe_pct_delta(target, current)
    if action in {"discount_change", "promotion_change"}:
        metadata["absolute_delta_pp"] = (safe_float(target, 0.0) - safe_float(current, 0.0)) * 100.0
    return {"candidate_id": "external_recommendation", "mode": "improve_external_recommendation", "action_type": action, "title": recommendation.get("comment") or "Внешняя рекомендация", "source": "external", "current_value": current, "target_value": target, "change_pct": relative_change_pct, "relative_change_pct": relative_change_pct, "absolute_delta_pp": metadata.get("absolute_delta_pp"), "objective": recommendation.get("objective", "profit"), "scenario_params": params, "metadata": metadata, "blockers": blockers}


def _make_alt(cid, base, target, action, objective, ctx, source="improved"):
    current_price = safe_float(ctx.get("price"), 0.0); current_discount = clamp(ctx.get("discount", 0.0), 0, 0.95); current_promo = clamp(ctx.get("promotion", 0.0), 0, 1); current_freight = safe_float(ctx.get("freight_value"), 0.0)
    factor_overrides = dict(ctx.get("factor_overrides") or {})
    current = current_price if action == "price_change" else current_discount if action == "discount_change" else current_promo if action == "promotion_change" else current_freight if action == "freight_change" else 1.0
    params = {"manual_price": current_price, "freight_multiplier": 1.0, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "factor_overrides": dict(factor_overrides), "overrides": {"discount": current_discount, "promotion": current_promo, "freight_value": current_freight}}
    if action == "price_change": params["manual_price"] = target
    elif action == "discount_change": params["overrides"]["discount"] = clamp(target, 0, 0.95)
    elif action == "promotion_change": params["overrides"]["promotion"] = clamp(target, 0, 1)
    elif action == "freight_change":
        params["freight_multiplier"] = 1.0
        params["overrides"]["freight_value"] = safe_float(target, current_freight)
    else: params["demand_multiplier"] = max(0.01, target)
    metadata = {}
    relative_change_pct = safe_pct_delta(target, current)
    if action in {"discount_change", "promotion_change"}:
        metadata["absolute_delta_pp"] = (safe_float(target, 0.0) - safe_float(current, 0.0)) * 100.0
    return {"candidate_id": cid, "mode": "improve_external_recommendation", "action_type": "baseline" if cid == "baseline_current" else action, "title": "Текущий вариант" if cid == "baseline_current" else f"Альтернатива: {target:.4g}", "source": "baseline" if cid == "baseline_current" else source, "current_value": current, "target_value": target, "change_pct": relative_change_pct, "relative_change_pct": relative_change_pct, "absolute_delta_pp": metadata.get("absolute_delta_pp"), "objective": objective, "scenario_params": params, "metadata": metadata}


def generate_alternatives_around_recommendation(recommendation_candidate: dict, trained_bundle: dict, current_context: dict | None, objective: str, horizon_days: int) -> list[dict]:
    ctx = _ctx(trained_bundle, current_context)
    action = recommendation_candidate.get("action_type")
    current = safe_float(recommendation_candidate.get("current_value"), 0.0)
    target = safe_float(recommendation_candidate.get("target_value"), current)
    values = [current]
    if action == "price_change":
        direction = 1 if target >= current else -1
        values += [current * (1 + direction * p / 100.0) for p in (3, 5, 8, 10)] + [(current + target) / 2.0, target]
    elif action == "discount_change":
        values += [current + (target-current)*r for r in (0.5, 1.0, 0.75)]
    elif action == "promotion_change":
        values += [target, 0.0, 1.0]
    elif action == "freight_change":
        values += [current + (target-current)*r for r in (0.5, 0.75, 1.0)]
    elif action == "demand_shock":
        values += [1.0 + (target-1.0)*r for r in (0.5, 1.0)]
    seen = set(); out=[]
    for i, v in enumerate(values):
        if action == "discount_change": v = clamp(v, 0, 0.95)
        if action == "promotion_change": v = clamp(v, 0, 1)
        key = round(v, 6)
        if key in seen: continue
        seen.add(key)
        out.append(_make_alt("baseline_current" if abs(v-current) < 1e-9 else f"alt_{i}_{key}", current, v, action, objective, ctx))
    # include exact external candidate id for verdict matching
    ext = dict(recommendation_candidate); ext["candidate_id"] = "external_recommendation"; ext["source"] = "external"
    if round(target, 6) not in seen:
        out.append(ext)
    else:
        out.append(ext)
    return out


def _verdict(input_eval: dict, optimizer: dict, evidence_policy: dict | None = None) -> dict:
    rel = input_eval.get("reliability") or {}; econ = rel.get("economic_significance") or {}
    best = optimizer.get("best_action")
    objective = str((input_eval.get("candidate") or {}).get("objective") or "profit")
    gate = resolve_recommendation_gate(
        price_policy=(rel.get("component_details") or {}).get("scenario_support", {}),
        data_quality=(rel.get("component_details") or {}).get("data_quality", {}),
        model_quality=(rel.get("component_details") or {}).get("model_quality", {}),
        factor_policy=(rel.get("component_details") or {}).get("factor_support", {}),
        economic_significance=econ,
        cost_policy={"cost_proxied": bool(econ.get("cost_proxied")), "cost_missing": bool(econ.get("cost_missing")), "profit_action": objective == "profit"},
        decision_reliability={
            "base_status": rel.get("decision_status"),
            "status_namespace": "decision",
            "warnings": rel.get("warnings", []),
            "blockers": input_eval.get("blockers") or rel.get("blockers", []),
            "allow_unknown_wape_for_test_recommendation": True,
        },
    )
    gate_status = gate["decision_status"]
    input_ok = gate_status in {"recommended", "test_recommended"} and rel.get("risk_level") != "high" and not gate["blockers"]
    profit = safe_float(econ.get("conservative_profit_delta_pct", econ.get("profit_delta_pct")), -999)
    nominal_profit = safe_float(econ.get("profit_delta_pct"), profit)
    evidence_policy = evidence_policy or {}
    evidence_quality = str(evidence_policy.get("evidence_quality", "none"))
    if evidence_quality == "none":
        gate_status = "experimental_only" if gate_status != "not_recommended" else gate_status
    elif evidence_quality == "weak" and gate_status == "recommended":
        gate_status = "test_recommended"
    material_blockers = [
        str(x) for x in (input_eval.get("blockers") or rel.get("blockers") or gate.get("blockers", []))
        if "wape_unknown" not in str(x).lower() and "wape is unknown" not in str(x).lower()
    ]
    if material_blockers or (objective == "profit" and nominal_profit <= 0) or (objective in {"revenue", "demand"} and profit < -5.0):
        verdict = "reject"; reason = "Экономический эффект недопустим для выбранной цели или есть блокеры."
    elif objective == "profit" and profit < 0:
        verdict = "test_only"; reason = "Номинальный эффект положительный, но после uncertainty/risk adjustment он неустойчив; допустим только ограниченный тест."
    elif evidence_quality == "none":
        verdict = "test_only"; reason = "У внешней рекомендации нет проверяемого обоснования; допустим только экспериментальный сценарий."
    elif objective in {"revenue", "demand"} and profit < 0:
        verdict = "test_only"; reason = "Для цели revenue/demand небольшой минус прибыли допустим только как ограниченный тест."
    elif input_ok and (not best or best.get("candidate_id") == "external_recommendation"):
        verdict = "accept"; reason = "Внешняя рекомендация проходит фильтры надёжности."
    elif rel.get("risk_level") == "high" or gate_status == "experimental_only":
        verdict = "test_only"; reason = "Есть потенциал, но риск/экстраполяция требуют ограниченного теста."
    else:
        verdict = "modify"; reason = "Потенциал есть, но найден более надёжный или экономически лучший вариант."
    return {"verdict": verdict, "reason": reason, "recommendation_gate": gate_status, "recommendation_gate_details": gate, "source_type": evidence_policy.get("source_type"), "evidence_quality": evidence_quality, "main_risks": gate.get("warnings", [])[:5] + gate.get("blockers", [])[:3], "what_is_valid": rel.get("reasons_positive", []), "what_is_weak": rel.get("reasons_negative", [])[:6]}


def audit_and_improve_recommendation(results: dict, trained_bundle: dict, recommendation: dict, runner: Callable, scenario_calc_mode: str, price_guardrail_mode: str, horizon_days: int = 30, objective: str = "profit", current_context: dict | None = None) -> dict:
    objective = recommendation.get("objective") or objective
    evidence_policy = classify_recommendation_evidence(recommendation)
    cand = build_candidate_from_recommendation(recommendation, trained_bundle, current_context, horizon_days)
    alts = generate_alternatives_around_recommendation(cand, trained_bundle, current_context, objective, horizon_days)
    evaluated = evaluate_decision_candidates(results, trained_bundle, alts, runner, scenario_calc_mode, price_guardrail_mode, horizon_days, objective, current_context=current_context)
    optimizer = rank_decision_candidates(evaluated, objective=objective)
    input_eval = next((e for e in evaluated if (e.get("candidate") or {}).get("candidate_id") == "external_recommendation"), evaluated[-1] if evaluated else {})
    verdict = _verdict(input_eval, optimizer, evidence_policy)
    improved = optimizer.get("best_action")
    if improved is None:
        improved = {"action_type": "no_change_or_test", "title": "Не менять параметр без теста", "reason": "Нет сценария с достаточной надёжностью и экономической значимостью"}
    optimizer_for_passport = dict(optimizer); optimizer_for_passport["best_action"] = improved
    audit_stub = {"audit_verdict": verdict}
    passport = build_decision_passport("improve_external_recommendation", optimizer_for_passport, input_recommendation=recommendation, audit_result=audit_stub, calculation_context=current_context)
    return {"input_recommendation": recommendation, "source_type": evidence_policy["source_type"], "evidence_quality": evidence_policy["evidence_quality"], "input_evaluation": input_eval, "audit_verdict": verdict, "improved_solution": improved, "safe_option": optimizer.get("safe_option"), "balanced_option": optimizer.get("balanced_option"), "aggressive_option": optimizer.get("aggressive_option"), "alternatives_table": optimizer.get("ranking_table", []), "decision_passport": passport}
