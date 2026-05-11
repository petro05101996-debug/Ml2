from __future__ import annotations

from typing import Any, Dict, List

from decision_math import safe_float
from recommendation_gate import resolve_recommendation_gate


def _econ(e: dict, objective: str) -> float:
    eff = e.get("expected_effect") or {}
    rel = e.get("reliability") or {}
    if objective == "revenue":
        return safe_float(eff.get("revenue_delta_pct"), 0.0)
    if objective == "demand":
        return safe_float(eff.get("demand_delta_pct"), 0.0)
    if objective == "risk_reduction":
        return safe_float(rel.get("score"), 0.0)
    return safe_float(eff.get("profit_delta_pct"), 0.0)


def _row(e: dict, objective: str) -> dict:
    cand = e.get("candidate") or {}
    rel = e.get("reliability") or {}
    comps = rel.get("components") or {}
    risk = rel.get("risk_level", "high")
    risk_penalty = 30.0 if risk == "high" else 12.0 if risk == "medium" else 0.0
    blocker_penalty = 100.0 if (e.get("blockers") or rel.get("blockers")) else 0.0
    econ = _econ(e, objective)
    econ_score = max(-100.0, min(100.0, econ * 5.0 + 50.0))
    score = econ_score * 0.55 + safe_float(rel.get("score"), 0.0) * 0.30 + safe_float(comps.get("scenario_support"), 0.0) * 0.10 + safe_float(comps.get("validation_readiness"), 0.0) * 0.05 - risk_penalty - blocker_penalty
    out = dict(e)
    out["decision_rank_score"] = round(score, 4)
    out["ranking_metrics"] = {"economic_value": econ, "economic_value_score": econ_score, "risk_penalty": risk_penalty, "blocker_penalty": blocker_penalty}
    return out


def _eligible_best(e: dict, objective: str, min_profit_uplift_pct: float) -> bool:
    rel = e.get("reliability") or {}; eff = e.get("expected_effect") or {}; econ = rel.get("economic_significance") or {}
    if (e.get("blockers") or rel.get("blockers")):
        return False

    status = str(rel.get("decision_status", ""))
    gate_details = rel.get("recommendation_gate_details") if isinstance(rel.get("recommendation_gate_details"), dict) else {}
    if gate_details and gate_details.get("blockers"):
        return False

    component_details = rel.get("component_details") if isinstance(rel.get("component_details"), dict) else {}
    model_quality = component_details.get("model_quality") if isinstance(component_details.get("model_quality"), dict) else {}
    has_full_model_quality = "wape" in model_quality and model_quality.get("wape") is not None
    if has_full_model_quality:
        gate = resolve_recommendation_gate(
            price_policy=component_details.get("scenario_support", {}),
            data_quality=component_details.get("data_quality", {}),
            model_quality=model_quality,
            factor_policy=component_details.get("factor_support", {}),
            economic_significance=econ,
            cost_policy={"cost_proxied": bool(econ.get("cost_proxied")), "cost_missing": bool(econ.get("cost_missing")), "profit_action": objective == "profit"},
            decision_reliability={
                "base_status": status,
                "status_namespace": "decision",
                "warnings": rel.get("warnings", []),
                "blockers": e.get("blockers") or rel.get("blockers", []),
                "allow_unknown_wape_for_test_recommendation": True,
            },
        )
        status = str(gate.get("decision_status", status))

    if status not in {"recommended", "test_recommended"}:
        return False
    if safe_float(rel.get("score"), 0) < 65 or rel.get("risk_level") == "high":
        return False
    if objective == "profit" and safe_float(econ.get("conservative_profit_delta_pct", eff.get("conservative_profit_delta_pct")), -999) < min_profit_uplift_pct:
        return False
    return True


def _compact(e: dict | None) -> dict | None:
    if e is None: return None
    cand = e.get("candidate") or {}; rel = e.get("reliability") or {}
    scenario_result = e.get("scenario_result") or {}
    effective = scenario_result.get("effective_scenario") or {}
    return {"candidate_id": cand.get("candidate_id"), "title": cand.get("title"), "action_type": cand.get("action_type"), "objective": cand.get("objective"), "current_value": cand.get("current_value"), "target_value": cand.get("target_value"), "change_pct": cand.get("change_pct"), "decision_rank_score": e.get("decision_rank_score"), "expected_effect": e.get("expected_effect"), "reliability": {"score": rel.get("score"), "risk_level": rel.get("risk_level"), "decision_status": rel.get("decision_status"), "recommendation_gate": rel.get("recommendation_gate"), "label": rel.get("label"), "statistical_support": (rel.get("statistical_support") or {}).get("level")}, "requested_price": effective.get("requested_price_gross", scenario_result.get("requested_price")), "applied_price": effective.get("applied_price_gross", scenario_result.get("model_price")), "price_clipped": bool(effective.get("price_clipped", scenario_result.get("price_clipped", False))), "support_label": scenario_result.get("support_label"), "confidence_label": scenario_result.get("confidence_label"), "full_reliability": rel, "warnings": e.get("warnings", []), "blockers": e.get("blockers", [])}


def rank_decision_candidates(evaluated_candidates: list[dict], objective: str = "profit", min_profit_uplift_pct: float = 3.0) -> dict:
    ranked = [_row(e, objective) for e in (evaluated_candidates or [])]
    ranked.sort(key=lambda x: x.get("decision_rank_score", -9999), reverse=True)
    options = select_decision_options(ranked, objective, min_profit_uplift_pct=min_profit_uplift_pct)
    options["ranked_candidates"] = ranked
    options["ranking_table"] = [_compact(e) for e in ranked]
    return options


def select_decision_options(ranked_candidates: list[dict], objective: str = "profit", min_profit_uplift_pct: float = 3.0) -> dict:
    valid = [e for e in ranked_candidates if _eligible_best(e, objective, min_profit_uplift_pct)]
    best = valid[0] if valid else None
    safe = next((e for e in ranked_candidates if _eligible_best(e, objective, min_profit_uplift_pct) and (e.get("reliability") or {}).get("risk_level") == "low" and safe_float((e.get("reliability") or {}).get("score"), 0) >= 75 and safe_float((e.get("expected_effect") or {}).get("conservative_profit_delta_pct", (e.get("expected_effect") or {}).get("profit_delta_pct")), -1) > 0 and not (e.get("blockers") or [])), None)
    balanced = valid[0] if valid else None
    aggressive_pool = [
        e for e in ranked_candidates
        if not (e.get("blockers") or (e.get("reliability") or {}).get("blockers"))
        and not bool((e.get("reliability") or {}).get("technical_error"))
        and (e.get("reliability") or {}).get("decision_status") in {"recommended", "test_recommended", "controlled_test_only", "experimental_only"}
    ]
    aggressive_pool = sorted(aggressive_pool, key=lambda e: _econ(e, objective), reverse=True)
    aggressive = aggressive_pool[0] if aggressive_pool else None
    notrec = [e for e in ranked_candidates if (e.get("reliability") or {}).get("decision_status") == "not_recommended"]
    summary_text = (
        "Найден лучший вариант среди проверенных сценариев с учётом экономического эффекта и риска."
        if best
        else "Надёжного решения для рекомендации нет. Лучшее действие — не менять параметры без ограниченного теста."
    )
    return {"best_action": _compact(best), "safe_option": _compact(safe), "balanced_option": _compact(balanced), "aggressive_option": _compact(aggressive), "not_recommended_options": [_compact(e) for e in notrec], "ranking_table": [_compact(e) for e in ranked_candidates], "summary": {"message": summary_text, "valid_candidate_count": len(valid), "candidate_count": len(ranked_candidates)}}
