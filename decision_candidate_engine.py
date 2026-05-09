from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from decision_math import clamp, finite_or_none, safe_float, safe_pct_delta
from decision_reliability import evaluate_decision_reliability


def _base_ctx(trained_bundle: dict, current_context: Optional[dict] = None) -> dict:
    ctx = dict((trained_bundle or {}).get("base_ctx") or {})
    if current_context:
        ctx.update({k: v for k, v in current_context.items() if v is not None})
    return ctx


def _candidate(cid: str, action_type: str, title: str, source: str, current, target, objective: str, params: dict, metadata: Optional[dict] = None) -> dict:
    cp = safe_pct_delta(target, current) if current is not None and target is not None else None
    metadata = dict(metadata or {})
    if action_type in {"discount_change", "promotion_change"} and current is not None and target is not None:
        metadata.setdefault("absolute_delta_pp", (safe_float(target, 0.0) - safe_float(current, 0.0)) * 100.0)
    return {"candidate_id": cid, "mode": "find_best_decision", "action_type": action_type, "title": title, "source": source, "current_value": current, "target_value": target, "change_pct": cp, "objective": objective, "scenario_params": params, "metadata": metadata}


def _numeric_column(df: Any, name: str):
    if hasattr(df, "columns") and name in df.columns:
        try:
            return df[name].dropna().astype(float)
        except Exception:
            return None
    return None


def _dedupe(candidates: List[dict]) -> List[dict]:
    seen = set(); deduped: List[dict] = []
    for c in candidates:
        key = (c.get("action_type"), round(safe_float(c.get("target_value"), -999999), 6), c.get("source"))
        if key not in seen:
            seen.add(key); deduped.append(c)
    return deduped


def _price_candidate(current_price: float, target: float, objective: str, current_discount: float, current_promo: float, horizon_days: int, source: str, support_zone: str) -> dict:
    pct = safe_pct_delta(target, current_price)
    return _candidate(
        f"price_{support_zone}_{target:.4f}",
        "price_change",
        f"Цена {target:.2f} ({pct:+.1f}%)",
        source,
        current_price,
        target,
        objective,
        {"manual_price": target, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "overrides": {"discount": current_discount, "promotion": current_promo}},
        {"horizon_days": horizon_days, "support_zone": support_zone, "extrapolated": support_zone != "safe"},
    )


def generate_decision_candidates(trained_bundle: dict, current_context: dict | None, objective: str = "profit", allowed_actions: list[str] | None = None, horizon_days: int = 30, price_search_pct: float = 0.20) -> list[dict]:
    ctx = _base_ctx(trained_bundle, current_context)
    current_price = safe_float(ctx.get("price"), 0.0)
    current_discount = clamp(ctx.get("discount", 0.0), 0.0, 0.95)
    current_promo = clamp(ctx.get("promotion", ctx.get("promo", 0.0)), 0.0, 1.0)
    allowed = set(allowed_actions or ["price_change", "discount_change", "promotion_change", "demand_shock"])
    daily = (trained_bundle or {}).get("daily_base")
    out: List[dict] = []
    out.append(_candidate("baseline_current", "baseline", "Текущий вариант", "baseline", current_price, current_price, objective, {"manual_price": current_price, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "overrides": {"discount": current_discount, "promotion": current_promo}}, {"horizon_days": horizon_days, "support_zone": "baseline"}))

    if current_price > 0 and "price_change" in allowed:
        price_series = _numeric_column(daily, "price")
        hard_min = max(0.01, current_price * 0.5)
        hard_max = current_price * 1.5
        search_pct = clamp(price_search_pct, 0.05, 0.50)
        if price_series is not None and len(price_series) >= 5:
            hist_min = max(hard_min, float(price_series.quantile(0.05)))
            hist_max = min(hard_max, float(price_series.quantile(0.95)))
        else:
            hist_min = max(hard_min, current_price * (1.0 - search_pct))
            hist_max = min(hard_max, current_price * (1.0 + search_pct))
        if hist_min > hist_max:
            hist_min, hist_max = min(hist_min, hist_max), max(hist_min, hist_max)
        points = 31
        if hist_max > hist_min:
            step = (hist_max - hist_min) / float(points - 1)
            safe_values = [hist_min + i * step for i in range(points)]
        else:
            safe_values = [current_price]
        for target in safe_values:
            if target > 0:
                out.append(_price_candidate(current_price, float(target), objective, current_discount, current_promo, horizon_days, "generated", "safe"))
        near_low = max(hard_min, hist_min * 0.85)
        near_high = min(hard_max, hist_max * 1.15)
        near_values = [near_low, hist_min * 0.90, hist_min * 0.95, hist_max * 1.05, hist_max * 1.10, near_high]
        for target in near_values:
            if target > 0 and not (hist_min <= target <= hist_max):
                out.append(_price_candidate(current_price, float(target), objective, current_discount, current_promo, horizon_days, "generated", "near_range"))

    if "discount_change" in allowed:
        values = {0.0, current_discount, 0.05, 0.10, 0.15, 0.20}
        disc_series = _numeric_column(daily, "discount")
        if disc_series is not None and len(disc_series) >= 5:
            for q in (0.25, 0.50, 0.75):
                values.add(float(disc_series.quantile(q)))
        for target in sorted(clamp(v, 0.0, 0.95) for v in values):
            out.append(_candidate(f"discount_abs_{target:.4f}", "discount_change", f"Скидка {target:.0%}", "generated", current_discount, target, objective, {"manual_price": current_price, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "overrides": {"discount": target, "promotion": current_promo}}, {"horizon_days": horizon_days, "support_zone": "safe"}))

    promo_series = _numeric_column(daily, "promotion")
    has_promo_support = False
    if promo_series is not None and len(promo_series) >= 10:
        positive = int((promo_series > 0).sum()); zero = int((promo_series <= 0).sum())
        has_promo_support = positive >= 5 and zero >= 5
    if has_promo_support and "promotion_change" in allowed:
        for target in [0.0, current_promo, 1.0]:
            out.append(_candidate(f"promotion_{target:.0f}", "promotion_change", f"Промо {target:.0%}", "generated", current_promo, target, objective, {"manual_price": current_price, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "overrides": {"discount": current_discount, "promotion": target}}, {"horizon_days": horizon_days, "support_zone": "safe"}))

    if "demand_shock" in allowed:
        for mult in [0.95, 1.0, 1.05, 1.10]:
            out.append(_candidate(f"demand_{mult:.2f}x", "demand_shock", f"Гипотеза спроса {mult:.0%}", "generated", 1.0, mult, objective, {"manual_price": current_price, "discount_multiplier": 1.0, "demand_multiplier": mult, "overrides": {"discount": current_discount, "promotion": current_promo}}, {"horizon_days": horizon_days, "manual_hypothesis": True, "external_evidence": False}))
    return _dedupe(out)


def run_decision_candidate(trained_bundle: dict, candidate: dict, runner: Callable, scenario_calc_mode: str, price_guardrail_mode: str, horizon_days: int) -> dict:
    params = dict(candidate.get("scenario_params") or {})
    overrides = dict(params.get("overrides") or {})
    runner_bundle = copy.deepcopy(trained_bundle)
    result = runner(
        runner_bundle,
        manual_price=safe_float(params.get("manual_price", candidate.get("target_value")), 0.0),
        freight_multiplier=safe_float(params.get("freight_multiplier", 1.0), 1.0),
        demand_multiplier=safe_float(params.get("demand_multiplier", 1.0), 1.0),
        horizon_days=int(horizon_days),
        discount_multiplier=safe_float(params.get("discount_multiplier", 1.0), 1.0),
        cost_multiplier=safe_float(params.get("cost_multiplier", 1.0), 1.0),
        stock_cap=safe_float(params.get("stock_cap", 0.0), 0.0),
        overrides=overrides,
        factor_overrides=dict(params.get("factor_overrides") or {}),
        scenario_calc_mode=scenario_calc_mode,
        price_guardrail_mode=price_guardrail_mode,
    )
    return result


def _effect(scenario: dict, baseline: dict) -> dict:
    def profit(d):
        for k in ("profit_total_adjusted", "profit_total", "profit_total_raw", "profit"):
            v = finite_or_none((d or {}).get(k))
            if v is not None: return v
        return None
    demand_s = finite_or_none(scenario.get("demand_total")); demand_b = finite_or_none(baseline.get("demand_total"))
    rev_s = finite_or_none(scenario.get("revenue_total")); rev_b = finite_or_none(baseline.get("revenue_total"))
    profit_s = profit(scenario); profit_b = profit(baseline)
    return {"demand_delta_pct": safe_pct_delta(demand_s, demand_b), "revenue_delta_pct": safe_pct_delta(rev_s, rev_b), "profit_delta_pct": safe_pct_delta(profit_s, profit_b)}


def _failed_candidate_evaluation(candidate: dict, baseline_result: dict, exc: Exception) -> dict:
    return {
        "candidate": candidate,
        "scenario_result": {},
        "baseline_result": baseline_result,
        "expected_effect": {},
        "reliability": {
            "score": 0,
            "risk_level": "high",
            "decision_status": "not_recommended",
            "technical_error": True,
            "error_type": type(exc).__name__,
        },
        "status": "not_recommended",
        "warnings": [str(exc)],
        "blockers": ["Candidate calculation failed"],
    }


def evaluate_decision_candidates(results: dict, trained_bundle: dict, candidates: list[dict], runner: Callable, scenario_calc_mode: str, price_guardrail_mode: str, horizon_days: int, objective: str) -> list[dict]:
    baseline = next((c for c in candidates if c.get("action_type") == "baseline"), candidates[0] if candidates else None)
    if baseline is None:
        return []
    try:
        baseline_result = run_decision_candidate(trained_bundle, baseline, runner, scenario_calc_mode, price_guardrail_mode, horizon_days)
    except Exception as exc:
        return [_failed_candidate_evaluation(c, {}, exc) for c in candidates]
    evaluated: List[dict] = []
    for cand in candidates:
        try:
            scenario = baseline_result if cand.get("candidate_id") == baseline.get("candidate_id") else run_decision_candidate(trained_bundle, cand, runner, scenario_calc_mode, price_guardrail_mode, horizon_days)
            rel = evaluate_decision_reliability(results, trained_bundle, cand, scenario, baseline_result, objective=objective, price_guardrail_mode=price_guardrail_mode)
            eff = _effect(scenario, baseline_result)
            eff["conservative_profit_delta_pct"] = rel.get("economic_significance", {}).get("conservative_profit_delta_pct")
            evaluated.append({"candidate": cand, "scenario_result": scenario, "baseline_result": baseline_result, "expected_effect": eff, "reliability": rel, "economic_checks": rel.get("economic_significance", {}), "statistical_support": rel.get("statistical_support", {}), "validation_plan": {}, "status": rel.get("decision_status"), "warnings": rel.get("warnings", []), "blockers": rel.get("blockers", [])})
        except Exception as exc:
            evaluated.append(_failed_candidate_evaluation(cand, baseline_result, exc))
    # Local price refinement: evaluate +/- 1/2/3% around top safe price candidates.
    existing = {((e.get("candidate") or {}).get("action_type"), round(safe_float((e.get("candidate") or {}).get("target_value"), -1), 6)) for e in evaluated}
    price_pool = [e for e in evaluated if (e.get("candidate") or {}).get("action_type") == "price_change" and ((e.get("candidate") or {}).get("metadata") or {}).get("support_zone") == "safe" and not (e.get("blockers") or [])]
    price_pool.sort(key=lambda e: safe_float((e.get("expected_effect") or {}).get("profit_delta_pct"), -9999), reverse=True)
    refinements: List[dict] = []
    ctx = _base_ctx(trained_bundle, None)
    current_price = safe_float(ctx.get("price"), 0.0)
    current_discount = clamp(ctx.get("discount", 0.0), 0.0, 0.95)
    current_promo = clamp(ctx.get("promotion", ctx.get("promo", 0.0)), 0.0, 1.0)
    for e in price_pool[:3]:
        center = safe_float((e.get("candidate") or {}).get("target_value"), 0.0)
        for pct in (-3, -2, -1, 1, 2, 3):
            target = center * (1.0 + pct / 100.0)
            key = ("price_change", round(target, 6))
            if target > 0 and key not in existing:
                existing.add(key)
                refinements.append(_price_candidate(current_price, target, objective, current_discount, current_promo, horizon_days, "refined", "safe"))
    for cand in refinements:
        try:
            scenario = run_decision_candidate(trained_bundle, cand, runner, scenario_calc_mode, price_guardrail_mode, horizon_days)
            rel = evaluate_decision_reliability(results, trained_bundle, cand, scenario, baseline_result, objective=objective, price_guardrail_mode=price_guardrail_mode)
            eff = _effect(scenario, baseline_result)
            eff["conservative_profit_delta_pct"] = rel.get("economic_significance", {}).get("conservative_profit_delta_pct")
            evaluated.append({"candidate": cand, "scenario_result": scenario, "baseline_result": baseline_result, "expected_effect": eff, "reliability": rel, "economic_checks": rel.get("economic_significance", {}), "statistical_support": rel.get("statistical_support", {}), "validation_plan": {}, "status": rel.get("decision_status"), "warnings": rel.get("warnings", []), "blockers": rel.get("blockers", [])})
        except Exception as exc:
            evaluated.append(_failed_candidate_evaluation(cand, baseline_result, exc))
    return evaluated
