from __future__ import annotations

import copy
import time
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






def _support_metadata(horizon_days: int, supported: bool, reason: str = "") -> dict:
    meta = {
        "horizon_days": horizon_days,
        "support_zone": "safe" if supported else "unsupported_hypothesis",
        "extrapolated": not supported,
    }
    if not supported:
        meta["unsupported_hypothesis"] = True
        meta["blockers"] = [reason or "factor_support_missing"]
        meta["warnings"] = ["factor_support_missing"]
    return meta

def _model_metadata(trained_bundle: dict, action_type: str, scenario_calc_mode: str | None = None) -> dict:
    bundle = trained_bundle or {}
    scenario_calc_mode = str(scenario_calc_mode or bundle.get("analysis_scenario_calc_mode", ""))

    baseline_bundle = bundle.get("baseline_bundle", {}) or {}
    catboost_bundle = bundle.get("catboost_full_factor_bundle", {}) or {}

    if scenario_calc_mode == "catboost_full_factors":
        model_features = list(catboost_bundle.get("feature_cols", []) or [])
        effect_source = "catboost_full_factor_reprediction"
    else:
        model_features = list(baseline_bundle.get("features", []) or [])
        effect_source = "baseline_plus_scenario_layer"

    scenario_layer_features = ["price", "discount", "promotion", "freight_value", "demand_shock"]

    return {
        "scenario_calc_mode": scenario_calc_mode,
        "model_features": model_features,
        "effect_source": effect_source,
        "scenario_layer_features": scenario_layer_features,
        "action_type": action_type,
    }




def _enrich_candidate_metadata(candidate: dict, trained_bundle: dict, scenario_calc_mode: str) -> dict:
    out = copy.deepcopy(candidate)
    meta = dict(out.get("metadata") or {})
    model_meta = _model_metadata(
        trained_bundle,
        str(out.get("action_type", "")),
        scenario_calc_mode=scenario_calc_mode,
    )
    meta.update(model_meta)
    out["metadata"] = meta
    return out


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
        key = (c.get("candidate_id") if c.get("action_type") == "combined_change" else c.get("action_type"), round(safe_float(c.get("target_value"), -999999), 6), c.get("source"))
        if key not in seen:
            seen.add(key); deduped.append(c)
    return deduped



def _combined_candidate(
    cid: str,
    title: str,
    current_price: float,
    target_price: float,
    current_discount: float,
    target_discount: float,
    current_promo: float,
    target_promo: float,
    current_freight: float,
    target_freight: float,
    objective: str,
    horizon_days: int,
    factor_overrides: Optional[dict] = None,
) -> dict:
    return _candidate(
        cid,
        "combined_change",
        title,
        "generated_combined",
        1.0,
        1.0,
        objective,
        {
            "manual_price": target_price,
            "freight_multiplier": 1.0,
            "discount_multiplier": 1.0,
            "demand_multiplier": 1.0,
            "factor_overrides": dict(factor_overrides or {}),
            "overrides": {"discount": target_discount, "promotion": target_promo, "freight_value": target_freight},
        },
        {
            "horizon_days": horizon_days,
            "support_zone": "safe",
            "combined_actions": {
                "price_change_pct": safe_pct_delta(target_price, current_price),
                "discount_delta_pp": (target_discount - current_discount) * 100.0,
                "promotion_delta_pp": (target_promo - current_promo) * 100.0,
                "freight_change_pct": safe_pct_delta(target_freight, current_freight),
            },
        },
    )


def _first_positive_cost(*values) -> float | None:
    for value in values:
        parsed = finite_or_none(value)
        if parsed is not None and parsed > 0:
            return float(parsed)
    return None


def _scenario_cost(scenario: dict, context: dict | None = None) -> float | None:
    effective = (scenario or {}).get("effective_scenario") or {}
    daily = (scenario or {}).get("daily")
    daily_cost = None
    if hasattr(daily, "columns") and "cost" in daily.columns:
        try:
            daily_cost = float(daily["cost"].dropna().astype(float).median())
        except Exception:
            daily_cost = None
    return _first_positive_cost(
        (scenario or {}).get("scenario_cost"),
        effective.get("scenario_cost"),
        effective.get("cost"),
        daily_cost,
        (context or {}).get("cost"),
    )


def _violates_constraints(candidate: dict, scenario: dict, baseline: dict, constraints: dict | None, context: dict | None = None) -> List[str]:
    constraints = constraints or {}
    if not constraints:
        return []
    issues: List[str] = []
    eff = _effect(scenario, baseline)
    params = (candidate.get("scenario_params") or {})
    price = safe_float(params.get("manual_price", candidate.get("target_value")), 0.0)
    revenue = safe_float(scenario.get("revenue_total"), 0.0)
    profit = safe_float(scenario.get("profit_total_adjusted", scenario.get("profit_total", 0.0)), 0.0)
    margin_pct = profit / max(revenue, 1e-9) * 100.0 if revenue > 0 else -999.0
    current_price = safe_float(candidate.get("current_value"), price) if candidate.get("action_type") == "price_change" else safe_float((baseline or {}).get("applied_price_gross", price), price)
    max_price_change = constraints.get("max_price_change_pct")
    if max_price_change is not None and abs(safe_pct_delta(price, current_price)) > safe_float(max_price_change, 999):
        issues.append("max_price_change_pct")
    if constraints.get("forbid_negative_profit", False) and profit < 0:
        issues.append("forbid_negative_profit")
    if constraints.get("min_margin_pct") is not None and margin_pct < safe_float(constraints.get("min_margin_pct"), -999):
        issues.append("min_margin_pct")
    if constraints.get("max_demand_drop_pct") is not None and eff.get("demand_delta_pct", 0.0) < -abs(safe_float(constraints.get("max_demand_drop_pct"), 999)):
        issues.append("max_demand_drop_pct")
    if constraints.get("max_revenue_drop_pct") is not None and eff.get("revenue_delta_pct", 0.0) < -abs(safe_float(constraints.get("max_revenue_drop_pct"), 999)):
        issues.append("max_revenue_drop_pct")
    cost = _scenario_cost(scenario, context) or _scenario_cost(baseline, context)
    if constraints.get("forbid_price_below_cost", False):
        if cost is None:
            issues.append("cost_missing_blocks_profit_recommendation")
        elif price < cost:
            issues.append("forbid_price_below_cost")
    if constraints.get("require_cost_for_profit", False) and str(candidate.get("objective", "profit")) == "profit":
        cost_source = str((scenario or {}).get("cost_source") or (baseline or {}).get("cost_source") or (context or {}).get("cost_source") or "")
        profit_reliable = (scenario or {}).get("profit_is_reliable")
        if cost is None:
            issues.append("cost_missing_blocks_profit_recommendation")
        if bool((scenario or {}).get("cost_proxied") or (baseline or {}).get("cost_proxied") or (scenario or {}).get("cost_is_proxy") or (baseline or {}).get("cost_is_proxy")):
            issues.append("require_cost_for_profit_proxy_cost")
        if cost_source and cost_source != "provided":
            issues.append("require_cost_for_profit_non_provided_cost")
        if profit_reliable is False:
            issues.append("require_cost_for_profit_unreliable_profit")
    min_uplift = constraints.get("min_expected_profit_uplift_pct")
    if candidate.get("action_type") != "baseline" and min_uplift is not None and eff.get("profit_delta_pct", 0.0) < safe_float(min_uplift, 0.0):
        issues.append("min_expected_profit_uplift_pct")
    return issues

def _freight_candidate(
    current_price: float,
    current_discount: float,
    current_promo: float,
    current_freight: float,
    target_freight: float,
    objective: str,
    horizon_days: int,
    source: str,
    support_zone: str,
    factor_overrides: Optional[dict] = None,
) -> dict:
    return _candidate(
        f"freight_{support_zone}_{target_freight:.4f}",
        "freight_change",
        f"Логистика {target_freight:.2f} ({safe_pct_delta(target_freight, current_freight):+.1f}%)",
        source,
        current_freight,
        target_freight,
        objective,
        {
            "manual_price": current_price,
            "freight_multiplier": 1.0,
            "discount_multiplier": 1.0,
            "demand_multiplier": 1.0,
            "factor_overrides": dict(factor_overrides or {}),
            "overrides": {
                "discount": current_discount,
                "promotion": current_promo,
                "freight_value": target_freight,
            },
        },
        {
            "horizon_days": horizon_days,
            "support_zone": support_zone,
            "extrapolated": support_zone != "safe",
        },
    )


def _price_candidate(current_price: float, target: float, objective: str, current_discount: float, current_promo: float, current_freight: float, horizon_days: int, source: str, support_zone: str, factor_overrides: Optional[dict] = None) -> dict:
    pct = safe_pct_delta(target, current_price)
    return _candidate(
        f"price_{support_zone}_{target:.4f}",
        "price_change",
        f"Цена {target:.2f} ({pct:+.1f}%)",
        source,
        current_price,
        target,
        objective,
        {"manual_price": target, "freight_multiplier": 1.0, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "factor_overrides": dict(factor_overrides or {}), "overrides": {"discount": current_discount, "promotion": current_promo, "freight_value": current_freight}},
        {"horizon_days": horizon_days, "support_zone": support_zone, "extrapolated": support_zone != "safe"},
    )


def generate_decision_candidates(trained_bundle: dict, current_context: dict | None, objective: str = "profit", allowed_actions: list[str] | None = None, horizon_days: int = 30, price_search_pct: float = 0.20) -> list[dict]:
    ctx = _base_ctx(trained_bundle, current_context)
    current_price = safe_float(ctx.get("price"), 0.0)
    current_discount = clamp(ctx.get("discount", 0.0), 0.0, 0.95)
    current_promo = clamp(ctx.get("promotion", ctx.get("promo", 0.0)), 0.0, 1.0)
    current_freight = safe_float(ctx.get("freight_value"), 0.0)
    factor_overrides = dict(ctx.get("factor_overrides") or {})
    allowed = set(allowed_actions or ["price_change", "discount_change", "promotion_change", "freight_change", "demand_shock"])
    daily = (trained_bundle or {}).get("daily_base")
    out: List[dict] = []
    out.append(_candidate("baseline_current", "baseline", "Текущий вариант", "baseline", current_price, current_price, objective, {"manual_price": current_price, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "freight_multiplier": 1.0, "factor_overrides": dict(factor_overrides), "overrides": {"discount": current_discount, "promotion": current_promo, "freight_value": current_freight}}, {"horizon_days": horizon_days, "support_zone": "baseline"}))

    price_supported = False
    discount_supported = False
    promotion_supported = False
    freight_supported = False

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
                out.append(_price_candidate(current_price, float(target), objective, current_discount, current_promo, current_freight, horizon_days, "generated", "safe", factor_overrides=factor_overrides))
        near_low = max(hard_min, hist_min * 0.85)
        near_high = min(hard_max, hist_max * 1.15)
        near_values = [near_low, hist_min * 0.90, hist_min * 0.95, hist_max * 1.05, hist_max * 1.10, near_high]
        for target in near_values:
            if target > 0 and not (hist_min <= target <= hist_max):
                out.append(_price_candidate(current_price, float(target), objective, current_discount, current_promo, current_freight, horizon_days, "generated", "near_range", factor_overrides=factor_overrides))

    price_supported = price_supported or any(c.get("action_type") == "price_change" for c in out)

    if "discount_change" in allowed:
        values = {0.0, current_discount, 0.05, 0.10, 0.15, 0.20}
        disc_series = _numeric_column(daily, "discount")
        if disc_series is not None and len(disc_series) >= 5:
            discount_supported = int(disc_series.dropna().round(4).nunique()) >= 3 and float(disc_series.max() - disc_series.min()) > 0.01
            for q in (0.25, 0.50, 0.75):
                values.add(float(disc_series.quantile(q)))
        else:
            discount_supported = False
        for target in sorted(clamp(v, 0.0, 0.95) for v in values):
            out.append(_candidate(
                f"discount_abs_{target:.4f}",
                "discount_change",
                f"Скидка {target:.0%}",
                "generated" if discount_supported else "unsupported_hypothesis",
                current_discount,
                target,
                objective,
                {"manual_price": current_price, "freight_multiplier": 1.0, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "factor_overrides": dict(factor_overrides), "overrides": {"discount": target, "promotion": current_promo, "freight_value": current_freight}},
                _support_metadata(horizon_days, discount_supported, "factor_support_missing:discount"),
            ))

    promo_series = _numeric_column(daily, "promotion")
    has_promo_support = False
    if promo_series is not None and len(promo_series) >= 10:
        positive = int((promo_series > 0).sum()); zero = int((promo_series <= 0).sum())
        has_promo_support = positive >= 5 and zero >= 5
    promotion_supported = bool(has_promo_support and "promotion_change" in allowed)
    if promotion_supported:
        for target in [0.0, current_promo, 1.0]:
            out.append(_candidate(f"promotion_{target:.0f}", "promotion_change", f"Промо {target:.0%}", "generated", current_promo, target, objective, {"manual_price": current_price, "freight_multiplier": 1.0, "discount_multiplier": 1.0, "demand_multiplier": 1.0, "factor_overrides": dict(factor_overrides), "overrides": {"discount": current_discount, "promotion": target, "freight_value": current_freight}}, {"horizon_days": horizon_days, "support_zone": "safe"}))

    if "freight_change" in allowed and current_freight > 0:
        freight_series = _numeric_column(daily, "freight_value")
        values = {
            current_freight,
            current_freight * 0.90,
            current_freight * 0.95,
            current_freight * 1.05,
            current_freight * 1.10,
        }

        if freight_series is not None and len(freight_series.dropna()) >= 5:
            freight_supported = int(freight_series.dropna().round(4).nunique()) >= 3
            for q in (0.10, 0.25, 0.50, 0.75, 0.90):
                values.add(float(freight_series.quantile(q)))
        else:
            freight_supported = False

        for target in sorted(v for v in values if v >= 0):
            cand = _freight_candidate(
                current_price=current_price,
                current_discount=current_discount,
                current_promo=current_promo,
                current_freight=current_freight,
                target_freight=float(target),
                objective=objective,
                horizon_days=horizon_days,
                source="generated" if freight_supported else "unsupported_hypothesis",
                support_zone="safe" if freight_supported else "unsupported_hypothesis",
                factor_overrides=factor_overrides,
            )
            if not freight_supported:
                meta = dict(cand.get("metadata") or {})
                meta.update(_support_metadata(horizon_days, False, "factor_support_missing:freight_value"))
                cand["metadata"] = meta
            out.append(cand)

    if "demand_shock" in allowed:
        for mult in [0.95, 1.0, 1.05, 1.10]:
            out.append(_candidate(f"demand_{mult:.2f}x", "demand_shock", f"Гипотеза спроса {mult:.0%}", "generated", 1.0, mult, objective, {"manual_price": current_price, "freight_multiplier": 1.0, "discount_multiplier": 1.0, "demand_multiplier": mult, "factor_overrides": dict(factor_overrides), "overrides": {"discount": current_discount, "promotion": current_promo, "freight_value": current_freight}}, {"horizon_days": horizon_days, "manual_hypothesis": True, "external_evidence": False}))

    price_targets = [current_price * 0.95, current_price * 1.05]
    discount_targets = [clamp(current_discount + 0.05, 0.0, 0.95), clamp(current_discount - 0.05, 0.0, 0.95)]
    promo_targets = [1.0 if current_promo < 0.5 else 0.0]
    freight_targets = [max(0.0, current_freight * 0.95)]
    combos = []
    if price_supported and promotion_supported:
        combos.extend([
            (price_targets[0], current_discount, promo_targets[0], current_freight, "Цена + промо"),
            (price_targets[1], current_discount, promo_targets[0], current_freight, "Цена + промо"),
        ])
    if price_supported and discount_supported:
        combos.extend([
            (price_targets[0], discount_targets[0], current_promo, current_freight, "Цена + скидка"),
            (price_targets[1], discount_targets[1], current_promo, current_freight, "Цена + скидка"),
        ])
    if price_supported and freight_supported:
        combos.append((price_targets[0], current_discount, current_promo, freight_targets[0], "Цена + логистика"))
    if discount_supported and promotion_supported:
        combos.append((current_price, discount_targets[0], promo_targets[0], current_freight, "Скидка + промо"))
    for idx, (p_t, d_t, promo_t, freight_t, title) in enumerate(combos[:10]):
        out.append(_combined_candidate(f"combined_{idx}", title, current_price, p_t, current_discount, d_t, current_promo, promo_t, current_freight, freight_t, objective, horizon_days, factor_overrides=factor_overrides))
    for c in out:
        meta = dict(c.get("metadata") or {})
        meta.update(_model_metadata(trained_bundle, str(c.get("action_type", ""))))
        c["metadata"] = meta
    return _dedupe(out)


def run_decision_candidate(trained_bundle: dict, candidate: dict, runner: Callable, scenario_calc_mode: str, price_guardrail_mode: str, horizon_days: int) -> dict:
    params = dict(candidate.get("scenario_params") or {})
    overrides = dict(params.get("overrides") or {})
    runner_bundle = dict(trained_bundle)
    if isinstance(runner_bundle.get("base_ctx"), dict):
        runner_bundle["base_ctx"] = dict(runner_bundle["base_ctx"])
    if isinstance(runner_bundle.get("latest_row"), dict):
        runner_bundle["latest_row"] = dict(runner_bundle["latest_row"])
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




def _apply_model_quality_gate(rel: dict, trained_bundle: dict, results: dict) -> dict:
    gate = dict((trained_bundle or {}).get("model_quality_gate") or (results or {}).get("model_quality_gate") or {})
    status = str(gate.get("status", "")).strip()
    if not status:
        return rel
    out = dict(rel or {})
    warnings = list(out.get("warnings", []) or [])
    blockers = list(out.get("blockers", []) or [])
    warnings.extend([f"Model quality gate: {w}" for w in gate.get("warnings", []) or []])
    if status == "blocked":
        out["decision_status"] = "not_recommended"
        blockers.extend([f"Model quality gate: {b}" for b in gate.get("blockers", []) or ["blocked"]])
    elif status == "experimental_only" and out.get("decision_status") in {"recommended", "test_recommended", "controlled_test_only"}:
        out["decision_status"] = "experimental_only"
        warnings.append("Model quality gate limits decision to experimental_only.")
    elif status == "controlled_test_only" and out.get("decision_status") == "recommended":
        out["decision_status"] = "test_recommended"
        warnings.append("Model quality gate limits decision to controlled_test_only.")
    out["model_quality_gate"] = gate
    out["warnings"] = list(dict.fromkeys(str(x) for x in warnings if x))
    out["blockers"] = list(dict.fromkeys(str(x) for x in blockers if x))
    return out

def evaluate_decision_candidates(
    results: dict,
    trained_bundle: dict,
    candidates: list[dict],
    runner: Callable,
    scenario_calc_mode: str,
    price_guardrail_mode: str,
    horizon_days: int,
    objective: str,
    current_context: dict | None = None,
    constraints: dict | None = None,
    max_candidates: int = 24,
    max_refinements: int = 12,
    timeout_sec: float = 20.0,
) -> list[dict]:
    deadline = time.monotonic() + float(timeout_sec)
    enriched_candidates = [
        _enrich_candidate_metadata(c, trained_bundle, scenario_calc_mode)
        for c in (candidates or [])
    ]
    enriched_candidates = enriched_candidates[:int(max_candidates)]
    baseline = next((c for c in enriched_candidates if c.get("action_type") == "baseline"), enriched_candidates[0] if enriched_candidates else None)
    if baseline is None:
        return []
    try:
        baseline_result = run_decision_candidate(trained_bundle, baseline, runner, scenario_calc_mode, price_guardrail_mode, horizon_days)
    except Exception as exc:
        return [_failed_candidate_evaluation(c, {}, exc) for c in enriched_candidates]
    evaluated: List[dict] = []

    def _append_timeout_cutoff() -> None:
        evaluated.append({
            "candidate": {"candidate_id": "timeout_cutoff", "action_type": "technical"},
            "scenario_result": {},
            "baseline_result": baseline_result,
            "expected_effect": {},
            "reliability": {
                "score": 0,
                "risk_level": "high",
                "decision_status": "not_recommended",
                "technical_error": True,
            },
            "status": "not_recommended",
            "warnings": ["Decision candidate evaluation stopped by timeout."],
            "blockers": ["decision_evaluation_timeout"],
        })

    for cand in enriched_candidates:
        if time.monotonic() > deadline:
            _append_timeout_cutoff()
            break
        try:
            scenario = baseline_result if cand.get("candidate_id") == baseline.get("candidate_id") else run_decision_candidate(trained_bundle, cand, runner, scenario_calc_mode, price_guardrail_mode, horizon_days)
            rel = evaluate_decision_reliability(results, trained_bundle, cand, scenario, baseline_result, objective=objective, price_guardrail_mode=price_guardrail_mode)
            rel = _apply_model_quality_gate(rel, trained_bundle, results)
            candidate_blockers = list(cand.get("blockers") or (cand.get("metadata") or {}).get("blockers") or [])
            constraint_issues = _violates_constraints(cand, scenario, baseline_result, constraints or (current_context or {}).get("constraints"), _base_ctx(trained_bundle, current_context))
            if candidate_blockers or constraint_issues:
                rel = dict(rel)
                rel["decision_status"] = "not_recommended"
                rel["blockers"] = list(rel.get("blockers", [])) + [f"Candidate invalid: {x}" for x in candidate_blockers] + [f"Business constraint violated: {x}" for x in constraint_issues]
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
    ctx = _base_ctx(trained_bundle, current_context)
    current_price = safe_float(ctx.get("price"), 0.0)
    current_discount = clamp(ctx.get("discount", 0.0), 0.0, 0.95)
    current_promo = clamp(ctx.get("promotion", ctx.get("promo", 0.0)), 0.0, 1.0)
    factor_overrides = dict(ctx.get("factor_overrides") or {})
    for e in price_pool[:3]:
        center = safe_float((e.get("candidate") or {}).get("target_value"), 0.0)
        for pct in (-3, -2, -1, 1, 2, 3):
            target = center * (1.0 + pct / 100.0)
            key = ("price_change", round(target, 6))
            if target > 0 and key not in existing:
                existing.add(key)
                refinements.append(_price_candidate(current_price, target, objective, current_discount, current_promo, safe_float(ctx.get("freight_value"), 0.0), horizon_days, "refined", "safe", factor_overrides=factor_overrides))
    for cand in refinements[:int(max_refinements)]:
        if time.monotonic() > deadline:
            _append_timeout_cutoff()
            break
        cand = _enrich_candidate_metadata(cand, trained_bundle, scenario_calc_mode)
        try:
            scenario = run_decision_candidate(trained_bundle, cand, runner, scenario_calc_mode, price_guardrail_mode, horizon_days)
            rel = evaluate_decision_reliability(results, trained_bundle, cand, scenario, baseline_result, objective=objective, price_guardrail_mode=price_guardrail_mode)
            rel = _apply_model_quality_gate(rel, trained_bundle, results)
            candidate_blockers = list(cand.get("blockers") or (cand.get("metadata") or {}).get("blockers") or [])
            constraint_issues = _violates_constraints(cand, scenario, baseline_result, constraints or (current_context or {}).get("constraints"), _base_ctx(trained_bundle, current_context))
            if candidate_blockers or constraint_issues:
                rel = dict(rel)
                rel["decision_status"] = "not_recommended"
                rel["blockers"] = list(rel.get("blockers", [])) + [f"Candidate invalid: {x}" for x in candidate_blockers] + [f"Business constraint violated: {x}" for x in constraint_issues]
            eff = _effect(scenario, baseline_result)
            eff["conservative_profit_delta_pct"] = rel.get("economic_significance", {}).get("conservative_profit_delta_pct")
            evaluated.append({"candidate": cand, "scenario_result": scenario, "baseline_result": baseline_result, "expected_effect": eff, "reliability": rel, "economic_checks": rel.get("economic_significance", {}), "statistical_support": rel.get("statistical_support", {}), "validation_plan": {}, "status": rel.get("decision_status"), "warnings": rel.get("warnings", []), "blockers": rel.get("blockers", [])})
        except Exception as exc:
            evaluated.append(_failed_candidate_evaluation(cand, baseline_result, exc))
    return evaluated
