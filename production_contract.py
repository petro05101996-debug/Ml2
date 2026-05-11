from __future__ import annotations

from typing import Any, Dict, List


HARD_DATA_BLOCKERS = {
    "ambiguous_sales_mapping_requires_confirmation",
    "quantity_missing_not_inferable",
    "price_revenue_quantity_discount_mismatch_gt_15pct",
}

TEST_ONLY_WARNINGS = {
    "cost_proxy",
    "stock_missing",
    "short_history",
    "low_price_variability",
}


def resolve_data_quality_gate(quality: Dict[str, Any]) -> Dict[str, Any]:
    quality = quality or {}

    errors: List[str] = [str(x) for x in quality.get("errors", []) or []]
    blockers: List[str] = [str(x) for x in quality.get("blockers", []) or []]
    warnings: List[str] = [str(x) for x in quality.get("warnings", []) or []]

    cost_is_proxy = bool(quality.get("cost_is_proxy", False))
    cost_source = str(quality.get("cost_source", "unknown"))
    target_semantics = quality.get("target_semantics", {}) or {}
    demand_censoring_risk = str(target_semantics.get("demand_censoring_risk", "unknown"))

    hard_blockers = [b for b in blockers if b in HARD_DATA_BLOCKERS]

    if errors:
        status = "blocked"
        recommendation_status = "not_recommended"
        can_run_calculation = False
        can_show_forecast = False
        can_show_what_if = False
        can_recommend_action = False

    elif hard_blockers:
        status = "diagnostic_only"
        recommendation_status = "not_recommended"
        can_run_calculation = False
        can_show_forecast = False
        can_show_what_if = False
        can_recommend_action = False

    elif cost_is_proxy or cost_source != "provided":
        status = "test_only"
        recommendation_status = "test_recommended"
        can_run_calculation = True
        can_show_forecast = True
        can_show_what_if = True
        can_recommend_action = False
        warnings.append("cost is not provided; profit recommendations are limited to test-only.")

    elif demand_censoring_risk in {"medium", "medium_high", "high", "unknown"}:
        status = "test_only"
        recommendation_status = "test_recommended"
        can_run_calculation = True
        can_show_forecast = True
        can_show_what_if = True
        can_recommend_action = False
        warnings.append("Demand may be censored by stockouts or unknown stock; recommendations require controlled test.")

    else:
        status = "production_candidate"
        recommendation_status = "recommended"
        can_run_calculation = True
        can_show_forecast = True
        can_show_what_if = True
        can_recommend_action = True

    return {
        "status": status,
        "recommendation_status": recommendation_status,
        "errors": errors,
        "blockers": blockers,
        "hard_blockers": hard_blockers,
        "warnings": list(dict.fromkeys(warnings)),
        "usage_policy": {
            "can_run_calculation": can_run_calculation,
            "can_show_forecast": can_show_forecast,
            "can_show_what_if": can_show_what_if,
            "can_recommend_action": can_recommend_action,
        },
    }
