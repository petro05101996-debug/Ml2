from __future__ import annotations

from typing import Any, Dict, List


def build_baseline_confidence_state(rolling_summary: Dict[str, Any]) -> Dict[str, Any]:
    n = int(rolling_summary.get("n_valid_windows", 0) or 0)
    w = rolling_summary.get("median_wape")
    b = rolling_summary.get("median_bias_pct")
    sr = rolling_summary.get("median_sum_ratio")
    flat = float(rolling_summary.get("flat_window_share", 1.0) or 1.0)
    std_ratio = float(rolling_summary.get("median_std_ratio", 0.0) or 0.0)
    strategy = str(rolling_summary.get("strategy", ""))
    issues: List[str] = []
    if n >= 3 and w is not None and w <= 20 and abs(b) <= 0.05 and 0.95 <= sr <= 1.05 and flat <= 0.25 and std_ratio >= 0.60:
        lvl = "high"
    elif n >= 2 and w is not None and w <= 28 and abs(b) <= 0.08 and 0.90 <= sr <= 1.10 and flat <= 0.50:
        lvl = "medium"
    else:
        lvl = "low"
        issues.append("baseline_backtest_weak")
    if strategy in {"median7", "mean28"} and lvl == "high":
        lvl = "medium"
        issues.append("baseline_strategy_simple_cap")
    if flat > 0.5:
        issues.append("baseline_flat_forecast")
    return {"level": lvl, "metrics": rolling_summary, "issues": issues}


def build_factor_confidence_state(factor_backtest_summary, ood_flags, baseline_level: str = "low", scenario_outside_factor_backtest_range: bool = False) -> Dict[str, Any]:
    issues = []
    if not factor_backtest_summary.get("trained", False):
        issues = ["factor_model_unavailable"]
        if ood_flags:
            issues.append("factor_ood_detected")
        return {"level": "low", "metrics": factor_backtest_summary, "issues": issues}

    pss = float(factor_backtest_summary.get("price_sign_stability", 0.0) or 0.0)
    ood_share = float(factor_backtest_summary.get("factor_ood_share", 0.0) or 0.0)
    n_train = int(factor_backtest_summary.get("n_train_rows", 0) or 0)
    n_windows = int(factor_backtest_summary.get("n_valid_windows", 0) or 0)
    price_unique = int(factor_backtest_summary.get("price_unique_count", 0) or 0)
    n_var = int(factor_backtest_summary.get("variative_controllable_count", factor_backtest_summary.get("n_variative_controllable_features", 0)) or 0)
    train_scope = str(factor_backtest_summary.get("train_scope", "none"))

    if pss >= 0.75 and ood_share <= 0.10 and n_windows >= 1 and n_train >= 80 and price_unique >= 4:
        lvl = "high"
    elif pss >= 0.55 and n_windows >= 1:
        lvl = "medium"
    else:
        lvl = "low"

    if n_windows < 1:
        issues.append("factor_backtest_missing")
        lvl = "low"
    if n_train < 45:
        issues.append("factor_train_too_small")
        lvl = "low"
    if price_unique < 3 and n_var < 2:
        issues.append("factor_variation_weak")
        lvl = "low"
    if n_train < 80 or price_unique < 4:
        if lvl == "high":
            lvl = "medium"
        issues.append("factor_data_limited")
    if train_scope == "pooled":
        if lvl == "high":
            lvl = "medium"
        issues.append("factor_trained_on_pooled_scope")
    if ood_flags:
        issues.append("factor_ood_detected")
        if lvl == "high":
            lvl = "medium"
    if scenario_outside_factor_backtest_range:
        issues.append("scenario_outside_factor_backtest_range")
        if lvl == "high":
            lvl = "medium"
    if baseline_level == "low":
        lvl = "low"
    elif baseline_level == "medium" and lvl == "high":
        lvl = "medium"
    return {"level": lvl, "metrics": factor_backtest_summary, "issues": issues}


def build_shock_confidence_state(shocks) -> Dict[str, Any]:
    if not shocks:
        return {"level": "high", "metrics": {"n_shocks": 0}, "issues": []}
    lvl = "low"
    if all(str(s.get("confidence", "low")) == "medium" and float(s.get("intensity", 0.0)) <= 0.20 for s in shocks):
        lvl = "medium"
    if any(str(s.get("confidence", "low")) == "high" and float(s.get("intensity", 0.0)) <= 0.10 for s in shocks):
        lvl = "medium"
    return {"level": lvl, "metrics": {"n_shocks": len(shocks)}, "issues": []}


def combine_confidence_states(
    baseline_state,
    factor_state,
    shock_state,
    intervals_available: bool = False,
    *,
    factor_role: str | None = None,
    scenario_outside_factor_backtest_range: bool = False,
    scenario_equals_current_but_delta_nonzero: bool = False,
    explainability_available: bool = True,
) -> Dict[str, Any]:
    order = {"low": 0, "medium": 1, "high": 2}
    inv = {0: "low", 1: "medium", 2: "high"}
    overall = inv[min(order.get(baseline_state["level"], 0), order.get(factor_state["level"], 0), order.get(shock_state["level"], 0))]
    issues = baseline_state.get("issues", []) + factor_state.get("issues", []) + shock_state.get("issues", [])
    if factor_role == "advisory_only":
        issues.append("factor_advisory_only")
    if scenario_outside_factor_backtest_range:
        issues.append("scenario_outside_factor_backtest_range")
    if scenario_equals_current_but_delta_nonzero:
        issues.append("scenario_equals_current_but_delta_nonzero")
    if not explainability_available:
        issues.append("explainability_unavailable")
    return {
        "baseline_confidence": baseline_state,
        "factor_confidence": factor_state,
        "shock_confidence": shock_state,
        "overall_confidence": overall,
        "issues": issues,
        "intervals_available": bool(intervals_available),
    }
