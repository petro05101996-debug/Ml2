from __future__ import annotations

from typing import Any, Dict, List


def build_baseline_confidence_state(rolling_summary: Dict[str, Any]) -> Dict[str, Any]:
    n = int(rolling_summary.get("n_valid_windows", 0) or 0)
    w = rolling_summary.get("median_wape")
    b = rolling_summary.get("median_bias_pct")
    sr = rolling_summary.get("median_sum_ratio")
    issues: List[str] = []
    if n >= 2 and w is not None and w <= 25 and abs(b) <= 0.05 and 0.95 <= sr <= 1.05:
        lvl = "high"
    elif n >= 1 and w is not None and w <= 35 and abs(b) <= 0.10:
        lvl = "medium"
    else:
        lvl = "low"
        issues.append("baseline_backtest_weak")
    return {"level": lvl, "metrics": rolling_summary, "issues": issues}


def build_factor_confidence_state(factor_backtest_summary, ood_flags) -> Dict[str, Any]:
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
    n_var = int(factor_backtest_summary.get("n_variative_controllable_features", 0) or 0)

    if pss >= 0.75 and ood_share <= 0.10 and n_windows >= 1 and n_train >= 60:
        lvl = "high"
    elif pss >= 0.55 and n_windows >= 1:
        lvl = "medium"
    else:
        lvl = "low"

    if n_windows < 1:
        issues.append("factor_backtest_missing")
        lvl = "low"
    if n_train < 60:
        issues.append("factor_train_too_small")
        lvl = "low"
    if price_unique < 4 and n_var <= 1:
        issues.append("factor_variation_weak")
        lvl = "low"
    if ood_flags:
        issues.append("factor_ood_detected")
        if lvl == "high":
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


def combine_confidence_states(baseline_state, factor_state, shock_state, intervals_available: bool = False) -> Dict[str, Any]:
    order = {"low": 0, "medium": 1, "high": 2}
    inv = {0: "low", 1: "medium", 2: "high"}
    overall = inv[min(order.get(baseline_state["level"], 0), order.get(factor_state["level"], 0), order.get(shock_state["level"], 0))]
    issues = baseline_state.get("issues", []) + factor_state.get("issues", []) + shock_state.get("issues", [])
    return {
        "baseline_confidence": baseline_state,
        "factor_confidence": factor_state,
        "shock_confidence": shock_state,
        "overall_confidence": overall,
        "issues": issues,
        "intervals_available": bool(intervals_available),
    }
