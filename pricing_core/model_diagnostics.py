from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def build_model_diagnostics(
    backtest_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    ood_flags: List[str],
    data_sufficiency: Dict[str, Any],
    behavior_checks: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    issues: List[str] = []
    if backtest_df.empty:
        return {"overall_confidence": "low", "score": 0.2, "issues": ["backtest_unavailable"], "ood_flags": ood_flags, "data_sufficiency": data_sufficiency}

    med_wape = float(pd.to_numeric(backtest_df.get("wape", np.nan), errors="coerce").median())
    med_bias = float(pd.to_numeric(backtest_df.get("bias_pct", np.nan), errors="coerce").abs().median())
    med_std_ratio = float(pd.to_numeric(backtest_df.get("std_ratio", np.nan), errors="coerce").median())

    score = 1.0
    score -= min(max(med_wape - 0.2, 0.0), 0.6)
    score -= min(max(med_bias - 0.05, 0.0), 0.2)
    score -= 0.15 if med_std_ratio < 0.6 else 0.0
    score -= 0.1 if len(ood_flags) > 0 else 0.0
    score -= 0.1 if int(data_sufficiency.get("n_weeks", 0)) < 26 else 0.0
    checks = behavior_checks or {}
    score -= 0.15 if checks.get("manual_scenario_fallback_applied", False) else 0.0
    score = float(np.clip(score, 0.0, 1.0))

    if med_std_ratio < 0.6:
        issues.append("flat_forecast_risk")
    if med_bias > 0.10:
        issues.append("bias_high")
    if med_wape > 0.35:
        issues.append("backtest_weak")
    if ood_flags:
        issues.append("scenario_out_of_training_range")
    if checks.get("scenario_changed_but_forecast_unchanged", False):
        issues.append("scenario_ineffective")
    if checks.get("scenario_unchanged_but_forecast_changed", False):
        issues.append("scenario_instability")
    if checks.get("price_monotonicity_violation", False):
        issues.append("price_monotonicity_violation")
    if checks.get("promo_sensitivity_missing", False):
        issues.append("promo_sensitivity_missing")
    if checks.get("manual_scenario_fallback_applied", False):
        issues.append("scenario_effect_from_manual_fallback")
    if checks.get("scenario_effect_only_from_fallback", False):
        issues.append("scenario_effect_only_from_fallback")
        score = max(0.0, score - 0.10)
    if checks.get("controls_changed", False) and (not checks.get("model_direct_sensitivity_present", True)):
        issues.append("model_direct_sensitivity_missing")
        score = max(0.0, score - 0.10)
    if checks.get("price_direction_suspicious", False):
        issues.append("price_direction_suspicious")
        score = max(0.0, score - 0.10)

    return {
        "overall_confidence": _label(score),
        "score": score,
        "issues": issues,
        "ood_flags": ood_flags,
        "backtest_summary": {
            "median_wape": med_wape,
            "median_abs_bias_pct": med_bias,
            "median_std_ratio": med_std_ratio,
            "n_windows": int(len(backtest_df)),
        },
        "benchmark_summary": benchmark_df.to_dict("records"),
        "data_sufficiency": data_sufficiency,
        "behavior_checks": checks,
    }
