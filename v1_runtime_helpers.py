from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def select_weekly_baseline_candidate(
    bundle_results: List[Dict[str, Any]],
    bundle_models: Dict[str, Any],
    bundle_features_selected: Dict[str, List[str]],
    baseline_bundle_name: str,
    nonlegacy_mode: str,
    wape_tol_pp: float,
    corr_tol: float,
    std_ratio_floor: float,
    std_ratio_cap: float,
) -> Dict[str, Any]:
    legacy_reference = next((row for row in bundle_results if row.get("name") == baseline_bundle_name), None)
    if legacy_reference is None and bundle_results:
        legacy_reference = bundle_results[0]
    selected_bundle = legacy_reference
    selection_reason = "legacy_retained_no_nonlegacy_passed_rule"
    candidates_payload: List[Dict[str, Any]] = []
    passed_alternatives: List[Dict[str, Any]] = []

    for candidate in bundle_results:
        row = dict(candidate)
        name = str(row.get("name", ""))
        if name == baseline_bundle_name:
            row["eligible_under_selection_rule"] = True
            row["rejection_reason"] = None
            candidates_payload.append(row)
            continue

        checks: List[str] = []
        if nonlegacy_mode != "active_production":
            checks.append("nonlegacy_mode_disabled")
        if legacy_reference is None:
            checks.append("legacy_reference_missing")
        else:
            if not (float(row.get("holdout_wape", np.inf)) <= float(legacy_reference.get("holdout_wape", np.inf)) + float(wape_tol_pp)):
                checks.append("wape_tolerance_failed")
            candidate_corr = float(row.get("corr", np.nan))
            legacy_corr = float(legacy_reference.get("corr", np.nan))
            if not np.isfinite(candidate_corr):
                checks.append("corr_non_finite")
            elif np.isfinite(legacy_corr) and candidate_corr < legacy_corr - float(corr_tol):
                checks.append("corr_tolerance_failed")
            candidate_std_ratio = float(row.get("std_ratio", np.nan))
            legacy_std_ratio = float(legacy_reference.get("std_ratio", np.nan))
            if not np.isfinite(candidate_std_ratio):
                checks.append("std_ratio_non_finite")
            elif np.isfinite(legacy_std_ratio):
                min_target = legacy_std_ratio + float(std_ratio_floor)
                max_target = legacy_std_ratio + float(std_ratio_cap)
                if candidate_std_ratio < min_target:
                    checks.append("std_ratio_improvement_failed")
                elif candidate_std_ratio > max_target:
                    checks.append("std_ratio_overshoot_failed")

        row["eligible_under_selection_rule"] = len(checks) == 0
        row["rejection_reason"] = None if len(checks) == 0 else checks[0]
        if len(checks) == 0:
            passed_alternatives.append(row)
        candidates_payload.append(row)

    if passed_alternatives:
        passed_alternatives = sorted(
            passed_alternatives,
            key=lambda row: (
                float(row.get("holdout_wape", np.inf)),
                -(float(row.get("std_ratio", -np.inf)) if np.isfinite(float(row.get("std_ratio", np.nan))) else -np.inf),
                -(float(row.get("corr", -np.inf)) if np.isfinite(float(row.get("corr", np.nan))) else -np.inf),
            ),
        )
        selected_bundle = passed_alternatives[0]
        selection_reason = "non_legacy_passed_selection_rule_best_wape"

    if selected_bundle is None:
        selected_name = baseline_bundle_name
        selected_bundle_obj = None
        selection_reason = "fallback_legacy_missing"
    else:
        selected_name = str(selected_bundle["name"])
        selected_bundle_obj = selected_bundle

    return {
        "selected_candidate_name": selected_name,
        "selected_bundle": selected_bundle_obj,
        "selection_reason": selection_reason,
        "comparison_payload": {
            "selected_candidate": selected_name,
            "selection_reason": selection_reason,
            "nonlegacy_baseline_mode": nonlegacy_mode,
            "selection_rule": {
                "wape_tolerance_pp": float(wape_tol_pp),
                "corr_tolerance_down": float(corr_tol),
                "std_ratio_floor": float(std_ratio_floor),
                "std_ratio_cap": float(std_ratio_cap),
                "primary_rank_metric": "holdout_wape",
                "secondary_rank_metric": "std_ratio",
                "tertiary_rank_metric": "corr",
            },
            "legacy_reference": dict(legacy_reference) if legacy_reference else {},
            "candidates": candidates_payload,
        },
    }


def compute_scenario_price_inputs(requested_price: float, train_min: float, train_max: float) -> Dict[str, Any]:
    requested = float(requested_price)
    low = float(train_min) if np.isfinite(train_min) else requested
    high = float(train_max) if np.isfinite(train_max) else requested
    model_price = float(np.clip(requested, low, high))
    clipped = bool(abs(requested - model_price) > 1e-9)
    clip_reason = (
        "price_below_train_min_weekly_baseline_clipped"
        if requested < low
        else "price_above_train_max_weekly_baseline_clipped"
        if requested > high
        else ""
    )
    return {
        "requested_price": requested,
        "model_price": model_price,
        "price_clipped": clipped,
        "clip_reason": clip_reason,
    }


def get_model_backend_status(model: Any) -> Dict[str, str]:
    backend = str(getattr(model, "model_backend", "") or "")
    reason = str(getattr(model, "backend_reason", "") or "")
    if backend:
        return {"model_backend": backend, "backend_reason": reason}
    cls_name = str(model.__class__.__name__).lower() if model is not None else ""
    if "catboost" in cls_name:
        return {"model_backend": "catboost", "backend_reason": "catboost_available"}
    return {"model_backend": "deterministic_fallback", "backend_reason": reason or "catboost_unavailable_or_disabled"}


def build_backend_warning(model_backend: str, backend_reason: str) -> str:
    if str(model_backend) == "deterministic_fallback":
        return f"CatBoost недоступен: используется deterministic fallback ({backend_reason})."
    return ""
