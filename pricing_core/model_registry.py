from __future__ import annotations

from typing import Any, Dict


def build_model_bundle(**kwargs) -> Dict[str, Any]:
    bundle = dict(kwargs)
    bundle.setdefault("engine_version", "v2_decomposed_baseline_factor_shock")
    return bundle


def extract_model_bundle_summary(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "engine_version": bundle.get("engine_version"),
        "baseline_bt_backend": (bundle.get("trained_baseline_bt") or {}).get("model_backend"),
        "baseline_final_backend": (bundle.get("trained_baseline_final") or {}).get("model_backend"),
        "factor_backend": (bundle.get("trained_factor") or {}).get("model_backend") if bundle.get("trained_factor") else None,
        "has_factor_model": bool(bundle.get("trained_factor") is not None),
        "n_target_rows": int(len(bundle.get("target_history", []))) if bundle.get("target_history") is not None else 0,
        "mode": bundle.get("mode", "baseline_only"),
        "intervals_available": bool(bundle.get("intervals_available", False)),
    }
