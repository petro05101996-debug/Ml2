"""Compatibility bridge after Streamlit UI removal."""
from __future__ import annotations
from typing import Any, Dict

try:
    from backend.app.main import app as app  # type: ignore
except Exception:  # pragma: no cover
    app = None

CATBOOST_FULL_FACTOR_MODE = "catboost_full_factors"


def run_full_pricing_analysis_universal(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from core_analysis import run_full_pricing_analysis_universal as _impl
    return _impl(*args, **kwargs)


def run_what_if_projection(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from what_if_runner import run_what_if_projection as _impl
    return _impl(*args, **kwargs)


def build_excel_export_buffer(*args: Any, **kwargs: Any):
    from export_builder import build_excel_export_buffer as _impl
    return _impl(*args, **kwargs)


def build_manual_scenario_artifacts(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {"status": "not_available_in_fastapi_runtime", "args": len(args), "kwargs": list(kwargs.keys())}


def classify_economic_verdict(*args: Any, **kwargs: Any) -> str:
    return "advisory"


def build_recommended_mode_status(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {"recommended_mode": CATBOOST_FULL_FACTOR_MODE, "status": "advisory"}


def make_walk_forward_oof_baseline(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {"status": "not_available_in_fastapi_runtime", "oof": []}


__all__ = [k for k in globals() if not k.startswith("_")]
