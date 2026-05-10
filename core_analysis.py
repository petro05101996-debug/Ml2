"""Streamlit-independent entry points for full pricing analysis.

This module intentionally does not import ``streamlit`` at module import time so
unit tests and batch jobs can import core analysis contracts without UI deps.
"""
from __future__ import annotations

from typing import Any, Dict


def run_full_pricing_analysis_universal(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run the universal pricing analysis via the UI module implementation.

    The heavy implementation still lives in ``app.py`` for backward
    compatibility. Import is lazy to keep this core module Streamlit-free.
    """
    from app import run_full_pricing_analysis_universal as _impl

    return _impl(*args, **kwargs)
