"""Streamlit-independent what-if runner facade."""
from __future__ import annotations

from typing import Any, Dict


def run_what_if_projection(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from app import run_what_if_projection as _impl

    return _impl(*args, **kwargs)
