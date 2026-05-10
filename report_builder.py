"""Streamlit-independent report payload helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional


def build_business_report_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from app import build_business_report_payload as _impl

    return _impl(*args, **kwargs)
