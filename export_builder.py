"""Streamlit-independent export builder facade."""
from __future__ import annotations

from typing import Any, Dict, Optional
from io import BytesIO


def build_excel_export_buffer(result_dict: Dict[str, Any], what_if_result: Optional[Dict[str, Any]] = None) -> BytesIO:
    from app import build_excel_export_buffer as _impl

    return _impl(result_dict, what_if_result)
