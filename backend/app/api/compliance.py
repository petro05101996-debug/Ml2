from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from investment_lab.engine.safety_text_guard import check_text_safety

router = APIRouter(prefix="/api/compliance", tags=["compliance"])


class ComplianceCheckRequest(BaseModel):
    text: str


class ComplianceCheckResponse(BaseModel):
    is_safe: bool
    violations: list[str]
    sanitized_text: str
    disclaimer_required: bool


@router.post("/check-text", response_model=ComplianceCheckResponse)
def check_text(payload: ComplianceCheckRequest) -> ComplianceCheckResponse:
    result = check_text_safety(payload.text)
    return ComplianceCheckResponse(
        is_safe=result.is_safe,
        violations=result.violations,
        sanitized_text=result.sanitized_text,
        disclaimer_required=result.disclaimer_required,
    )
