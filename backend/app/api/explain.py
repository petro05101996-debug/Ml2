from __future__ import annotations

from fastapi import APIRouter

from investment_lab.data.knowledge_base import build_knowledge_card

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/instrument/{instrument_type}")
def get_instrument(instrument_type: str) -> dict:
    return build_knowledge_card(instrument_type)
