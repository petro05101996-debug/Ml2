from __future__ import annotations

import re
from typing import Any, Dict, List

MARKETING_PATTERNS = [
    r"без риска",
    r"гарантированн\w*\s+высок\w*\s+доходност\w*",
    r"до\s*\d+[\.,]?\d*%",
    r"лучше вклада",
    r"эксклюзивно",
    r"только сегодня",
    r"успейте",
    r"доходност\w*\s+выше\s+рынка",
]


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def parse_offer_text(text: str, source: str | None = None) -> Dict[str, Any]:
    declared_return = _find(r"(\d+[\.,]?\d*)\s*%", text)
    term_months = _find(r"(\d{1,2})\s*(?:месяц|месяцев|мес)", text)

    marketing: List[str] = []
    for p in MARKETING_PATTERNS:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            marketing.append(m.group(0))

    instrument_type = None
    for token, label in {
        "вклад": "deposit",
        "облигац": "bond",
        "офз": "ofz",
        "фонд": "fund",
        "акци": "stock",
    }.items():
        if token in text.casefold():
            instrument_type = label
            break

    fees = None if "комис" not in text.casefold() else "mentioned"
    early_exit = None if "досроч" not in text.casefold() else "mentioned"
    tax_notes = None if "налог" not in text.casefold() else "mentioned"

    unknown_fields = []
    if fees is None:
        unknown_fields.append("UNKNOWN_FEES")
    if early_exit is None:
        unknown_fields.append("UNKNOWN_EARLY_EXIT")
    if tax_notes is None:
        unknown_fields.append("TAX_ASSUMPTION")

    src = (source or "").casefold()
    source_risk = "NON_OFFICIAL_SOURCE" if src in {"реклама", "telegram", "соцсети", "telegram/соцсети"} or "telegram" in src or "соцсет" in src else None
    if source_risk:
        unknown_fields.append(source_risk)

    return {
        "instrument_type": instrument_type,
        "declared_return": declared_return,
        "term_months": int(term_months) if term_months else None,
        "capital_protection": None,
        "early_exit_conditions": early_exit,
        "fees": fees,
        "tax_notes": tax_notes,
        "source_risk": source_risk,
        "marketing_phrases": marketing,
        "marketing_note": "В тексте есть маркетинговая формулировка. Нужно уточнить, какие условия и риски стоят за заявленной доходностью." if marketing else None,
        "unknown_fields": unknown_fields,
    }
