from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

FORBIDDEN_PHRASES = [
    "лучше соответствует",
    "балл соответствия",
    "вам подходит",
    "подходит вам",
    "соответствует вашим целям",
    "соответствует риск-профилю",
    "лучший вариант",
    "рекомендуем",
    "советуем",
    "покупайте",
    "продавайте",
    "держите",
    "с максимальным соответствием",
]

FORBIDDEN_REGEX = [
    re.compile(r"\bрекоменду\w+", re.IGNORECASE),
    re.compile(r"\bсовету\w+", re.IGNORECASE),
    re.compile(r"\bподход\w+\s+вам\b", re.IGNORECASE),
    re.compile(r"\bлучш\w+\s+(вариант|выбор|сценарий|инструмент)", re.IGNORECASE),
    re.compile(r"\bсоответств\w+\s+(целям|риск-профилю|профилю)", re.IGNORECASE),
]

DISCLAIMER = (
    "Отчёт носит информационно-аналитический характер. Он не является индивидуальной "
    "инвестиционной рекомендацией, не содержит предложения купить, продать или удерживать "
    "финансовый инструмент, не определяет пригодность инструмента для пользователя и не "
    "формирует инвестиционный профиль. Все расчёты основаны на введённых данных и допущениях."
)


@dataclass
class SafetyCheckResult:
    is_safe: bool
    violations: List[str]
    sanitized_text: str
    disclaimer_required: bool


def find_violations(text: str) -> List[str]:
    low = text.casefold()
    v: List[str] = []
    for p in FORBIDDEN_PHRASES:
        if p in low:
            v.append(p)
    for rx in FORBIDDEN_REGEX:
        if rx.search(text):
            v.append(f"regex:{rx.pattern}")
    return sorted(set(v))


def sanitize_advisory_text(text: str, context: str = "generic") -> str:
    violations = find_violations(text)
    if not violations:
        return text
    if context == "scenario":
        prefix = "В сценарии обнаружены риск-флаги и/или неизвестные параметры. "
    elif context == "offer":
        prefix = "В предложении обнаружены нераскрытые условия и риск-флаги. "
    elif context == "portfolio":
        prefix = "В портфеле обнаружены факторы риска и неизвестные параметры. "
    elif context == "knowledge":
        prefix = "Текст карточки был нейтрализован для исключения рекомендательной лексики. "
    elif context == "report":
        prefix = "В отчёте были заменены формулировки, которые могли выглядеть как рекомендация. "
    else:
        prefix = "Текст содержал формулировки, которые могут восприниматься как рекомендация. Формулировка заменена на нейтральную. "
    return prefix + DISCLAIMER


def check_text_safety(text: str) -> SafetyCheckResult:
    violations = find_violations(text)
    sanitized = sanitize_advisory_text(text)
    return SafetyCheckResult(
        is_safe=not violations,
        violations=violations,
        sanitized_text=sanitized,
        disclaimer_required=bool(violations),
    )
