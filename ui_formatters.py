"""Pure UI formatting helpers shared by Streamlit and tests."""
from __future__ import annotations

from typing import Any


MODE_DESCRIPTIONS = {
    "legacy_current": "Базовый режим — текущий план + простые сценарные множители.",
    "enhanced_local_factors": "Расширенный сценарный режим — дневной пересчёт факторов поверх текущего плана.",
    "catboost_full_factors": "Факторная модель — прогноз спроса пересчитывается по доступным факторам.",
}

TECHNICAL_TERM_LABELS = {
    "CatBoost": "факторная модель",
    "CatBoost Full Factors": "Факторная модель",
    "legacy": "базовый режим",
    "enhanced": "расширенный сценарный режим",
    "guardrail": "защитный режим",
    "Validation gate": "проверка корректности сценария",
    "OOD": "сценарий вне похожей истории",
    "JSON": "технический файл",
    "bundle": "расчётный контекст",
}

STATUS_LABELS = {
    "recommended": "Рекомендация",
    "eligible": "Можно рассмотреть",
    "test_recommended": "Можно рассмотреть",
    "controlled_test_only": "Можно рассмотреть",
    "test_only": "Можно рассмотреть",
    "experimental_only": "Гипотеза",
    "not_recommended": "Не рекомендуется",
    "blocked": "Заблокировано",
}


def scenario_mode_label(mode_code: Any) -> str:
    return MODE_DESCRIPTIONS.get(str(mode_code), str(mode_code))


def decision_status_label(status_code: Any) -> str:
    return STATUS_LABELS.get(str(status_code), str(status_code))


def human_technical_label(value: Any) -> str:
    text = str(value)
    for raw, label in TECHNICAL_TERM_LABELS.items():
        text = text.replace(raw, label)
    return text
