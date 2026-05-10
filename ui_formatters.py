"""Pure UI formatting helpers shared by Streamlit and tests."""
from __future__ import annotations

from typing import Any


MODE_DESCRIPTIONS = {
    "legacy_current": "Legacy — безопасный базовый прогноз + простые сценарные множители.",
    "enhanced_local_factors": "Enhanced — прозрачный what-if по факторам, но не полноценное ML-обучение на всех факторах.",
    "catboost_full_factors": "CatBoost Full Factors — ML-режим, где модель обучается на доступных факторах и пересчитывает прогноз при их изменении.",
}

STATUS_LABELS = {
    "recommended": "Можно рекомендовать",
    "eligible": "Можно рекомендовать",
    "test_recommended": "Лучше проверить тестом",
    "test_only": "Лучше проверить тестом",
    "experimental_only": "Только экспериментальный сценарий",
    "not_recommended": "Не рекомендовать",
}


def scenario_mode_label(mode_code: Any) -> str:
    return MODE_DESCRIPTIONS.get(str(mode_code), str(mode_code))


def decision_status_label(status_code: Any) -> str:
    return STATUS_LABELS.get(str(status_code), str(status_code))
