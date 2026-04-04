from __future__ import annotations

CONFIG = {
    "MIN_COVERAGE_DAYS": 80,
    "MIN_TOTAL_SALES": 100,
    "MIN_UNIQUE_PRICES": 6,
    "MIN_PRICE_CHANGES": 5,
    "MIN_REL_PRICE_SPAN": 0.08,
    "ENSEMBLE_SIZE": 5,
    "MAX_TRAIN_ROWS_PER_MODEL": 6000,
    "HORIZON_DAYS_DEFAULT": 30,
    "COST_PROXY_RATIO": 0.65,
    "RF_TREES": 250,
    "RF_DEPTH": 12,
}

OBJECTIVE_LABEL_TO_MODE = {
    "Максимум прибыли": "maximize_profit",
    "Максимум выручки": "maximize_revenue",
    "Сохранить объём": "protect_volume",
    "Сбалансированный режим": "balanced_mode",
}

OBJECTIVE_HINTS = {
    "Максимум прибыли": "Максимум прибыли — выбираем вариант с наибольшим ожидаемым денежным результатом.",
    "Максимум выручки": "Максимум выручки — приоритет отдаётся росту оборота.",
    "Сохранить объём": "Сохранить объём — уменьшаем риск падения продаж.",
    "Сбалансированный режим": "Сбалансированный режим — компромисс между прибылью, выручкой и объёмом.",
}
