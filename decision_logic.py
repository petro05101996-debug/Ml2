from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


DECISION_DEFAULTS: Dict[str, float] = {
    "absolute_lift_threshold": 1500.0,
    "relative_lift_threshold": 0.015,
    "max_volume_drop_for_action": 0.12,
    "max_price_step_for_immediate": 0.06,
    "max_price_step_for_action": 0.12,
    "high_confidence": 0.72,
    "medium_confidence": 0.45,
    "history_days_poor": 45,
    "history_days_medium": 75,
}


def _risk_from_confidence(confidence: float) -> str:
    if confidence >= 0.72:
        return "low"
    if confidence >= 0.45:
        return "medium"
    return "high"


def _primary_lever(base_ctx: Dict[str, Any], data_quality: Dict[str, Any], confidence: float, price_delta_pct: float, profit_delta: float) -> str:
    if data_quality.get("level") in {"poor", "unavailable"}:
        return "data_quality"
    stock = float(base_ctx.get("stock", 0.0) or 0.0)
    sales = float(base_ctx.get("sales", 0.0) or 0.0)
    freight = float(base_ctx.get("freight_value", 0.0) or 0.0)
    price = float(base_ctx.get("price", 0.0) or 0.0)
    promo = float(base_ctx.get("promotion", 0.0) or 0.0)
    rating = float(base_ctx.get("rating", base_ctx.get("review_score", 4.5)) or 4.5)

    if stock > 0 and sales > 0 and stock <= sales * 0.9:
        return "stock"
    if price > 0 and freight >= price * 0.25:
        return "freight"
    if rating < 3.8:
        return "promotion"
    if promo > 0.25 and abs(price_delta_pct) < 2.0:
        return "promotion"
    if confidence < DECISION_DEFAULTS["medium_confidence"] and abs(price_delta_pct) > 6:
        return "data_quality"
    if profit_delta <= 0 and abs(price_delta_pct) > 0:
        return "other"
    return "price"


def build_decision_layer(
    *,
    current_price: float,
    recommended_price: float,
    current_profit: float,
    recommended_profit: float,
    current_revenue: Optional[float],
    recommended_revenue: Optional[float],
    current_volume: Optional[float],
    recommended_volume: Optional[float],
    confidence: float,
    history_days: int,
    data_quality: Optional[Dict[str, Any]] = None,
    base_ctx: Optional[Dict[str, Any]] = None,
    reason_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_quality = dict(data_quality or {})
    base_ctx = dict(base_ctx or {})
    reason_hints = dict(reason_hints or {})

    current_price = float(current_price)
    recommended_price = float(recommended_price)
    delta_price = recommended_price - current_price
    price_delta_pct = (delta_price / max(current_price, 1e-9))

    profit_delta = float(recommended_profit - current_profit)
    revenue_delta = None if current_revenue is None or recommended_revenue is None else float(recommended_revenue - current_revenue)
    volume_delta = None if current_volume is None or recommended_volume is None else float(recommended_volume - current_volume)
    rel_profit = profit_delta / max(abs(float(current_profit)), 1.0)
    rel_volume = 0.0 if volume_delta is None else float(volume_delta / max(abs(float(current_volume)), 1.0))

    abs_thr = DECISION_DEFAULTS["absolute_lift_threshold"]
    rel_thr = DECISION_DEFAULTS["relative_lift_threshold"]
    meaningful_lift = (profit_delta >= abs_thr) or (rel_profit >= rel_thr)

    risk_level = _risk_from_confidence(confidence)
    downside_multiplier = float(np.clip(1.0 - (0.55 * (1.0 - confidence)), 0.35, 0.98))
    downside_profit_delta = float(profit_delta * downside_multiplier)
    downside_revenue_delta = None if revenue_delta is None else float(revenue_delta * downside_multiplier)
    downside_volume_delta = None if volume_delta is None else float(volume_delta * max(0.7, 1.0 - (1.0 - confidence) * 0.5))

    guardrails_triggered: List[str] = []
    if not meaningful_lift:
        guardrails_triggered.append("minimum_meaningful_lift")
    if confidence < DECISION_DEFAULTS["medium_confidence"]:
        guardrails_triggered.append("low_confidence")
    if abs(price_delta_pct) > DECISION_DEFAULTS["max_price_step_for_action"]:
        guardrails_triggered.append("price_step_too_large")
    if rel_volume < -DECISION_DEFAULTS["max_volume_drop_for_action"]:
        guardrails_triggered.append("volume_drop_risk")
    if history_days < DECISION_DEFAULTS["history_days_poor"]:
        guardrails_triggered.append("short_history")
    if data_quality.get("level") in {"poor", "unavailable"}:
        guardrails_triggered.append("data_quality")
    if downside_profit_delta < 0:
        guardrails_triggered.append("downside_negative")

    primary_lever = _primary_lever(base_ctx, data_quality, confidence, price_delta_pct * 100.0, profit_delta)

    if data_quality.get("level") == "unavailable" or confidence < 0.25:
        decision_type = "no_decision"
        implementation_mode = "do_not_change"
        decision_strength = "weak"
    elif (not meaningful_lift) and profit_delta >= 0:
        decision_type = "hold"
        implementation_mode = "monitor_only"
        decision_strength = "weak"
    elif primary_lever != "price":
        decision_type = "test"
        implementation_mode = "monitor_only" if confidence < DECISION_DEFAULTS["medium_confidence"] else "pilot"
        decision_strength = "weak" if confidence < DECISION_DEFAULTS["medium_confidence"] else "moderate"
    elif guardrails_triggered:
        decision_type = "test"
        implementation_mode = "pilot" if confidence >= DECISION_DEFAULTS["medium_confidence"] else "monitor_only"
        decision_strength = "moderate" if confidence >= DECISION_DEFAULTS["medium_confidence"] else "weak"
    else:
        decision_type = "action"
        immediate_ok = (confidence >= DECISION_DEFAULTS["high_confidence"]) and (abs(price_delta_pct) <= DECISION_DEFAULTS["max_price_step_for_immediate"])
        implementation_mode = "immediate" if immediate_ok else "pilot"
        decision_strength = "strong" if immediate_ok else "moderate"

    if decision_type in {"hold", "no_decision"}:
        recommended_price_out = current_price
    else:
        recommended_price_out = recommended_price

    if decision_type == "hold":
        seller_reason = "Ожидаемый экономический эффект изменения цены ниже порога значимости."
    elif decision_type == "no_decision":
        seller_reason = "Надёжность данных/модели недостаточна для управленческого решения по цене."
    elif primary_lever != "price":
        seller_reason = "По текущим сигналам главный рычаг сейчас не цена, а операционные факторы."
    else:
        seller_reason = "Сценарий цены проходит пороги значимости и риск-ограничения."

    key_driver_positive = reason_hints.get("key_driver_positive") or ("Рост unit-экономики" if profit_delta >= 0 else "Поддержка объёма")
    key_driver_negative = reason_hints.get("key_driver_negative") or ("Риск просадки объёма" if rel_volume < 0 else "Неопределённость спроса")
    why_wins = reason_hints.get("reason_why_this_scenario_wins") or "Этот сценарий даёт лучший баланс прибыли и риска среди допустимых вариантов."

    monitor_metrics = ["profit", "revenue", "volume", "margin", "conversion_proxy"]
    review_after_days = 7 if implementation_mode in {"immediate", "pilot"} else 14
    if decision_type == "action":
        test_plan = "Внедрите изменение в ограниченном масштабе и контролируйте KPI ежедневно 7 дней."
        success_rule = "Сохранение или рост прибыли при допустимой просадке объёма не более 8%."
        rollback_rule = "Откатить изменение, если прибыль падает 3 дня подряд или объём падает >12%."
    elif decision_type == "test":
        test_plan = "Запустите пилот на 7–14 дней в ограниченном сегменте."
        success_rule = "Пилот подтверждает направление: прибыль не ниже baseline и риск контролируем."
        rollback_rule = "Остановить пилот при ухудшении прибыли или резком падении объёма."
    elif decision_type == "hold":
        test_plan = "Сохраняйте текущую цену и мониторьте внешние факторы до следующего пересмотра."
        success_rule = "Стабильные KPI без ухудшения маржи и объёма."
        rollback_rule = "Не применимо: цена не меняется."
    else:
        test_plan = "Сначала улучшите данные/операционные факторы, затем пересчитайте сценарий."
        success_rule = "Достигнут приемлемый уровень данных и прогнозной надёжности."
        rollback_rule = "Не применять ценовые изменения до улучшения входных данных."

    return {
        "decision_type": decision_type,
        "decision_strength": decision_strength,
        "implementation_mode": implementation_mode,
        "primary_lever": primary_lever,
        "guardrails_triggered": guardrails_triggered,
        "absolute_lift_threshold": abs_thr,
        "relative_lift_threshold": rel_thr,
        "minimum_meaningful_lift_passed": bool(meaningful_lift),
        "recommended_price": float(recommended_price_out),
        "price_delta_pct": float(((recommended_price_out - current_price) / max(current_price, 1e-9)) * 100.0),
        "expected_profit_change": float(profit_delta),
        "expected_revenue_change": revenue_delta,
        "expected_volume_change": volume_delta,
        "conservative_view": {
            "scenario": "conservative",
            "expected_profit_change": float(downside_profit_delta),
            "expected_revenue_change": downside_revenue_delta,
            "expected_volume_change": downside_volume_delta,
            "assumption": "Ослабленная реакция спроса/эффекта с учётом uncertainty и confidence.",
        },
        "seller_friendly_summary": (
            "Оставьте цену без изменений и мониторьте метрики." if decision_type in {"hold", "no_decision"}
            else ("Рекомендуется пилотное изменение цены." if implementation_mode == "pilot" else "Рекомендуется внедрить изменение цены.")
        ),
        "seller_friendly_reason": seller_reason,
        "seller_friendly_risk": f"Риск: {risk_level}. Уверенность: {int(round(confidence * 100))}/100.",
        "seller_friendly_next_step": test_plan,
        "test_plan": test_plan,
        "what_to_monitor": monitor_metrics,
        "monitor_metrics": monitor_metrics,
        "success_condition": success_rule,
        "rollback_condition": rollback_rule,
        "success_rule": success_rule,
        "rollback_rule": rollback_rule,
        "review_after_days": int(review_after_days),
        "why_not_more": "Более сильный шаг повышает риск потери объёма и ухудшения downside-сценария.",
        "why_not_less": "Слабее шаг может не дать заметного финансового эффекта при тех же операционных усилиях.",
        "why_not_now": "Немедренное масштабное внедрение не рекомендуется без пилота при текущем уровне неопределённости.",
        "assumptions": [
            "Оценки основаны на исторических паттернах и what-if симуляции, а не на causal гарантии.",
            "Внешние изменения (конкуренты, сезонность, логистика) могут изменить фактический эффект.",
        ],
        "seller_notes": [
            "Используйте решение как support-инструмент и подтверждайте пилотом.",
            "При нестабильном запасе или логистике сначала стабилизируйте операционные факторы.",
        ],
        "when_not_to_use": [
            "Резкая смена промо-стратегии относительно истории.",
            "Нестабильный запас/перебои поставок.",
            "Необычный скачок себестоимости или логистики.",
            "Слишком короткая история или низкое качество данных.",
        ],
        "risk_level": risk_level,
        "confidence_level": "high" if confidence >= DECISION_DEFAULTS["high_confidence"] else ("medium" if confidence >= DECISION_DEFAULTS["medium_confidence"] else "low"),
        "key_driver_positive": key_driver_positive,
        "key_driver_negative": key_driver_negative,
        "reason_why_this_scenario_wins": why_wins,
    }
