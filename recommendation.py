from __future__ import annotations

from typing import Any, Dict, Optional

from decision_logic import build_decision_layer


def _confidence_to_score(confidence: float) -> int:
    score = int(round(float(confidence) * 100))
    return max(0, min(100, score))


def _confidence_level(score: int) -> str:
    if score <= 40:
        return "низкая"
    if score <= 70:
        return "средняя"
    return "высокая"


def _fmt_money(value: float) -> str:
    return f"{float(value):+,.0f} ₽"


def _fmt_units(value: float) -> str:
    return f"{float(value):+,.1f} шт."


def build_business_recommendation(
    current_price: float,
    recommended_price: float,
    current_profit: float,
    recommended_profit: float,
    confidence: float,
    elasticity: float,
    history_days: int,
    current_revenue: Optional[float] = None,
    recommended_revenue: Optional[float] = None,
    current_volume: Optional[float] = None,
    recommended_volume: Optional[float] = None,
    data_quality: Optional[Dict[str, Any]] = None,
    base_ctx: Optional[Dict[str, Any]] = None,
    reason_hints: Optional[Dict[str, Any]] = None,
    predictive_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_price = float(current_price)
    recommended_price = float(recommended_price)
    delta_abs = recommended_price - current_price
    delta_pct = ((recommended_price - current_price) / max(current_price, 1e-9)) * 100.0
    rel = delta_pct / 100.0
    profit_lift = float(recommended_profit - current_profit)

    revenue_lift = None
    if current_revenue is not None and recommended_revenue is not None:
        revenue_lift = float(recommended_revenue) - float(current_revenue)

    volume_lift = None
    if current_volume is not None and recommended_volume is not None:
        volume_lift = float(recommended_volume) - float(current_volume)

    confidence_score = _confidence_to_score(confidence)
    confidence_level = _confidence_level(confidence_score)
    risk_level = "низкий" if confidence_score > 70 else ("средний" if confidence_score > 40 else "высокий")

    if rel > 0.01:
        plain_action = f"Рекомендуется повысить цену на {delta_pct:.1f}%."
        plain_reason = "Маржа растёт быстрее, чем ожидаемое снижение объёма продаж."
    elif rel < -0.01:
        plain_action = f"Рекомендуется снизить цену на {abs(delta_pct):.1f}%."
        plain_reason = "Снижение цены поддерживает объём продаж и может стабилизировать оборот."
    else:
        plain_action = "Рекомендуется сохранить текущую цену без изменений."
        plain_reason = "Текущий уровень цены близок к локальному оптимуму по данным модели."

    if abs(rel) > 0.10 and rel > 0:
        plain_reason = "Шаг заметный: потенциал прибыли выше, но важно внимательно отслеживать реакцию спроса."
    if elasticity < -1.4:
        plain_reason = "Спрос чувствителен к цене: лучше внедрять изменение плавно и следить за объёмом ежедневно."
    elif elasticity > -0.5 and rel < 0:
        plain_reason = "Спрос слабо реагирует на цену: сильное снижение может не дать ожидаемого прироста продаж."
    if history_days < 60:
        plain_reason = "История продаж короткая, поэтому рекомендация предварительная и требует пилотной проверки."

    numbers_parts = [f"прибыль {_fmt_money(profit_lift)}"]
    if revenue_lift is not None:
        numbers_parts.append(f"выручка {_fmt_money(revenue_lift)}")
    if volume_lift is not None:
        numbers_parts.append(f"объём {_fmt_units(volume_lift)}")
    expected_numbers = "Ожидаемое изменение: " + "; ".join(numbers_parts) + "."

    risk_items = [
        "Фактическая реакция спроса может отличаться от исторической оценки модели.",
        "При резких внешних изменениях (акции конкурентов, сезонность, логистика) эффект может быть ниже прогноза.",
    ]
    if history_days < 60:
        risk_items.append("История короче 60 дней: модельная оценка менее стабильна.")
    if confidence_score <= 40:
        risk_items.append("Низкая уверенность модели: решение лучше запускать как короткий тест.")

    warning_text = ""
    if confidence_score <= 40:
        warning_text = "Рекомендация предварительная — данных недостаточно для высокой уверенности."

    confidence_note = (
        f"Уровень доверия: {confidence_level} ({confidence_score}/100). "
        "Это эвристическая оценка и не является гарантией результата."
    )

    next_steps = [
        "Запустите тест на горизонте 7–14 дней и ежедневно контролируйте продажи, выручку и прибыль.",
        "Заранее определите порог отката (например, при существенном падении объёма).",
        "После теста сравните факт с прогнозом и обновите сценарий.",
    ]

    factor_notes = [
        "Если меняете цену, одновременно проверяйте влияние скидки и промо.",
        "При росте логистики (freight) итоговая прибыль может быть ниже прогноза.",
        "Рейтинг и количество отзывов влияют на доверие покупателей и устойчивость спроса.",
    ]

    summary = (
        f"{plain_action} Текущая цена: {current_price:,.2f} ₽, рекомендуемая: {recommended_price:,.2f} ₽ "
        f"({delta_abs:+,.2f} ₽; {delta_pct:+.2f}%). {expected_numbers}"
    )

    decision = build_decision_layer(
        current_price=current_price,
        recommended_price=recommended_price,
        current_profit=current_profit,
        recommended_profit=recommended_profit,
        current_revenue=current_revenue,
        recommended_revenue=recommended_revenue,
        current_volume=current_volume,
        recommended_volume=recommended_volume,
        confidence=float(confidence),
        history_days=int(history_days),
        data_quality=data_quality,
        base_ctx=base_ctx,
        reason_hints=reason_hints,
        predictive_gate=predictive_gate,
    )

    if decision["decision_type"] in {"hold", "no_decision"}:
        plain_action = "Рекомендуется сохранить текущую цену без изменений."
        recommended_price = float(current_price)
        delta_abs = 0.0
        delta_pct = 0.0

    structured = {
        "what_to_do": decision["seller_friendly_summary"],
        "why": decision["seller_friendly_reason"],
        "expected_profit_change": float(profit_lift),
        "expected_revenue_change": revenue_lift,
        "expected_volume_change": volume_lift,
        "expected_margin_change": None,
        "recommended_price": float(decision.get("recommended_price", recommended_price)),
        "price_delta_pct": float(decision.get("price_delta_pct", delta_pct)),
        "risk_level": decision.get("risk_level", risk_level),
        "confidence_level": decision.get("confidence_level", confidence_level),
        "decision_type": decision["decision_type"],
        "decision_strength": decision["decision_strength"],
        "implementation_mode": decision["implementation_mode"],
        "primary_lever": decision["primary_lever"],
        "test_plan": decision["test_plan"],
        "rollback_condition": decision["rollback_condition"],
        "what_to_monitor": decision["what_to_monitor"],
        "why_not_more": decision["why_not_more"],
        "why_not_less": decision["why_not_less"],
        "why_not_now": decision["why_not_now"],
        "assumptions": decision["assumptions"],
        "seller_notes": decision["seller_notes"],
        "key_driver_positive": decision["key_driver_positive"],
        "key_driver_negative": decision["key_driver_negative"],
        "reason_why_this_scenario_wins": decision["reason_why_this_scenario_wins"],
        "conservative_view": decision["conservative_view"],
        "guardrails": {
            "triggered": decision["guardrails_triggered"],
            "absolute_lift_threshold": decision["absolute_lift_threshold"],
            "relative_lift_threshold": decision["relative_lift_threshold"],
            "minimum_meaningful_lift_passed": decision["minimum_meaningful_lift_passed"],
        },
        "rollout": {
            "test_plan": decision["test_plan"],
            "monitor_metrics": decision["monitor_metrics"],
            "success_rule": decision["success_rule"],
            "rollback_rule": decision["rollback_rule"],
            "review_after_days": decision["review_after_days"],
        },
        "when_not_to_use": decision["when_not_to_use"],
    }

    return {
        "summary": summary,
        "what_to_do": plain_action,
        "why": plain_reason,
        "expected_numbers": expected_numbers,
        "risks": risk_items,
        "confidence_note": confidence_note,
        "next_steps": next_steps,
        "factor_notes": factor_notes,
        "price_context": {
            "current_price": current_price,
            "recommended_price": float(decision.get("recommended_price", recommended_price)),
            "delta_abs": float(decision.get("recommended_price", recommended_price) - current_price),
            "delta_pct": float(decision.get("price_delta_pct", delta_pct)),
        },
        "metrics_context": {
            "profit_delta": profit_lift,
            "revenue_delta": revenue_lift,
            "volume_delta": volume_lift,
            "confidence_score": confidence_score,
            "risk_level": risk_level,
        },
        "plain_action": plain_action,
        "plain_reason": plain_reason,
        "plain_effect": f"Ожидаемый эффект: {profit_lift:+,.0f} ₽ за расчётный горизонт.",
        "risk_text": f"Риск: {decision.get('risk_level', risk_level)}.",
        "confidence_text": f"Надёжность рекомендации: {confidence_level} ({confidence_score}/100).",
        "warning_text": warning_text,
        "risks_text": " ".join(risk_items),
        "conditions_text": "Решение следует внедрять поэтапно с мониторингом ключевых KPI.",
        "next_steps_text": " ".join(next_steps),
        "summary_lines": [summary, plain_reason, expected_numbers, confidence_note] + ([warning_text] if warning_text else []),
        "action": plain_action,
        "expected_effect": f"Ожидаемый эффект: {profit_lift:+,.0f} ₽ за расчётный горизонт.",
        "risk": decision.get("risk_level", risk_level),
        "confidence": float(confidence),
        "decision_layer": decision,
        "structured": structured,
        # flattened aliases for UI/API friendliness
        **{k: v for k, v in structured.items() if k not in {"rollout", "guardrails", "conservative_view"}},
        "rollout_plan": structured["rollout"],
        "guardrails": structured["guardrails"],
        "conservative_result": structured["conservative_view"],
        "seller_friendly_summary": decision["seller_friendly_summary"],
        "seller_friendly_reason": decision["seller_friendly_reason"],
        "seller_friendly_risk": decision["seller_friendly_risk"],
        "seller_friendly_next_step": decision["seller_friendly_next_step"],
    }
