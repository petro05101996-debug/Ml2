from __future__ import annotations

from typing import Any, Dict, List


def build_reason_codes(confidence: float, elasticity: float, rel_change: float, history_days: int) -> List[str]:
    codes: List[str] = []
    if history_days < 60:
        codes.append("недостаточно_данных")
    if confidence < 0.5:
        codes.append("высокая_неопределённость")
    if elasticity < -1.4:
        codes.append("высокая_чувствительность_спроса")
    elif elasticity > -0.5:
        codes.append("низкая_эластичность")
    if abs(rel_change) < 0.01:
        codes.append("изменение_не_даёт_значимого_эффекта")
    if rel_change > 0.1:
        codes.append("цена_близка_к_верхней_границе")
    return codes


def build_business_recommendation(
    current_price: float,
    recommended_price: float,
    current_profit: float,
    recommended_profit: float,
    confidence: float,
    elasticity: float,
    history_days: int,
) -> Dict[str, Any]:
    rel = (recommended_price - current_price) / max(current_price, 1e-9)
    profit_lift = recommended_profit - current_profit
    action = "не менять цену"
    if rel > 0.01:
        action = f"поднять цену на {rel * 100:.1f}%"
    elif rel < -0.01:
        action = f"снизить цену на {abs(rel) * 100:.1f}%"

    risk = "низкий"
    if confidence < 0.45:
        risk = "высокий"
    elif confidence < 0.7:
        risk = "средний"

    reason_codes = build_reason_codes(confidence, elasticity, rel, history_days)
    why = "Рекомендация сформирована на основе прогноза спроса, эластичности и прибыли."
    if "высокая_чувствительность_спроса" in reason_codes:
        why += " Спрос чувствителен к цене, поэтому шаг должен быть осторожным."
    if "недостаточно_данных" in reason_codes:
        why += " История короткая, поэтому включен консервативный режим."

    alt = "Использовать более осторожный сценарий (±2-3% к цене)"
    if risk == "высокий":
        alt = "Отложить изменение цены и накопить больше данных"

    return {
        "action": action,
        "why": why,
        "expected_effect": f"Ожидаемое изменение прибыли: {profit_lift:,.0f}",
        "risk": risk,
        "confidence": float(confidence),
        "alternative": alt,
        "reason_codes": reason_codes,
    }
