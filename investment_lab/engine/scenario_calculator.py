from __future__ import annotations

from typing import Any, Dict, List


def _num(payload: dict, key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except Exception:
        return default


def calculate_scenario(payload: Dict[str, Any]) -> Dict[str, Any]:
    amount = _num(payload, "amount", 100000.0)
    months = max(1.0, _num(payload, "term_months", 12.0))
    annual_return = _num(payload, "annual_return_pct", 10.0) / 100.0
    fees_pct = _num(payload, "fees_pct", 0.0) / 100.0
    tax_pct = _num(payload, "tax_pct", 13.0) / 100.0 if payload.get("include_taxes", True) else 0.0
    inflation_pct = _num(payload, "inflation_pct", 8.0) / 100.0 if payload.get("include_inflation", True) else 0.0
    stress_drop_pct = _num(payload, "stress_drop_pct", 35.0) / 100.0

    gross = amount * (1 + annual_return * months / 12.0)
    fees = gross * fees_pct
    taxable_income = max(0.0, gross - amount)
    tax = taxable_income * tax_pct
    net_nominal = gross - fees - tax
    inflation_loss = net_nominal * inflation_pct * months / 12.0
    net_real = net_nominal - inflation_loss

    stress_nominal = max(0.0, net_nominal * (1 - stress_drop_pct))
    drawdown = max(0.0, net_nominal - stress_nominal)

    liquidity_score = max(0, min(100, int(70 - (12 if payload.get("early_exit_penalty") else 0) - (8 if months > 24 else 0))))
    risk_score = max(0, min(100, int(35 + stress_drop_pct * 120 + (10 if fees_pct > 0.015 else 0) + (10 if inflation_pct > 0.08 else 0))))
    complexity_score = max(0, min(100, int(30 + (10 if payload.get("instrument_type") in {"bond", "fund"} else 0) + (12 if payload.get("unknown_fields_count", 0) > 2 else 0))))

    sensitivity: List[Dict[str, str]] = [
        {"factor": "Досрочный выход", "effect": "Может снизить итоговый результат", "explanation": "При досрочном выходе часто теряется часть дохода и начислений."},
        {"factor": "Комиссии", "effect": "Снижают чистый результат", "explanation": "Даже небольшие комиссии на длинном сроке заметно уменьшают итог."},
        {"factor": "Инфляция", "effect": "Снижает реальную доходность", "explanation": "Номинальный рост может частично или полностью съедаться ростом цен."},
    ]

    return {
        "base_result": {"starting_amount": amount, "gross_result": round(gross, 2), "net_nominal": round(net_nominal, 2), "net_real": round(net_real, 2)},
        "stress_result": {"stress_nominal": round(stress_nominal, 2), "drawdown": round(drawdown, 2), "stress_drop_pct": round(stress_drop_pct * 100, 2)},
        "fees_impact": {"fees_paid": round(fees, 2), "fees_pct": round(fees_pct * 100, 3)},
        "tax_impact": {"tax_paid": round(tax, 2), "tax_pct": round(tax_pct * 100, 2)},
        "inflation_impact": {"inflation_loss": round(inflation_loss, 2), "inflation_pct": round(inflation_pct * 100, 2)},
        "liquidity": {"score": liquidity_score, "label": "низкая" if liquidity_score < 40 else "средняя" if liquidity_score < 70 else "высокая"},
        "risk": {"score": risk_score, "label": "низкий" if risk_score < 40 else "средний" if risk_score < 70 else "высокий"},
        "complexity": {"score": complexity_score, "label": "низкая" if complexity_score < 40 else "средняя" if complexity_score < 70 else "высокая"},
        "sensitivity_summary": sensitivity,
    }


def calculate_portfolio(payload: Dict[str, Any]) -> Dict[str, Any]:
    positions = payload.get("positions") or []
    total = sum(float(p.get("amount", 0)) for p in positions) or 1.0
    composition = [{"name": p.get("name", "позиция"), "amount": float(p.get("amount", 0)), "share_pct": round(float(p.get("amount", 0)) / total * 100, 2)} for p in positions]
    largest = max((x["share_pct"] for x in composition), default=0.0)
    concentration = {"largest_position_share_pct": largest, "status": "high" if largest > 35 else "medium" if largest > 20 else "low"}
    stress_drop = float(payload.get("stress_drop_pct", 25.0))
    stress_value = total * (1 - stress_drop / 100.0)
    liquidity = {"score": int(max(0, min(100, 75 - (15 if largest > 40 else 5 if largest > 25 else 0)))), "note": "Высокая концентрация увеличивает зависимость результата от одной позиции."}
    return {
        "composition": composition,
        "largest_position_share": largest,
        "asset_class_shares": payload.get("asset_class_shares", []),
        "concentration": concentration,
        "liquidity": liquidity,
        "stress_result": {"base_value": round(total, 2), "stress_value": round(stress_value, 2), "stress_drop_pct": stress_drop},
    }
