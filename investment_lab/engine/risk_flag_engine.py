from __future__ import annotations

from typing import Any, Dict, List

ALL_REQUIRED_FLAG_CODES = [
    "UNKNOWN_FEES",
    "UNKNOWN_EARLY_EXIT",
    "LOW_DATA_QUALITY",
    "HIGH_CONCENTRATION",
    "LOW_LIQUIDITY",
    "DURATION_RISK",
    "CREDIT_RISK",
    "MARKET_RISK",
    "INFLATION_RISK",
    "TAX_ASSUMPTION",
    "MARKETING_LANGUAGE",
    "NON_OFFICIAL_SOURCE",
    "CAPITAL_LOSS_RISK",
]


def make_risk_flag(
    code: str,
    severity: str,
    title: str,
    plain_explanation: str,
    why_it_matters: str,
    what_to_check: List[str],
    source: str,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "plain_explanation": plain_explanation,
        "why_it_matters": why_it_matters,
        "what_to_check": what_to_check,
        "source": source,
    }


def unknown_field_payload(code: str) -> Dict[str, Any]:
    mapping = {
        "UNKNOWN_FEES": (
            "Комиссии не указаны",
            "В предложении не указаны комиссии за покупку, продажу или управление.",
            "Комиссии могут снизить итоговый результат.",
            ["Есть ли комиссия за покупку?", "Есть ли комиссия за продажу?", "Есть ли комиссия за управление?"],
        ),
        "UNKNOWN_EARLY_EXIT": (
            "Не раскрыты условия досрочного выхода",
            "Неясно, что произойдёт, если деньги понадобятся раньше срока.",
            "При досрочном выходе можно потерять часть дохода или столкнуться с ограничениями.",
            ["Можно ли выйти раньше срока?", "Есть ли штраф?", "Сохраняется ли доход?"],
        ),
        "TAX_ASSUMPTION": (
            "Налоговые условия не раскрыты",
            "В предложении нет явного описания налогового режима.",
            "Налоги влияют на итоговый результат после удержаний.",
            ["Какой налоговый режим применяется?", "Есть ли льготы/вычеты?"],
        ),
    }
    title, plain, why, checklist = mapping.get(
        code,
        ("Параметр не раскрыт", "Не хватает данных для точного расчёта.", "Неизвестные условия снижают надёжность расчёта.", ["Запросите полные условия у поставщика"]) ,
    )
    return {
        "code": code,
        "title": title,
        "plain_explanation": plain,
        "why_it_matters": why,
        "what_to_check": checklist,
    }


def flags_from_offer(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    unknown_codes = set(parsed.get("unknown_fields", []))
    if "UNKNOWN_FEES" in unknown_codes:
        flags.append(make_risk_flag("UNKNOWN_FEES", "medium", "Комиссии не указаны", unknown_field_payload("UNKNOWN_FEES")["plain_explanation"], "Комиссии могут снизить итоговый результат.", unknown_field_payload("UNKNOWN_FEES")["what_to_check"], "offer"))
    if "UNKNOWN_EARLY_EXIT" in unknown_codes:
        flags.append(make_risk_flag("UNKNOWN_EARLY_EXIT", "medium", "Не раскрыты условия досрочного выхода", unknown_field_payload("UNKNOWN_EARLY_EXIT")["plain_explanation"], "При досрочном выходе можно потерять часть дохода.", unknown_field_payload("UNKNOWN_EARLY_EXIT")["what_to_check"], "offer"))
    if "TAX_ASSUMPTION" in unknown_codes:
        flags.append(make_risk_flag("TAX_ASSUMPTION", "low", "Налоговые условия не раскрыты", unknown_field_payload("TAX_ASSUMPTION")["plain_explanation"], "Налоги могут заметно изменить итоговый результат.", unknown_field_payload("TAX_ASSUMPTION")["what_to_check"], "offer"))
    if parsed.get("marketing_phrases"):
        flags.append(make_risk_flag("MARKETING_LANGUAGE", "low", "Маркетинговые формулировки в тексте", "В тексте есть рекламные формулировки о доходности.", "Маркетинговые формулировки не заменяют полные условия продукта.", ["Где полный документ условий?", "Какие комиссии и риски указаны официально?"], "offer"))
    if parsed.get("source_risk") == "NON_OFFICIAL_SOURCE":
        flags.append(make_risk_flag("NON_OFFICIAL_SOURCE", "medium", "Неофициальный источник предложения", "Источник предложения не является официальным каналом условий.", "В неофициальном канале могут отсутствовать ключевые ограничения и комиссии.", ["Сверьте условия в официальном документе", "Проверьте сайт/договор"], "offer"))
    if parsed.get("capital_protection") is None:
        flags.append(make_risk_flag("CAPITAL_LOSS_RISK", "medium", "Не раскрыт риск потери капитала", "В тексте не подтверждена защита капитала.", "Без явного условия можно столкнуться с потерей части капитала.", ["Есть ли гарантия возврата капитала?", "Какие исключения указаны в договоре?"], "offer"))
    return flags
