from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_main_decision_text(result_bundle: Dict[str, Any]) -> Dict[str, str]:
    current_price = float(result_bundle.get("current_price", 0.0))
    biz = result_bundle.get("business_recommendation", {})
    structured = biz.get("structured", {}) if isinstance(biz, dict) else {}
    recommended_price_num = float(
        structured.get(
            "recommended_price",
            biz.get("recommended_price", result_bundle.get("recommended_price", result_bundle.get("best_price", current_price))),
        )
    )
    best_price = recommended_price_num if np.isfinite(recommended_price_num) and recommended_price_num > 0 else current_price

    delta_abs = best_price - current_price
    delta_pct = ((best_price - current_price) / current_price) * 100.0 if current_price > 0 else 0.0
    abs_lift = float(result_bundle.get("best_profit", 0.0) - result_bundle.get("current_profit", 0.0))

    dq = result_bundle.get("data_quality", {})
    action_line = biz.get("seller_friendly_summary", biz.get("plain_action", "Рекомендуется сохранить текущую цену без изменений."))
    reason_line = biz.get("seller_friendly_reason", biz.get("plain_reason", "Текущая цена близка к оптимальной в рамках доступных данных."))
    effect_line = biz.get("plain_effect", f"Ожидаемый эффект: {abs_lift:+,.0f} ₽ за 30 дней.")
    reliability_line = biz.get("seller_friendly_risk", f"{biz.get('risk_text', 'Риск: n/a.')} {biz.get('confidence_text', '')}".strip())

    warning_line = ""
    if dq:
        warning_line = dq.get("warning", "")
    if not warning_line and result_bundle.get("warning"):
        warning_line = str(result_bundle.get("warning"))

    decision_type = str(biz.get("decision_type", "action"))
    implementation_mode = str(biz.get("implementation_mode", "pilot"))
    primary_lever = str(biz.get("primary_lever", "price"))
    decision_strength = str(biz.get("decision_strength", "moderate"))

    return {
        "action": action_line,
        "reason": reason_line,
        "effect": effect_line,
        "reliability": reliability_line,
        "recommended_price": f"{best_price:,.2f} ₽",
        "price_delta": f"{delta_abs:+,.2f} ₽ ({delta_pct:+.1f}%)",
        "profit_delta": f"{abs_lift:+,.0f} ₽ / 30 дн.",
        "warning": warning_line,
        "decision_type": decision_type,
        "implementation_mode": implementation_mode,
        "primary_lever": primary_lever,
        "decision_strength": decision_strength,
    }


def scenario_param_diff_table(scenario_inputs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not scenario_inputs:
        return pd.DataFrame()
    base = scenario_inputs[0]
    labels_units = {
        "price": ("Цена", "₽"),
        "demand_multiplier": ("Сила реакции спроса", "×"),
        "freight_multiplier": ("Изменение логистики", "×"),
        "cost_multiplier": ("Изменение себестоимости", "×"),
        "discount_multiplier": ("Изменение скидки", "×"),
        "promotion": ("Промо", "доля"),
        "stock_cap": ("Лимит запаса", "шт."),
        "rating": ("Рейтинг", "балл"),
        "reviews_count": ("Отзывы", "шт."),
        "horizon_days": ("Горизонт", "дней"),
    }
    rows = []
    for row in scenario_inputs:
        for key, (label, unit) in labels_units.items():
            val = float(row.get(key, np.nan))
            base_val = float(base.get(key, np.nan))
            diff = val - base_val if np.isfinite(val) and np.isfinite(base_val) else np.nan
            rows.append(
                {
                    "Сценарий": str(row.get("name", "scenario")),
                    "Параметр": label,
                    "Ед.": unit,
                    "Значение сценария": val,
                    "Базовое значение": base_val,
                    "Δ к базовому": diff,
                }
            )
    return pd.DataFrame(rows)


def format_factor_value(value: Any, unit: str) -> str:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return "—"
    if unit == "%":
        return f"{float(num) * 100:.1f}%"
    return f"{float(num):,.4f} {unit}" if unit not in {"—", ""} else f"{float(num):,.4f}"
