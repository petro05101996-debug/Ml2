from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def build_v2_result_contract(result_v2: Dict[str, Any]) -> Dict[str, Any]:
    delta = result_v2.get("delta_summary")
    if isinstance(delta, pd.DataFrame):
        delta_row = delta.iloc[0].to_dict() if len(delta) else {}
    else:
        delta_row = delta or {}

    conf = result_v2.get("confidence", {})
    factor_model_trained = bool(result_v2.get("factor_model_trained", False))
    mode = str(result_v2.get("mode", "fallback_elasticity"))

    if mode == "baseline_plus_scenario":
        headline_action = "Показан baseline и сценарный прогноз. Сравните дельты до внедрения."
        headline_reason = "Факторная модель обучена; используйте отчёт как сценарный, а не как legacy price recommendation."
    elif mode == "fallback_elasticity":
        headline_action = "Сценарий рассчитан в fallback-режиме эластичности."
        headline_reason = "ML factor model недоступна/слаба; применён детерминированный scenario fallback."
    else:
        headline_action = "Используйте baseline forecast: сценарный отклик недоступен."
        headline_reason = "Не удалось применить ни ML factor model, ни fallback-слой."

    return {
        "headline_action": headline_action,
        "headline_reason": headline_reason,
        "baseline_total_demand": float(delta_row.get("baseline_total_demand", 0.0)),
        "scenario_total_demand": float(delta_row.get("scenario_total_demand", 0.0)),
        "demand_delta_pct": float(delta_row.get("demand_delta_pct", 0.0)),
        "baseline_total_revenue": float(delta_row.get("baseline_total_revenue", 0.0)),
        "scenario_total_revenue": float(delta_row.get("scenario_total_revenue", 0.0)),
        "revenue_delta_pct": float(delta_row.get("revenue_delta_pct", 0.0)),
        "baseline_total_profit": float(delta_row.get("baseline_total_profit", 0.0)),
        "scenario_total_profit": float(delta_row.get("scenario_total_profit", 0.0)),
        "profit_delta_pct": float(delta_row.get("profit_delta_pct", 0.0)),
        "overall_confidence": conf.get("overall_confidence", "low"),
        "factor_model_trained": factor_model_trained,
        "mode": mode,
        "scenario_effect_source": result_v2.get("scenario_effect_source", mode),
        "baseline_granularity": result_v2.get("baseline_granularity", "daily"),
        "baseline_strategy": result_v2.get("baseline_strategy", "xgb_recursive"),
        "baseline_selector_reason": result_v2.get("baseline_selector_reason", ""),
        "best_daily_strategy": result_v2.get("best_daily_strategy"),
        "best_weekly_strategy": result_v2.get("best_weekly_strategy"),
        "ood_flags": result_v2.get("ood_flags", []),
        "warnings": conf.get("issues", []),
        "economics_mode": result_v2.get("economics_mode", "net_price"),
        "unit_price_input_type": result_v2.get("unit_price_input_type", "net"),
    }
