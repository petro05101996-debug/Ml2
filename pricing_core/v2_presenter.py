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

    if factor_model_trained:
        headline_action = "Показан baseline и сценарный прогноз. Сравните дельты до внедрения."
        headline_reason = "Факторная модель обучена; используйте отчёт как сценарный, а не как legacy price recommendation."
    else:
        headline_action = "Используйте baseline forecast, сценарный факторный отклик недостаточно надёжен."
        headline_reason = "Факторная модель не обучена или слаба; отображается baseline-only режим."

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
        "mode": result_v2.get("mode", "baseline_only"),
        "ood_flags": result_v2.get("ood_flags", []),
        "warnings": conf.get("issues", []),
    }
