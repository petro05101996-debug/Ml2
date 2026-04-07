from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def build_v2_result_contract(result_v2: Dict[str, Any]) -> Dict[str, Any]:
    delta = result_v2.get("delta_summary_current_vs_scenario", result_v2.get("delta_summary"))
    neutral_delta = result_v2.get("delta_summary_neutral_vs_current")
    if isinstance(delta, pd.DataFrame):
        delta_row = delta.iloc[0].to_dict() if len(delta) else {}
    else:
        delta_row = delta or {}
    if isinstance(neutral_delta, pd.DataFrame):
        neutral_row = neutral_delta.iloc[0].to_dict() if len(neutral_delta) else {}
    else:
        neutral_row = neutral_delta or {}

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
        "neutral_total_demand": float(neutral_row.get("neutral_total_demand", 0.0)),
        "as_is_total_demand": float(delta_row.get("as_is_total_demand", delta_row.get("baseline_total_demand", 0.0))),
        "scenario_total_demand": float(delta_row.get("scenario_total_demand", 0.0)),
        "as_is_vs_scenario_demand_delta_pct": float(delta_row.get("demand_delta_pct", 0.0)),
        "neutral_vs_current_demand_delta_pct": float(neutral_row.get("demand_delta_pct", 0.0)),
        "baseline_total_demand": float(delta_row.get("as_is_total_demand", delta_row.get("baseline_total_demand", 0.0))),
        "demand_delta_pct": float(delta_row.get("demand_delta_pct", 0.0)),
        "baseline_total_revenue": float(delta_row.get("as_is_total_revenue", delta_row.get("baseline_total_revenue", 0.0))),
        "scenario_total_revenue": float(delta_row.get("scenario_total_revenue", 0.0)),
        "revenue_delta_pct": float(delta_row.get("revenue_delta_pct", 0.0)),
        "baseline_total_profit": float(delta_row.get("as_is_total_profit", delta_row.get("baseline_total_profit", 0.0))),
        "scenario_total_profit": float(delta_row.get("scenario_total_profit", 0.0)),
        "profit_delta_pct": float(delta_row.get("profit_delta_pct", 0.0)),
        "overall_confidence": conf.get("overall_confidence", "low"),
        "baseline_confidence_level": (conf.get("baseline_confidence") or {}).get("level", "low"),
        "factor_confidence_level": (conf.get("factor_confidence") or {}).get("level", "low"),
        "factor_role": result_v2.get("factor_role", "unavailable"),
        "factor_model_trained": factor_model_trained,
        "mode": mode,
        "scenario_effect_source": result_v2.get("scenario_effect_source", mode),
        "baseline_granularity": result_v2.get("baseline_granularity", "daily"),
        "baseline_strategy": result_v2.get("baseline_strategy", "xgb_recursive"),
        "final_baseline_strategy": result_v2.get("final_baseline_strategy", result_v2.get("baseline_strategy", "xgb_recursive")),
        "final_baseline_granularity": result_v2.get("final_baseline_granularity", result_v2.get("baseline_granularity", "daily")),
        "final_baseline_source": result_v2.get("final_baseline_source", "plan_selection"),
        "baseline_selector_reason": result_v2.get("baseline_selector_reason", ""),
        "baseline_meets_quality_gate": bool(conf.get("baseline_meets_quality_gate", False)),
        "baseline_goal_wape_median_le_25": bool(conf.get("baseline_goal_wape_median_le_25", False)),
        "baseline_goal_wape_max_le_35": bool(conf.get("baseline_goal_wape_max_le_35", False)),
        "baseline_goal_abs_bias_le_7pct": bool(conf.get("baseline_goal_abs_bias_le_7pct", False)),
        "baseline_goal_sum_ratio_in_range": bool(conf.get("baseline_goal_sum_ratio_in_range", False)),
        "baseline_goal_std_ratio_ge_055": bool(conf.get("baseline_goal_std_ratio_ge_055", False)),
        "baseline_rejection_reason": str(conf.get("baseline_rejection_reason", "")),
        "baseline_runner_up_strategy": result_v2.get("baseline_runner_up_strategy", ""),
        "baseline_runner_up_score": result_v2.get("baseline_runner_up_score"),
        "best_daily_strategy": result_v2.get("best_daily_strategy"),
        "best_weekly_strategy": result_v2.get("best_weekly_strategy"),
        "scenario_controls_changed": bool(result_v2.get("scenario_controls_changed", False)),
        "scenario_delta_zero_reason": str(result_v2.get("scenario_delta_zero_reason", "")),
        "ood_flags": result_v2.get("ood_flags", []),
        "warnings": conf.get("issues", []),
        "economics_mode": result_v2.get("economics_mode", "net_price"),
        "unit_price_input_type": result_v2.get("unit_price_input_type", "net"),
    }
