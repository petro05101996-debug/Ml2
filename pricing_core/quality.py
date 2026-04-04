from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_price_delta_pct(current_price: float, recommended_price: float) -> float:
    if current_price <= 0 or (not np.isfinite(current_price)):
        return 0.0
    return float(((recommended_price - current_price) / current_price) * 100.0)


def _resolve_recommended_price(result_bundle: Dict[str, Any], current_price: float) -> float:
    for key in ("recommended_price", "best_price", "target_price", "optimal_price", "final_price"):
        try:
            val = float(result_bundle.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0:
            return val
    return float(current_price)


def assess_data_quality(history_days: int, n_points: int, missing_share: float, holdout_wape: float) -> Dict[str, Any]:
    issues: List[str] = []
    if history_days < 60:
        issues.append("История данных короткая (меньше 60 дней).")
    if n_points < 45:
        issues.append("Слишком мало наблюдений для устойчивого прогноза.")
    if missing_share > 0.2:
        issues.append("Высокая доля пропусков в данных.")
    if np.isfinite(holdout_wape) and holdout_wape > 35:
        issues.append("Ошибка прогноза выше нормы, результат может быть неточным.")

    level = "good"
    confidence_cap = 1.0
    if history_days < 30 or n_points < 20:
        level = "unavailable"
        confidence_cap = 0.2
    elif issues:
        if len(issues) >= 3 or (np.isfinite(holdout_wape) and holdout_wape > 50):
            level = "poor"
            confidence_cap = 0.45
        else:
            level = "medium"
            confidence_cap = 0.75

    return {
        "level": level,
        "label": {
            "good": "Можно использовать рекомендацию",
            "medium": "Рекомендация предварительная",
            "poor": "Качество данных низкое",
            "unavailable": "Недостаточно данных для надёжной рекомендации",
        }[level],
        "issues": issues,
        "confidence_cap": float(confidence_cap),
        "can_recommend": level not in {"unavailable"},
    }


def generate_explanation(result_bundle: Dict[str, Any], data_quality: Optional[Dict[str, Any]] = None, scenario_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    current_price = float(result_bundle.get("current_price", 0.0))
    best_price = _resolve_recommended_price(result_bundle, current_price)
    current_profit = float(result_bundle.get("current_profit", 0.0))
    best_profit = float(result_bundle.get("best_profit", current_profit))
    current_sales = float(result_bundle.get("forecast_current", pd.DataFrame()).get("pred_sales", pd.Series(dtype=float)).sum()) if "forecast_current" in result_bundle else 0.0
    best_sales = float(result_bundle.get("forecast_optimal", pd.DataFrame()).get("pred_sales", pd.Series(dtype=float)).sum()) if "forecast_optimal" in result_bundle else 0.0

    price_change_pct = _safe_price_delta_pct(current_price, best_price)
    profit_change_pct = ((best_profit - current_profit) / max(abs(current_profit), 1e-9)) * 100.0
    sales_change_pct = ((best_sales - current_sales) / max(abs(current_sales), 1e-9)) * 100.0
    quality_level = (data_quality or {}).get("level", "good")

    pros: List[str] = []
    cons: List[str] = []
    if profit_change_pct >= 1:
        pros.append(f"Ожидается рост прибыли примерно на {profit_change_pct:+.1f}%.")
    elif profit_change_pct <= -1:
        cons.append(f"Прибыль может снизиться примерно на {profit_change_pct:.1f}%.")
    if sales_change_pct < -2:
        cons.append(f"Продажи могут снизиться примерно на {abs(sales_change_pct):.1f}%.")
    elif sales_change_pct > 2:
        pros.append(f"Продажи могут вырасти примерно на {sales_change_pct:.1f}%.")
    if quality_level in {"poor", "unavailable", "medium"}:
        cons.append("Результат лучше проверить дополнительным пилотом.")
    if scenario_metrics and float(scenario_metrics.get("delta_profit", 0.0)) > 0:
        pros.append("Сценарный анализ подтверждает потенциал роста прибыли.")

    return {
        "summary": f"Цена {price_change_pct:+.1f}% может изменить прибыль на {profit_change_pct:+.1f}% и объём на {sales_change_pct:+.1f}%.",
        "pros": pros,
        "cons": cons,
        "price_change_pct": float(price_change_pct),
        "profit_change_pct": float(profit_change_pct),
        "sales_change_pct": float(sales_change_pct),
        "mean_elasticity": None,
    }
