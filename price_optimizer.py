from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recommendation_gate import evaluate_price_monotonic_sanity, resolve_recommendation_gate


PRICE_OPT_STATUS_CURRENT_OK = "current_price_ok"
PRICE_OPT_STATUS_INCREASE = "price_increase_recommended"
PRICE_OPT_STATUS_DECREASE = "price_decrease_recommended"
PRICE_OPT_STATUS_INSUFFICIENT_SUPPORT = "insufficient_support"
PRICE_OPT_STATUS_NO_PROFIT_DATA = "no_profit_data"
PRICE_OPT_STATUS_NO_VALID_CANDIDATES = "no_valid_candidates"
PRICE_OPT_STATUS_RISKY_ONLY = "risky_only"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _safe_pct_delta(new_value: float, base_value: float) -> float:
    new_value = _finite_float(new_value, 0.0)
    base_value = _finite_float(base_value, 0.0)
    denom = max(abs(base_value), 1e-9)
    return float(((new_value - base_value) / denom) * 100.0)


def _extract_train_prices(trained_bundle: Dict[str, Any], fallback_price: float) -> pd.Series:
    daily_base = trained_bundle.get("daily_base", pd.DataFrame())
    if isinstance(daily_base, pd.DataFrame) and "price" in daily_base.columns:
        prices = pd.to_numeric(daily_base["price"], errors="coerce").dropna()
        prices = prices[prices > 0]
        if len(prices):
            return prices.astype(float)
    return pd.Series([float(fallback_price)], dtype=float)


def build_price_candidate_grid(
    trained_bundle: Dict[str, Any],
    current_price: float,
    candidate_count: int = 25,
    search_pct: float = 0.20,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    current_price = max(0.01, _finite_float(current_price, 0.01))
    candidate_count = int(np.clip(candidate_count, 9, 41))
    search_pct = float(np.clip(search_pct, 0.05, 0.35))

    train_prices = _extract_train_prices(trained_bundle, current_price)
    train_min = float(train_prices.min())
    train_max = float(train_prices.max())
    unique_prices = int(train_prices.round(4).nunique())
    price_span_pct = float((train_max - train_min) / max(current_price, 1e-9))

    local_lo = current_price * (1.0 - search_pct)
    local_hi = current_price * (1.0 + search_pct)
    safe_lo = max(0.01, max(local_lo, train_min * 0.95))
    safe_hi = max(safe_lo, min(local_hi, train_max * 1.05))

    warnings: List[str] = []
    support_level = "high"
    if unique_prices < 4:
        support_level = "low"
        warnings.append("В истории мало уникальных цен. Рекомендация по оптимальной цене может быть ненадёжной.")
    elif unique_prices < 7:
        support_level = "medium"
    if price_span_pct < 0.06:
        support_level = "low"
        warnings.append("Исторический диапазон цен слишком узкий. Модель плохо видит реакцию спроса на изменение цены.")
    if safe_hi <= safe_lo * 1.001:
        safe_lo = current_price * 0.97
        safe_hi = current_price * 1.03
        support_level = "low"
        warnings.append("Безопасный диапазон цен почти отсутствует. Оптимизатор покажет ориентир, но не жёсткую рекомендацию.")

    grid = np.linspace(safe_lo, safe_hi, candidate_count)
    grid = np.unique(np.round(grid.astype(float), 4))
    grid = np.unique(np.append(grid, current_price))
    grid = np.sort(grid)

    return grid, {
        "train_min": train_min,
        "train_max": train_max,
        "unique_prices": unique_prices,
        "price_span_pct": price_span_pct,
        "safe_price_min": float(safe_lo),
        "safe_price_max": float(safe_hi),
        "support_level": support_level,
        "warnings": warnings,
    }


def _row_from_result(price: float, result: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    effective = result.get("effective_scenario", {}) if isinstance(result, dict) else {}
    demand = _finite_float(result.get("demand_total", 0.0), 0.0)
    revenue = _finite_float(result.get("revenue_total", 0.0), 0.0)
    profit = _finite_float(result.get("profit_total_adjusted", result.get("profit_total", result.get("profit_total_raw", 0.0))), 0.0)
    confidence = _finite_float(result.get("confidence", 0.0), 0.0)
    margin_pct = float((profit / max(revenue, 1e-9)) * 100.0) if revenue > 0 else 0.0
    price_clipped = bool(result.get("price_clipped", False) or result.get("clip_applied", False) or effective.get("price_clipped", False))
    ood_flag = bool(result.get("ood_flag", False) or result.get("factor_ood_flag", False) or effective.get("price_out_of_range", False))
    meta = result.get("scenario_engine_meta", {}) if isinstance(result.get("scenario_engine_meta", {}), dict) else {}
    metrics = meta.get("holdout_metrics", {}) if isinstance(meta.get("holdout_metrics", {}), dict) else {}
    target_semantics = meta.get("target_semantics", {}) if isinstance(meta.get("target_semantics", {}), dict) else {}
    return {
        "price": float(price),
        "is_current": bool(abs(float(price) - float(current_price)) <= max(0.01, abs(current_price) * 1e-6)),
        "demand": demand,
        "revenue": revenue,
        "profit": profit,
        "margin_pct": margin_pct,
        "confidence": confidence,
        "confidence_label": str(result.get("confidence_label", "")),
        "support_label": str(result.get("support_label", "")),
        "price_clipped": price_clipped,
        "ood_flag": ood_flag,
        "validation_ok": bool((result.get("validation_gate", {}) or {}).get("ok", True)),
        "monotonicity_status": str((result.get("monotonicity_policy", {}) or {}).get("status", "passed")),
        "ood_importance_score": _finite_float(result.get("ood_importance_score", 0.0), 0.0),
        "model_wape": _finite_float(metrics.get("wape", result.get("wape", np.nan)), np.nan),
        "naive_improvement_pct": _finite_float(metrics.get("naive_improvement_pct", np.nan), np.nan),
        "rolling_retrain_verdict": str((meta.get("rolling_retrain_backtest", {}) or metrics.get("rolling_retrain_backtest", {}) or {}).get("verdict", "")),
        "stockout_share": _finite_float(target_semantics.get("stockout_share", result.get("stockout_share", 0.0)), 0.0),
        "cost_proxied": bool(result.get("cost_proxied", result.get("cost_is_proxy", False))),
        "cost_missing": bool(result.get("cost_missing", False)),
    }



def _price_stability(best_row: Dict[str, Any], valid: pd.DataFrame) -> Dict[str, Any]:
    if len(valid) <= 1:
        return {"stability_score": 0.0, "stable": False, "near_best_range": [float(best_row["price"]), float(best_row["price"])]}
    work = valid.copy().sort_values("price").reset_index(drop=True)
    prices = pd.to_numeric(work["price"], errors="coerce")
    best_price = float(best_row["price"])
    idx = int((prices - best_price).abs().idxmin())
    lo = max(0, idx - 2)
    hi = min(len(work), idx + 3)
    neigh = work.iloc[lo:hi].copy()
    best_profit = max(abs(float(best_row.get("profit", 0.0))), 1e-9)
    profits = pd.to_numeric(neigh["profit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    demands = pd.to_numeric(neigh["demand"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    profit_cv = float(np.std(profits) / best_profit) if len(profits) else 1.0
    demand_den = max(abs(float(np.nanmean(demands))) if len(demands) else 0.0, 1e-9)
    demand_cv = float(np.std(demands) / demand_den) if len(demands) else 1.0
    score = float(np.clip(1.0 - (0.7 * profit_cv + 0.3 * demand_cv), 0.0, 1.0))
    near = work[pd.to_numeric(work["profit"], errors="coerce") >= best_profit * 0.98]
    return {
        "stability_score": score,
        "stable": bool(score >= 0.55 and len(neigh) >= 3),
        "near_best_range": [float(near["price"].min()), float(near["price"].max())] if len(near) else [best_price, best_price],
    }

def _classify_recommendation(current_row: Dict[str, Any], best_row: Dict[str, Any], grid_meta: Dict[str, Any], min_profit_uplift_pct: float) -> Dict[str, Any]:
    current_price = float(current_row["price"])
    best_price = float(best_row["price"])
    profit_delta = float(best_row["profit"] - current_row["profit"])
    profit_delta_pct = _safe_pct_delta(float(best_row["profit"]), float(current_row["profit"]))
    demand_delta_pct = _safe_pct_delta(float(best_row["demand"]), float(current_row["demand"]))
    revenue_delta_pct = _safe_pct_delta(float(best_row["revenue"]), float(current_row["revenue"]))
    if grid_meta.get("support_level") == "low":
        return {"status": PRICE_OPT_STATUS_INSUFFICIENT_SUPPORT, "title": "Надёжно рекомендовать цену нельзя", "text": "В истории недостаточно ценового сигнала. Можно посмотреть расчёт как ориентир, но использовать его как автоматическую рекомендацию нельзя.", "profit_delta": profit_delta, "profit_delta_pct": profit_delta_pct, "demand_delta_pct": demand_delta_pct, "revenue_delta_pct": revenue_delta_pct}
    if profit_delta_pct < float(min_profit_uplift_pct):
        return {"status": PRICE_OPT_STATUS_CURRENT_OK, "title": "Текущая цена выглядит эффективной", "text": "Оптимизатор проверил соседние цены, но лучшая найденная цена не даёт достаточного прироста прибыли. Менять цену только на основе модели не стоит.", "profit_delta": profit_delta, "profit_delta_pct": profit_delta_pct, "demand_delta_pct": demand_delta_pct, "revenue_delta_pct": revenue_delta_pct}
    if best_price > current_price:
        return {"status": PRICE_OPT_STATUS_INCREASE, "title": "Цена может быть занижена", "text": "Модель показывает, что умеренное повышение цены может увеличить прибыль. Спрос может снизиться, но рост маржи компенсирует снижение объёма.", "profit_delta": profit_delta, "profit_delta_pct": profit_delta_pct, "demand_delta_pct": demand_delta_pct, "revenue_delta_pct": revenue_delta_pct}
    if best_price < current_price:
        return {"status": PRICE_OPT_STATUS_DECREASE, "title": "Цена может быть завышена", "text": "Модель показывает, что снижение цены может увеличить прибыль за счёт роста спроса.", "profit_delta": profit_delta, "profit_delta_pct": profit_delta_pct, "demand_delta_pct": demand_delta_pct, "revenue_delta_pct": revenue_delta_pct}
    return {"status": PRICE_OPT_STATUS_CURRENT_OK, "title": "Текущая цена выглядит эффективной", "text": "Лучшая найденная цена совпадает с текущей.", "profit_delta": profit_delta, "profit_delta_pct": profit_delta_pct, "demand_delta_pct": demand_delta_pct, "revenue_delta_pct": revenue_delta_pct}


def analyze_price_optimization(trained_bundle: Dict[str, Any], current_price: float, runner: Callable[..., Dict[str, Any]], horizon_days: int, scenario_calc_mode: str, price_guardrail_mode: str, overrides: Optional[Dict[str, Any]] = None, factor_overrides: Optional[Dict[str, Any]] = None, freight_multiplier: float = 1.0, demand_multiplier: float = 1.0, candidate_count: int = 25, search_pct: float = 0.20, min_confidence: float = 0.45, min_profit_uplift_pct: float = 3.0) -> Dict[str, Any]:
    overrides = dict(overrides or {})
    factor_overrides = dict(factor_overrides or {})
    grid, grid_meta = build_price_candidate_grid(trained_bundle, float(current_price), int(candidate_count), float(search_pct))
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for price in grid:
        try:
            result = runner(trained_bundle, manual_price=float(price), freight_multiplier=float(freight_multiplier), demand_multiplier=float(demand_multiplier), horizon_days=int(horizon_days), overrides=overrides, factor_overrides=factor_overrides, scenario_calc_mode=str(scenario_calc_mode), price_guardrail_mode=str(price_guardrail_mode))
            rows.append(_row_from_result(float(price), result, float(current_price)))
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc).lower():
                raise
            result = runner(trained_bundle, manual_price=float(price), freight_multiplier=float(freight_multiplier), demand_multiplier=float(demand_multiplier), horizon_days=int(horizon_days), overrides=overrides, scenario_calc_mode=str(scenario_calc_mode))
            rows.append(_row_from_result(float(price), result, float(current_price)))
        except Exception as exc:
            errors.append(f"Цена {float(price):.2f}: {exc}")
    candidates = pd.DataFrame(rows)
    if len(candidates) == 0:
        return {"status": PRICE_OPT_STATUS_NO_VALID_CANDIDATES, "current_price": float(current_price), "recommended_price": None, "recommendation_title": "Оптимизатор не смог рассчитать цены", "recommendation_text": "Не удалось построить расчёт по сетке цен.", "warnings": grid_meta.get("warnings", []) + errors[:5], "candidates": candidates, "grid_meta": grid_meta}
    current_rows = candidates[candidates["is_current"] == True]
    current_row = current_rows.iloc[0].to_dict() if len(current_rows) else candidates.iloc[(pd.to_numeric(candidates["price"], errors="coerce") - float(current_price)).abs().idxmin()].to_dict()
    if not np.isfinite(float(current_row.get("profit", np.nan))):
        return {"status": PRICE_OPT_STATUS_NO_PROFIT_DATA, "current_price": float(current_price), "recommended_price": None, "recommendation_title": "Недостаточно данных для оптимизации прибыли", "recommendation_text": "Прибыль не рассчитана корректно, поэтому оптимальную цену по прибыли выбирать нельзя.", "warnings": grid_meta.get("warnings", []) + errors[:5], "candidates": candidates, "grid_meta": grid_meta}
    candidates["valid_for_recommendation"] = (
        (pd.to_numeric(candidates["profit"], errors="coerce").notna())
        & (pd.to_numeric(candidates["profit"], errors="coerce") > 0)
        & (pd.to_numeric(candidates["revenue"], errors="coerce") > 0)
        & (pd.to_numeric(candidates["confidence"], errors="coerce") >= float(min_confidence))
        & (~candidates["price_clipped"].astype(bool))
        & (~candidates["ood_flag"].astype(bool))
        & (candidates["validation_ok"].astype(bool))
        & (~candidates["support_label"].astype(str).str.lower().isin(["low", "низкая", "низкий"]))
        & (~candidates["monotonicity_status"].astype(str).str.lower().eq("failed"))
    )
    valid = candidates[candidates["valid_for_recommendation"]].copy()
    if len(valid) == 0:
        best_display = candidates.sort_values(["profit", "confidence"], ascending=[False, False]).iloc[0].to_dict()
        return {"status": PRICE_OPT_STATUS_RISKY_ONLY, "current_price": float(current_price), "recommended_price": None, "best_display_price": float(best_display.get("price", current_price)), "recommendation_title": "Есть расчётные варианты, но они рискованные", "recommendation_text": "Оптимизатор нашёл цены с потенциальной прибылью, но они не прошли фильтры надёжности. Не используйте их как рекомендацию без пилотной проверки.", "warnings": grid_meta.get("warnings", []) + errors[:5], "candidates": candidates, "grid_meta": grid_meta}
    valid["score"] = pd.to_numeric(valid["profit"], errors="coerce").fillna(0.0) * (0.70 + 0.30 * pd.to_numeric(valid["confidence"], errors="coerce").fillna(0.0))
    best_row = valid.sort_values(["score", "profit", "confidence"], ascending=[False, False, False]).iloc[0].to_dict()
    best_score = float(best_row.get("score", 0.0))
    near_best = valid[valid["score"] >= best_score * 0.98].copy()
    rec_min = float(near_best["price"].min()) if len(near_best) else float(best_row["price"])
    rec_max = float(near_best["price"].max()) if len(near_best) else float(best_row["price"])
    stability = _price_stability(best_row, valid)
    classification = _classify_recommendation(current_row, best_row, grid_meta, float(min_profit_uplift_pct))
    monotonicity_policy = evaluate_price_monotonic_sanity(
        current_row.get("price"), best_row.get("price"), current_row.get("demand"), best_row.get("demand")
    )
    gate = resolve_recommendation_gate(
        model_quality={
            "wape": best_row.get("model_wape"),
            "naive_improvement_pct": best_row.get("naive_improvement_pct"),
            "stockout_share": best_row.get("stockout_share"),
            "rolling_retrain_backtest": {"verdict": best_row.get("rolling_retrain_verdict", "")},
        },
        factor_policy={"ood_importance_score": float(best_row.get("ood_importance_score", 0.0))},
        economic_significance={"profit_delta_pct": float(classification["profit_delta_pct"]), "profit_action": True, "cost_proxied": bool(best_row.get("cost_proxied", False)), "cost_missing": bool(best_row.get("cost_missing", False))},
        price_policy={"monotonicity_policy": monotonicity_policy},
        cost_policy={"cost_proxied": bool(best_row.get("cost_proxied", False)), "cost_missing": bool(best_row.get("cost_missing", False)), "profit_action": True},
        decision_reliability={"status_namespace": "decision", "base_status": "recommended", "allow_unknown_wape_for_test_recommendation": True},
    )
    warnings = list(grid_meta.get("warnings", []))
    warnings.extend(errors[:5])
    warnings.extend(monotonicity_policy.get("warnings", []))
    warnings.extend(gate.get("warnings", []))
    if not stability["stable"] and classification["status"] in {PRICE_OPT_STATUS_INCREASE, PRICE_OPT_STATUS_DECREASE}:
        classification["status"] = PRICE_OPT_STATUS_RISKY_ONLY
        classification["title"] = "Цена выглядит неустойчивой"
        classification["text"] = "Лучшая точка на сетке не подтверждается соседними ценами. Используйте диапазон как гипотезу для теста, а не как автоматическую рекомендацию."
        warnings.append("Оптимум цены неустойчив на соседних точках сетки.")

    gate_status = str(gate.get("decision_status", "not_recommended"))
    if gate_status == "not_recommended":
        classification["status"] = PRICE_OPT_STATUS_RISKY_ONLY
        classification["title"] = "Рекомендация заблокирована"
        classification["text"] = "Расчёт можно посмотреть как симуляцию, но использовать как рекомендацию нельзя."
    elif gate_status == "experimental_only":
        classification["status"] = PRICE_OPT_STATUS_RISKY_ONLY
        classification["title"] = "Только экспериментальная гипотеза"
        classification["text"] = "Сценарий можно проверять только через controlled test, не внедрять автоматически."
    elif gate_status == "test_recommended":
        classification["status"] = PRICE_OPT_STATUS_RISKY_ONLY
        classification["title"] = "Только через тест"
        classification["text"] = "Модель видит потенциальный эффект, но надёжности недостаточно для прямой рекомендации."

    price_label_key = "recommended_price" if gate_status == "recommended" else "hypothesis_price"
    return {"status": classification["status"], "decision_status": gate_status, "usage_policy": gate.get("usage_policy", {}), "price_label_key": price_label_key, "current_price": float(current_price), "recommended_price": float(best_row["price"]), "recommended_price_min": rec_min, "recommended_price_max": rec_max, "stability_score": float(stability["stability_score"]), "near_best_range": stability["near_best_range"], "recommendation_gate": gate.get("decision_status"), "recommendation_gate_details": gate, "monotonicity_policy": monotonicity_policy, "current_demand": float(current_row.get("demand", 0.0)), "recommended_demand": float(best_row.get("demand", 0.0)), "current_revenue": float(current_row.get("revenue", 0.0)), "recommended_revenue": float(best_row.get("revenue", 0.0)), "current_profit": float(current_row.get("profit", 0.0)), "recommended_profit": float(best_row.get("profit", 0.0)), "profit_delta": float(classification["profit_delta"]), "profit_delta_pct": float(classification["profit_delta_pct"]), "demand_delta_pct": float(classification["demand_delta_pct"]), "revenue_delta_pct": float(classification["revenue_delta_pct"]), "confidence": float(best_row.get("confidence", 0.0)), "confidence_label": str(best_row.get("confidence_label", "")), "support_label": str(best_row.get("support_label", "")), "recommendation_title": classification["title"], "recommendation_text": classification["text"], "warnings": warnings, "candidates": candidates.sort_values("price").reset_index(drop=True), "grid_meta": grid_meta, "scenario_calc_mode": str(scenario_calc_mode), "price_guardrail_mode": str(price_guardrail_mode)}


def build_price_optimizer_signature(current_price: float, horizon_days: int, scenario_calc_mode: str, price_guardrail_mode: str, overrides: Optional[Dict[str, Any]] = None, factor_overrides: Optional[Dict[str, Any]] = None, freight_multiplier: float = 1.0, demand_multiplier: float = 1.0, candidate_count: int = 25, search_pct: float = 0.20) -> Dict[str, Any]:
    return {"current_price": round(float(current_price), 6), "horizon_days": int(horizon_days), "scenario_calc_mode": str(scenario_calc_mode), "price_guardrail_mode": str(price_guardrail_mode), "overrides": dict(overrides or {}), "factor_overrides": dict(factor_overrides or {}), "freight_multiplier": round(float(freight_multiplier), 6), "demand_multiplier": round(float(demand_multiplier), 6), "candidate_count": int(candidate_count), "search_pct": round(float(search_pct), 6)}
