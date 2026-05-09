from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from decision_math import clamp, extract_metric, finite_or_none, safe_div, safe_float, safe_pct_delta, score_0_100

PROFIT_KEYS = ("profit_total_adjusted", "profit_total", "profit_total_raw", "profit", "gross_profit")


def _as_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.DataFrame()


def _history(results: Dict[str, Any], trained_bundle: Dict[str, Any]) -> pd.DataFrame:
    for key in ("history_daily", "daily_base", "df_daily"):
        df = _as_df(results.get(key))
        if len(df):
            return df
    return _as_df(trained_bundle.get("daily_base"))


def _series(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(dtype=float)


def _target(candidate: Dict[str, Any], name: str = "target_value") -> float | None:
    val = finite_or_none(candidate.get(name))
    if val is not None:
        return val
    params = candidate.get("scenario_params") or {}
    return finite_or_none(params.get("manual_price"))


def _metric(result: Dict[str, Any], keys=PROFIT_KEYS) -> float | None:
    for key in keys:
        val = finite_or_none(result.get(key))
        if val is not None:
            return val
    return None


def _data_quality(results: Dict[str, Any], trained_bundle: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    df = _history(results, trained_bundle)
    warnings: List[str] = []
    if len(df) == 0:
        return 0.0, {"history_days": 0, "nonzero_days": 0, "missing_share": 1.0}, ["История отсутствует: надёжность решения недостаточна."]
    history_days = int(len(df))
    sales = _series(df, "sales", "actual_sales", "demand")
    nonzero_days = int((sales.fillna(0.0) > 0).sum()) if len(sales) else 0
    dq = results.get("data_quality") or results.get("quality_report", {}).get("data_quality", {}) or {}
    missing_share = finite_or_none(dq.get("missing_share"))
    if missing_share is None:
        missing_share = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    duplicate_count = int(safe_float(dq.get("duplicate_count", dq.get("duplicates", 0)), 0.0))
    mean_sales = float(sales.fillna(0.0).mean()) if len(sales) else 0.0
    cv = float(sales.fillna(0.0).std(ddof=0) / max(abs(mean_sales), 1e-9)) if len(sales) else 0.0
    history_score = min(history_days / 180.0, 1.0) * 35.0
    nonzero_score = min(nonzero_days / 90.0, 1.0) * 25.0
    missing_score = max(0.0, 1.0 - float(missing_share) / 0.4) * 20.0
    variability_score = 20.0 if 0.03 <= cv <= 5.0 and nonzero_days >= 30 else (10.0 if nonzero_days >= 10 else 5.0)
    score = history_score + nonzero_score + missing_score + variability_score
    if duplicate_count > 0:
        score -= min(15.0, duplicate_count / max(history_days, 1) * 100.0)
        warnings.append("Найдены дубли в данных: надёжность снижена.")
    if history_days < 60:
        warnings.append("Короткая история (<60 дней): рекомендации нестабильны.")
    if nonzero_days < 30:
        warnings.append("Мало дней с ненулевыми продажами: статистическая поддержка слабая.")
    flat_sales = cv < 0.03 or nonzero_days < 10
    if flat_sales:
        warnings.append("История продаж почти плоская или почти вся нулевая: рекомендации нельзя считать устойчивыми.")
    if missing_share > 0.2:
        warnings.append("Высокая доля пропусков в данных: надёжность снижена.")
    return score_0_100(score), {"history_days": history_days, "nonzero_days": nonzero_days, "missing_share": float(missing_share), "sales_cv": cv, "flat_sales": flat_sales}, warnings


def _model_quality(results: Dict[str, Any], scenario_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    warnings: List[str] = []
    wape = extract_metric(results, "quality_report.holdout_metrics.wape", "quality_report.holdout_metrics.WAPE", "holdout_metrics.wape", "holdout_metrics.WAPE", default=None)
    if wape is not None and wape <= 1.5:
        wape *= 100.0
    confidence = finite_or_none(scenario_result.get("confidence"))
    confidence_score = score_0_100((confidence if confidence is not None else 0.5) * 100.0)
    if wape is None:
        wape_score = confidence_score
        warnings.append("Holdout WAPE не найден: качество модели оценено по heuristic confidence.")
    else:
        wape_score = score_0_100(100.0 - min(float(wape), 100.0))
        if wape > 40:
            warnings.append("Holdout WAPE высокий: рекомендации требуют осторожной проверки.")
        if wape > 60:
            warnings.append("Holdout WAPE >60: автоматическая рекомендация ограничена.")
    shape = results.get("shape_diagnostics", {}) or {}
    std_ratio = finite_or_none(shape.get("std_ratio_final"))
    shape_score = score_0_100(100.0 - abs((std_ratio if std_ratio is not None else 1.0) - 1.0) * 100.0)
    score = 0.55 * wape_score + 0.30 * confidence_score + 0.15 * shape_score
    if bool(shape.get("shape_quality_low", False)):
        score = min(score, 55.0)
        warnings.append("Модель плохо повторяет динамику: model quality ограничен.")
    return score_0_100(score), {"wape": wape, "confidence": confidence, "std_ratio_final": std_ratio}, warnings


def _range_score(target: float | None, s: pd.Series) -> Tuple[float, bool, float]:
    clean = s.dropna()
    if target is None or len(clean) == 0:
        return 20.0, False, math.inf
    lo = float(clean.min()); hi = float(clean.max())
    if lo <= target <= hi:
        return 95.0, True, 0.0
    span = max(hi - lo, abs((hi + lo) / 2.0), 1e-9)
    dist = min(abs(target - lo), abs(target - hi)) / span
    if dist <= 0.05:
        return 80.0, False, dist
    if dist <= 0.15:
        return 55.0, False, dist
    if dist <= 0.25:
        return 35.0, False, dist
    return 15.0, False, dist


def _factor_support(candidate: Dict[str, Any], df: pd.DataFrame, scenario_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str], List[str]]:
    action = str(candidate.get("action_type", ""))
    warnings: List[str] = []
    blockers: List[str] = []
    if action == "baseline":
        return 85.0, {"action_type": action}, [], []
    if len(df) == 0:
        return 0.0, {"action_type": action}, ["Нет истории для оценки поддержки фактора."], ["Нет исторической поддержки сценария."]
    if action == "price_change":
        price = _series(df, "price")
        uniq = int(price.dropna().round(4).nunique())
        mean = abs(float(price.dropna().mean())) if len(price.dropna()) else 0.0
        span_pct = float((price.max() - price.min()) / max(mean, 1e-9)) if len(price.dropna()) else 0.0
        target = _target(candidate)
        rscore, inside, _ = _range_score(target, price)
        base = 85.0 if uniq >= 7 and span_pct >= 0.12 else (62.0 if uniq >= 4 or span_pct >= 0.06 else 35.0)
        score = 0.65 * base + 0.35 * rscore
        sales = _series(df, "sales", "actual_sales", "demand")
        corr = None
        if len(price.dropna()) >= 10 and len(sales.dropna()) >= 10:
            try:
                # Time-trend demeaning: use residuals from a linear trend proxy before correlation.
                idx = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
                p_res = price - (price.cov(idx) / max(idx.var(ddof=0), 1e-9)) * (idx - idx.mean())
                s_res = sales - (sales.cov(idx) / max(idx.var(ddof=0), 1e-9)) * (idx - idx.mean())
                corr_val = float(p_res.corr(s_res))
                corr = corr_val if np.isfinite(corr_val) else None
            except Exception:
                corr = None
        model_features = candidate.get("metadata", {}).get("model_features") or candidate.get("metadata", {}).get("selected_features") or []
        model_uses_price = (not model_features) or "price" in model_features or "net_price" in model_features
        if not model_uses_price:
            score = min(score, 40.0)
            warnings.append("Фактор price не найден среди используемых моделью признаков: поддержка ограничена.")
        if corr is not None and safe_float(candidate.get("change_pct"), 0.0) > 0 and corr > 0.2:
            warnings.append("Историческая связь price-demand экономически необычна; возможны смешение факторов/сезонность.")
            score = min(score, 75.0)
        if uniq < 4 or span_pct < 0.06:
            warnings.append("Цена почти не менялась в истории: влияние цены слабо поддержано.")
            score = min(score, 40.0)
        return score_0_100(score), {"unique_price_count": uniq, "price_span_pct": span_pct, "target_inside_range": inside, "price_demand_trend_adjusted_corr": corr, "model_uses_price": model_uses_price}, warnings, blockers
    if action == "discount_change":
        disc = _series(df, "discount")
        if len(disc.dropna()) == 0:
            blockers.append("Нет исторической колонки discount для проверки скидки.")
            return 20.0, {}, warnings, blockers
        uniq = int(disc.dropna().round(4).nunique()); span = float(disc.max() - disc.min())
        nonzero_share = float((disc.fillna(0.0) > 0).mean())
        target = _target(candidate)
        similar = int((abs(disc - safe_float(target, 0.0)) <= 0.02).sum()) if target is not None else 0
        score = 80.0 if uniq >= 5 and span >= 0.05 and similar >= 5 else (55.0 if uniq >= 3 and span > 0.01 else 30.0)
        if score < 55:
            warnings.append("Скидка почти не менялась в истории: поддержка слабая.")
        return score, {"unique_discount_count": uniq, "discount_span": span, "nonzero_discount_share": nonzero_share, "similar_discount_observations": similar}, warnings, blockers
    if action == "promotion_change":
        promo = _series(df, "promotion", "promo")
        if len(promo.dropna()) == 0:
            blockers.append("Нет исторической колонки promotion для проверки промо.")
            return 20.0, {}, warnings, blockers
        positive = int((promo.fillna(0.0) > 0).sum())
        nonpromo = int((promo.fillna(0.0) <= 0).sum())
        share = float((promo.fillna(0.0) > 0).mean())
        score = 80.0 if 0.1 <= share <= 0.9 and positive >= 10 and nonpromo >= 10 else 35.0
        if score < 50:
            warnings.append("Промо почти всегда включено или выключено: влияние промо слабо поддержано.")
        return score, {"promo_positive_share": share, "promo_positive_observations": positive, "promo_zero_observations": nonpromo}, warnings, blockers
    if action == "demand_shock":
        warnings.append("Demand shock является ручной гипотезой, а не выученным причинным эффектом.")
        evidence = bool((candidate.get("metadata") or {}).get("external_evidence"))
        return (45.0 if not evidence else 60.0), {"manual_hypothesis": True, "external_evidence": evidence}, warnings, blockers
    if action == "stock_cap":
        if "stock" not in df.columns:
            warnings.append("Нет stock column: ограничения запасов нельзя надёжно проверить.")
            return 30.0, {}, warnings, blockers
        return 55.0, {"stock_zero_share": float((_series(df, "stock").fillna(0) <= 0).mean())}, warnings, blockers
    return 45.0, {"action_type": action}, warnings, blockers


def _scenario_support(candidate: Dict[str, Any], df: pd.DataFrame, scenario_result: Dict[str, Any], price_guardrail_mode: str | None = None) -> Tuple[float, Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = []
    blockers: List[str] = []
    action = str(candidate.get("action_type", ""))
    target = _target(candidate)
    s = _series(df, "price") if action in {"price_change", "baseline"} else _series(df, "discount") if action == "discount_change" else pd.Series(dtype=float)
    range_score, inside, dist = _range_score(target, s) if len(s) else (75.0, True, 0.0)
    flags = {
        "price_clipped": bool(scenario_result.get("price_clipped") or scenario_result.get("clip_applied")),
        "ood_flag": bool(scenario_result.get("ood_flag")),
        "validation_ok": bool((scenario_result.get("validation_gate") or {}).get("ok", True)),
        "price_out_of_range": bool((scenario_result.get("effective_scenario") or {}).get("price_out_of_range", False)),
        "extrapolation_applied": bool(scenario_result.get("extrapolation_applied", False)),
    }
    score = range_score
    if flags["ood_flag"] or flags["price_out_of_range"] or flags["extrapolation_applied"]:
        score = min(score, 45.0)
        warnings.append("Сценарий находится вне исторического диапазона или требует экстраполяции.")
    guardrail_mode = str(price_guardrail_mode or scenario_result.get("price_guardrail_mode") or (candidate.get("metadata") or {}).get("price_guardrail_mode") or "safe_clip")
    if flags["price_clipped"]:
        score = min(score, 35.0)
        msg = "Сценарий был ограничен guardrail; результат относится к applied value, а не к requested value."
        warnings.append(msg)
        if guardrail_mode != "economic_extrapolation":
            blockers.append(msg)
    if not flags["validation_ok"]:
        score = min(score, 35.0)
        warnings.append("Validation gate сценария не пройден.")
    return score_0_100(score), {"inside_range": inside, "distance_outside_range": dist, **flags}, warnings, blockers



def _normal_positive_probability(mean: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if mean > 0 else 0.0
    z = mean / (sigma * math.sqrt(2.0))
    return float(0.5 * (1.0 + math.erf(z)))


def estimate_effect_uncertainty(
    results: Dict[str, Any],
    baseline_result: Dict[str, Any] | None,
    scenario_result: Dict[str, Any],
    horizon_days: int | None = None,
    profit_delta_pct: float | None = None,
    wape: float | None = None,
    factor_support_score: float = 60.0,
    scenario_support_score: float = 60.0,
) -> Dict[str, Any]:
    """Estimate decision-effect uncertainty without claiming formal significance.

    Prefer residual bootstrap when residuals are available; otherwise use a
    conservative WAPE-based approximation for the total-horizon effect.
    """
    mean = safe_float(profit_delta_pct, 0.0)
    horizon = max(int(horizon_days or (scenario_result or {}).get("horizon_days") or 30), 1)
    residuals = results.get("holdout_residuals") or results.get("residuals") or results.get("quality_report", {}).get("holdout_residuals")
    try:
        residual_arr = np.asarray(residuals, dtype=float)
        residual_arr = residual_arr[np.isfinite(residual_arr)]
    except Exception:
        residual_arr = np.asarray([], dtype=float)
    if residual_arr.size >= 20:
        # Deterministic bootstrap quantiles: sample residual means with a fixed RNG.
        rng = np.random.default_rng(42)
        draws = rng.choice(residual_arr, size=(400, min(horizon, residual_arr.size)), replace=True).mean(axis=1)
        base_profit = _metric(baseline_result or {}) or 1.0
        noise_pct = draws / max(abs(base_profit), 1e-9) * 100.0
        dist = mean + noise_pct
        p10, p50, p90 = [float(np.percentile(dist, q)) for q in (10, 50, 90)]
        prob = float((dist > 0).mean())
        sigma_pct = float(np.std(dist, ddof=0))
        method = "residual_bootstrap"
    else:
        wape_pct = safe_float(wape, 35.0 if wape is None else wape)
        if wape_pct <= 1.5:
            wape_pct *= 100.0
        horizon_diversification = max(0.25, 1.0 / math.sqrt(float(horizon)))
        support_multiplier = 1.0 + max(0.0, 60.0 - factor_support_score) / 100.0 + max(0.0, 60.0 - scenario_support_score) / 100.0
        if bool(scenario_result.get("ood_flag")) or bool(scenario_result.get("extrapolation_applied")) or bool((scenario_result.get("effective_scenario") or {}).get("price_out_of_range")):
            support_multiplier += 0.35
        sigma_pct = max(2.0, wape_pct * horizon_diversification * support_multiplier)
        p10 = mean - 1.2816 * sigma_pct
        p50 = mean
        p90 = mean + 1.2816 * sigma_pct
        prob = _normal_positive_probability(mean, sigma_pct)
        method = "wape_approximation"
    effect_to_noise = safe_div(abs(mean), sigma_pct, 0.0)
    return {
        "method": method,
        "profit_delta_pct_mean": round(mean, 4),
        "profit_delta_pct_p10": round(float(p10), 4),
        "profit_delta_pct_p50": round(float(p50), 4),
        "profit_delta_pct_p90": round(float(p90), 4),
        "probability_profit_positive": round(float(prob), 4),
        "effect_to_noise_ratio": round(float(effect_to_noise), 4),
        "expected_model_error_pct": round(float(sigma_pct), 4),
    }


def _economic(candidate: Dict[str, Any], scenario: Dict[str, Any], baseline: Dict[str, Any] | None, objective: str, preliminary_score: float, min_profit_uplift_pct: float, uncertainty: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any], List[str], List[str]]:
    warnings: List[str] = []
    blockers: List[str] = []
    baseline = baseline or {}
    scen_profit = _metric(scenario); base_profit = _metric(baseline)
    scen_demand = _metric(scenario, ("demand_total", "demand", "actual_sales")); base_demand = _metric(baseline, ("demand_total", "demand", "actual_sales"))
    scen_rev = _metric(scenario, ("revenue_total", "revenue")); base_rev = _metric(baseline, ("revenue_total", "revenue"))
    profit_delta_pct = safe_pct_delta(scen_profit, base_profit)
    demand_delta_pct = safe_pct_delta(scen_demand, base_demand)
    revenue_delta_pct = safe_pct_delta(scen_rev, base_rev)
    margin_delta_pp = safe_float(scenario.get("margin_pct"), 0.0) - safe_float(baseline.get("margin_pct"), 0.0)
    profit_valid = scen_profit is not None and base_profit is not None
    if objective == "profit" and not profit_valid:
        blockers.append("Прибыль не рассчитана валидно: оптимизация по profit недоступна.")
    cost_proxied = bool(candidate.get("metadata", {}).get("cost_proxied") or scenario.get("cost_proxied") or baseline.get("cost_proxied"))
    if cost_proxied:
        warnings.append("Прибыль оценочная: себестоимость проксирована.")
    if objective == "profit" and revenue_delta_pct > 0 and profit_delta_pct < 0:
        blockers.append("Выручка растёт, но прибыль падает: нельзя рекомендовать как profit action.")
    if objective in {"revenue", "demand"} and profit_valid:
        if profit_delta_pct < -5.0:
            blockers.append("Прибыль падает более чем на 5%: нельзя рекомендовать даже при цели revenue/demand.")
        elif profit_delta_pct < 0.0:
            warnings.append("Прибыль снижается: решение допустимо только как controlled_test, не как auto recommendation.")
    if demand_delta_pct < -30:
        warnings.append("Спрос падает более чем на 30%: высокий бизнес-риск.")
    if objective == "profit" and margin_delta_pp < -2:
        warnings.append("Маржа падает при цели profit: economic consistency снижена.")
    if str(candidate.get("action_type")) == "discount_change" and safe_float(candidate.get("change_pct"), 0) > 0 and profit_delta_pct < 0:
        blockers.append("Скидка увеличивается, но прибыль падает.")
    if str(candidate.get("action_type")) == "price_change" and safe_float(candidate.get("change_pct"), 0) > 0 and demand_delta_pct > 0:
        warnings.append("Рост спроса при росте цены может быть следствием сезонности/промо/смешения факторов; нужен тест.")
    uncertainty = uncertainty or {}
    risk_factor = (100.0 - preliminary_score) / 100.0
    fallback_haircut_pct = max(2.0, abs(profit_delta_pct) * 0.5 * risk_factor)
    conservative_profit_delta_pct = safe_float(uncertainty.get("profit_delta_pct_p10"), profit_delta_pct - fallback_haircut_pct)
    risk_haircut_pct = max(0.0, profit_delta_pct - conservative_profit_delta_pct)
    if objective == "profit":
        if profit_delta_pct <= 0:
            score = 20.0
        elif conservative_profit_delta_pct < min_profit_uplift_pct:
            score = 55.0
            warnings.append("Рост прибыли исчезает или мал после uncertainty/risk adjustment.")
        else:
            score = min(100.0, 65.0 + conservative_profit_delta_pct * 3.0)
    elif objective == "revenue":
        score = min(100.0, max(20.0, 55.0 + revenue_delta_pct * 2.0))
    elif objective == "demand":
        score = min(100.0, max(20.0, 55.0 + demand_delta_pct * 2.0))
    else:
        score = preliminary_score
    if objective == "profit" and abs(profit_delta_pct) < safe_float(uncertainty.get("expected_model_error_pct"), 0.0):
        warnings.append("Эффект меньше шума модели: можно рассматривать только тест, не автоматическое внедрение.")
        score = min(score, 60.0)
    if objective == "profit" and safe_float(uncertainty.get("probability_profit_positive"), 1.0) < 0.75:
        warnings.append("Вероятность положительного profit uplift ниже 75% по uncertainty estimate.")
        score = min(score, 62.0)
    level = "negative" if conservative_profit_delta_pct <= 0 and objective == "profit" else ("strong" if conservative_profit_delta_pct >= min_profit_uplift_pct * 2 else "moderate" if conservative_profit_delta_pct >= min_profit_uplift_pct else "weak")
    details = {"profit_delta_pct": profit_delta_pct, "conservative_profit_delta_pct": conservative_profit_delta_pct, "risk_haircut_pct": risk_haircut_pct, "demand_delta_pct": demand_delta_pct, "revenue_delta_pct": revenue_delta_pct, "margin_delta_pp": margin_delta_pp, "cost_proxied": cost_proxied, "uncertainty": uncertainty, "level": level, "reasons": []}
    return score_0_100(score), details, warnings, blockers


def _status(score: float, risk: str, blockers: List[str], candidate: Dict[str, Any], scenario_meta: Dict[str, Any], wape: float | None) -> str:
    if blockers:
        return "not_recommended"
    if str(candidate.get("action_type")) == "demand_shock" and not bool(candidate.get("metadata", {}).get("external_evidence")):
        return "experimental_only"
    if scenario_meta.get("distance_outside_range", 0.0) > 0.25:
        return "experimental_only"
    if wape is not None and wape > 60:
        return "experimental_only"
    if score >= 80 and risk == "low":
        return "recommended"
    if score >= 65:
        return "test_recommended"
    if score >= 45:
        return "experimental_only"
    return "not_recommended"


def evaluate_decision_reliability(results: dict, trained_bundle: dict, candidate: dict, scenario_result: dict, baseline_result: dict | None = None, objective: str = "profit", min_profit_uplift_pct: float = 3.0, price_guardrail_mode: str | None = None) -> dict:
    warnings: List[str] = []
    blockers: List[str] = []
    df = _history(results or {}, trained_bundle or {})
    data_score, data_meta, w = _data_quality(results or {}, trained_bundle or {}); warnings += w
    model_score, model_meta, w = _model_quality(results or {}, scenario_result or {}); warnings += w
    factor_score, factor_meta, w, b = _factor_support(candidate or {}, df, scenario_result or {}); warnings += w; blockers += b
    scenario_score, scenario_meta, w, b = _scenario_support(candidate or {}, df, scenario_result or {}, price_guardrail_mode); warnings += w; blockers += b
    validation_score = 80.0 if not blockers else 35.0
    preliminary = 0.20 * data_score + 0.20 * model_score + 0.20 * factor_score + 0.15 * scenario_score + 0.10 * validation_score + 0.15 * 60.0
    raw_profit_delta = safe_pct_delta(_metric(scenario_result or {}), _metric(baseline_result or {}))
    uncertainty = estimate_effect_uncertainty(results or {}, baseline_result or {}, scenario_result or {}, horizon_days=(candidate.get("metadata") or {}).get("horizon_days") if isinstance(candidate, dict) else None, profit_delta_pct=raw_profit_delta, wape=model_meta.get("wape"), factor_support_score=factor_score, scenario_support_score=scenario_score)
    econ_score, econ_meta, w, b = _economic(candidate or {}, scenario_result or {}, baseline_result or {}, objective, preliminary, min_profit_uplift_pct, uncertainty=uncertainty); warnings += w; blockers += b
    score = 0.20 * data_score + 0.20 * model_score + 0.20 * factor_score + 0.15 * scenario_score + 0.15 * econ_score + 0.10 * validation_score
    if scenario_meta.get("ood_flag") or scenario_meta.get("price_out_of_range") or scenario_meta.get("extrapolation_applied"):
        score -= 10.0
    if model_meta.get("wape") is not None and model_meta["wape"] > 40:
        score -= 8.0
    if data_meta.get("flat_sales"):
        score = min(score, 64.0)
    if blockers:
        score = min(score, 44.0)
    if econ_meta.get("cost_proxied"):
        score = min(score, 75.0)
    score = score_0_100(score)
    risk = "high" if score < 55 or scenario_score < 45 or econ_meta.get("demand_delta_pct", 0.0) < -30 else ("low" if score >= 80 and scenario_score >= 75 and not warnings else "medium")
    if str((candidate or {}).get("action_type")) == "demand_shock" and not bool(((candidate or {}).get("metadata") or {}).get("external_evidence")):
        risk = "high" if score < 70 else "medium"
    if risk == "high":
        econ_meta["risk_haircut_pct"] = max(float(econ_meta.get("risk_haircut_pct", 0.0)), 5.0)
        econ_meta["conservative_profit_delta_pct"] = float(econ_meta.get("profit_delta_pct", 0.0)) - float(econ_meta["risk_haircut_pct"])
    component_avg = np.mean([data_score, model_score, factor_score, scenario_score])
    wape = model_meta.get("wape")
    if data_meta.get("history_days", 0) == 0 or factor_score < 25 or not baseline_result:
        support_level = "insufficient"
    elif data_score >= 70 and model_score >= 60 and factor_score >= 70 and scenario_score >= 70 and data_meta.get("nonzero_days", 0) >= 90 and ((wape is not None and wape <= 35) or safe_float(model_meta.get("confidence"), 0) >= 0.65):
        support_level = "strong"
    elif component_avg >= 55 and not blockers:
        support_level = "moderate"
    else:
        support_level = "weak"
    status = _status(score, risk, blockers, candidate or {}, scenario_meta, wape)
    if data_meta.get("flat_sales") and status == "recommended":
        status = "test_recommended"
    if objective in {"revenue", "demand"} and econ_meta.get("profit_delta_pct", 0.0) < 0 and status == "recommended":
        status = "test_recommended"
    if objective == "profit" and (econ_meta.get("conservative_profit_delta_pct", 0.0) < min_profit_uplift_pct or safe_float(econ_meta.get("uncertainty", {}).get("probability_profit_positive"), 1.0) < 0.75) and status == "recommended":
        status = "test_recommended"
    if objective == "profit" and abs(econ_meta.get("profit_delta_pct", 0.0)) < safe_float(econ_meta.get("uncertainty", {}).get("expected_model_error_pct"), 0.0) and status in {"recommended", "test_recommended"}:
        status = "experimental_only"
    if objective == "profit" and econ_meta.get("conservative_profit_delta_pct", 0.0) <= 0 and status in {"recommended", "test_recommended"}:
        status = "experimental_only"
    label = "high" if score >= 80 else "medium" if score >= 60 else "low"
    positives = []
    if scenario_score >= 70: positives.append("Сценарий находится внутри/рядом с историческим диапазоном.")
    if econ_meta.get("conservative_profit_delta_pct", -999) >= min_profit_uplift_pct: positives.append("Ожидаемая прибыль растёт после risk adjustment.")
    if factor_score >= 70: positives.append("Фактор имеет достаточную историческую вариативность.")
    negatives = list(dict.fromkeys(warnings + blockers))
    return {
        "score": round(score, 2), "label": label, "risk_level": risk, "decision_status": status,
        "statistical_support": {"level": support_level, "reasons": negatives[:6]},
        "economic_significance": econ_meta,
        "components": {"data_quality": round(data_score, 2), "model_quality": round(model_score, 2), "factor_support": round(factor_score, 2), "scenario_support": round(scenario_score, 2), "economic_consistency": round(econ_score, 2), "validation_readiness": round(validation_score, 2)},
        "component_details": {"data_quality": data_meta, "model_quality": model_meta, "factor_support": factor_meta, "scenario_support": scenario_meta},
        "reasons_positive": positives,
        "reasons_negative": negatives,
        "warnings": list(dict.fromkeys(warnings)),
        "blockers": list(dict.fromkeys(blockers)),
    }
