from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_adapter import build_daily_from_transactions_scoped
from recommendation import build_business_recommendation

from .core import CONFIG, assess_data_quality
from .v1_elasticity import estimate_v1_elasticity, evaluate_price_signal
from .v1_features import build_v1_feature_matrix, derive_v1_feature_spec
from .v1_forecast import predict_v1_baseline_log, train_v1_baseline_model
from .v1_optimizer import recommend_v1_price_horizon, simulate_v1_horizon_profit


def _safe_split_sizes(n: int) -> Tuple[int, int]:
    train_end = max(3, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n)
    return train_end, val_end


def _eval_wape(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.abs(actual).sum())
    if denom <= 1e-9:
        return 100.0
    return float(np.abs(actual - pred).sum() / denom * 100.0)


def _make_future_dates(last_date: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")})


def _compute_confidence(wape: float, quality_cap: float, can_recommend: bool) -> float:
    raw_conf = float(np.exp(-max(float(wape), 0.0) / 30.0))
    signal_cap = 1.0 if can_recommend else 0.55
    return float(np.clip(raw_conf * float(quality_cap) * signal_cap, 0.0, 1.0))


def _build_base_ctx(daily_base: pd.DataFrame, target_category: str, target_sku: str) -> Dict[str, Any]:
    return {
        "price": float(daily_base["price"].iloc[-1]),
        "cost": float(daily_base["cost"].iloc[-1]),
        "discount": float(daily_base.get("discount", daily_base.get("discount_rate", 0.0)).iloc[-1]),
        "promotion": float(daily_base.get("promotion", pd.Series([0.0])).iloc[-1]),
        "stock": float(daily_base.get("stock", pd.Series([0.0])).iloc[-1]),
        "freight_value": float(daily_base.get("freight_value", pd.Series([0.0])).iloc[-1]),
        "review_score": float(daily_base.get("review_score", pd.Series([4.5])).iloc[-1]),
        "reviews_count": float(daily_base.get("reviews_count", pd.Series([0.0])).iloc[-1]),
        "category": target_category,
        "product_id": target_sku,
    }


def can_recommend_price(price_signal: Dict[str, Any], holdout_metrics: Dict[str, Any], data_quality: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if int(price_signal.get("history_days", 0)) < CONFIG["MIN_COVERAGE_DAYS"]:
        reasons.append("history_days_below_min")
    if float(price_signal.get("total_sales", 0.0)) < CONFIG["MIN_TOTAL_SALES"]:
        reasons.append("total_sales_below_min")
    if int(price_signal.get("unique_prices", 0)) < CONFIG["MIN_UNIQUE_PRICES"]:
        reasons.append("unique_prices_below_min")
    if int(price_signal.get("price_changes", 0)) < CONFIG["MIN_PRICE_CHANGES"]:
        reasons.append("price_changes_below_min")
    if float(price_signal.get("rel_price_span", 0.0)) < CONFIG["MIN_REL_PRICE_SPAN"]:
        reasons.append("rel_price_span_below_min")
    if float(holdout_metrics.get("wape", 100.0)) > 35:
        reasons.append("holdout_wape_above_threshold")
    if str(data_quality.get("level", "unknown")) in {"poor", "unavailable"}:
        reasons.append("data_quality_too_low")
    return len(reasons) == 0, reasons


def run_full_pricing_analysis_universal_v1(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    objective_mode: str = "maximize_profit",
    horizon_days: int = 30,
    risk_lambda: float = 0.7,
) -> Dict[str, Any]:
    txn = normalized_txn.copy()
    daily_base = build_daily_from_transactions_scoped(txn, target_sku, target_category)
    daily_base = build_v1_feature_matrix(daily_base).dropna(subset=["sales", "price", "log_sales"]).reset_index(drop=True)
    if len(daily_base) < 10:
        raise ValueError("Слишком мало данных для v1 анализа.")

    feature_spec = derive_v1_feature_spec(daily_base)
    n = len(daily_base)
    train_end, val_end = _safe_split_sizes(n)
    train_df = daily_base.iloc[:train_end].copy()
    test_df = daily_base.iloc[val_end:].copy()

    baseline_models = train_v1_baseline_model(train_df, feature_spec, small_mode=len(train_df) < 120)
    if len(test_df):
        holdout_frame = test_df.reindex(columns=feature_spec["baseline_features"], fill_value=np.nan)
        pred_log, _ = predict_v1_baseline_log(holdout_frame, baseline_models, feature_spec)
        pred_test = np.expm1(pred_log)
        holdout_metrics = {"wape": _eval_wape(test_df["sales"], pd.Series(pred_test, index=test_df.index))}
    else:
        holdout_metrics = {"wape": 100.0}

    price_signal = evaluate_price_signal(daily_base)
    elasticity_info = estimate_v1_elasticity(train_df)

    history_days = int((daily_base["date"].max() - daily_base["date"].min()).days + 1)
    missing_share = float(txn.isna().mean().mean()) if len(txn) else 1.0
    data_quality = assess_data_quality(history_days, len(daily_base), missing_share, float(holdout_metrics.get("wape", 100.0)))
    can_rec, rec_reasons = can_recommend_price(price_signal, holdout_metrics, data_quality)

    base_ctx = _build_base_ctx(daily_base, target_category, target_sku)
    latest_row = dict(base_ctx)

    rec = recommend_v1_price_horizon(
        latest_row,
        baseline_models,
        daily_base,
        base_ctx,
        elasticity_info["elasticity_by_month"],
        float(elasticity_info["pooled_elasticity"]),
        feature_spec,
        n_days=int(horizon_days),
        objective_mode=objective_mode,
        risk_lambda=float(risk_lambda),
        can_recommend=can_rec,
    )

    future_dates = _make_future_dates(pd.Timestamp(daily_base["date"].max()), int(horizon_days))
    current_sim = simulate_v1_horizon_profit(
        latest_row,
        base_ctx["price"],
        future_dates,
        baseline_models,
        daily_base,
        base_ctx,
        elasticity_info["elasticity_by_month"],
        float(elasticity_info["pooled_elasticity"]),
        feature_spec,
        risk_lambda=float(risk_lambda),
    )
    optimal_price = float(rec.get("best_price", base_ctx["price"]))
    optimal_sim = simulate_v1_horizon_profit(
        latest_row,
        optimal_price,
        future_dates,
        baseline_models,
        daily_base,
        base_ctx,
        elasticity_info["elasticity_by_month"],
        float(elasticity_info["pooled_elasticity"]),
        feature_spec,
        risk_lambda=float(risk_lambda),
    )

    confidence = _compute_confidence(float(holdout_metrics.get("wape", 100.0)), float(data_quality.get("confidence_cap", 0.5)), can_rec)

    reason_hints = {
        "key_driver_positive": "Лучшая цена выбрана по adjusted profit",
        "key_driver_negative": "Сдерживающий фактор — риск падения объёма",
        "reason_why_this_scenario_wins": "Ценовой сигнал слабый, решение только пилотное" if not can_rec else "Лучшая цена выбрана по adjusted profit",
    }

    biz_rec = build_business_recommendation(
        current_price=float(base_ctx["price"]),
        recommended_price=float(optimal_price if can_rec else base_ctx["price"]),
        current_profit=float(current_sim["total_profit"]),
        recommended_profit=float(optimal_sim["total_profit"] if can_rec else current_sim["total_profit"]),
        current_revenue=float(current_sim["total_revenue"]),
        recommended_revenue=float(optimal_sim["total_revenue"] if can_rec else current_sim["total_revenue"]),
        current_volume=float(current_sim["total_volume"]),
        recommended_volume=float(optimal_sim["total_volume"] if can_rec else current_sim["total_volume"]),
        confidence=confidence,
        elasticity=float(elasticity_info["pooled_elasticity"]),
        history_days=history_days,
        data_quality=data_quality,
        base_ctx=base_ctx,
        reason_hints=reason_hints,
    )
    structured_decision = biz_rec.get("structured", {}).get("decision_type", "hold")
    if structured_decision == "action":
        legacy_decision = "change_price"
    elif structured_decision == "test":
        legacy_decision = "pilot_change"
    elif structured_decision == "hold":
        legacy_decision = "hold"
    else:
        legacy_decision = "no_decision"
    biz_rec["legacy_decision_type"] = legacy_decision
    if rec_reasons:
        biz_rec["decision_reasons"] = rec_reasons

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        daily_base.to_excel(writer, sheet_name="history", index=False)
        current_sim["daily"].to_excel(writer, sheet_name="baseline", index=False)
        optimal_sim["daily"].to_excel(writer, sheet_name="optimal", index=False)
        pd.DataFrame([holdout_metrics]).to_excel(writer, sheet_name="metrics", index=False)
    excel_buffer.seek(0)

    current_profit_raw = float(current_sim["total_profit"])
    best_profit_raw = float(current_sim["total_profit"] if not can_rec else optimal_sim["total_profit"])
    current_profit_adjusted = float(current_sim["adjusted_profit"])
    best_profit_adjusted = float(current_sim["adjusted_profit"] if not can_rec else optimal_sim["adjusted_profit"])

    return {
        "daily": daily_base,
        "recommendation": rec,
        "forecast_current": current_sim["daily"],
        "forecast_optimal": optimal_sim["daily"],
        "profit_curve": pd.DataFrame(rec.get("results", [])),
        "holdout_metrics": pd.DataFrame([holdout_metrics]),
        "elasticity_map": elasticity_info["elasticity_by_month"],
        "current_price": float(base_ctx["price"]),
        "best_price": float(base_ctx["price"] if not can_rec else optimal_price),
        "current_profit": current_profit_raw,
        "best_profit": best_profit_raw,
        "current_profit_adjusted": current_profit_adjusted,
        "best_profit_adjusted": best_profit_adjusted,
        "current_revenue": float(current_sim["total_revenue"]),
        "best_revenue": float(current_sim["total_revenue"] if not can_rec else optimal_sim["total_revenue"]),
        "current_volume": float(current_sim["total_volume"]),
        "best_volume": float(current_sim["total_volume"] if not can_rec else optimal_sim["total_volume"]),
        "profit_lift_pct": float(((best_profit_raw - current_profit_raw) / max(abs(current_profit_raw), 1e-9) * 100.0) if can_rec else 0.0),
        "data_quality": data_quality,
        "excel_buffer": excel_buffer,
        "business_recommendation": biz_rec,
        "_trained_bundle": {
            "baseline_models": baseline_models,
            "feature_spec": feature_spec,
            "daily_base": daily_base,
            "base_ctx": base_ctx,
            "latest_row": latest_row,
            "future_dates": future_dates,
            "elasticity_map": elasticity_info["elasticity_by_month"],
            "pooled_elasticity": float(elasticity_info["pooled_elasticity"]),
            "price_signal": price_signal,
            "confidence": confidence,
            "data_quality": data_quality,
            "can_recommend_price": can_rec,
            "recommendation_reasons": rec_reasons,
            "risk_lambda": float(risk_lambda),
        },
    }
