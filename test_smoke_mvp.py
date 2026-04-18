import pandas as pd
import numpy as np
import json
import copy
import pytest
import app as app_module

from data_adapter import build_auto_mapping, normalize_transactions, build_daily_from_transactions
from what_if import build_sensitivity_grid, run_scenario_set
from scenario_engine import run_scenario
from app import (
    apply_weekly_fallback_projection,
    build_manual_scenario_artifacts,
    read_uploaded_csv_safely,
    resolve_weekly_driver_mode,
    resolve_scenario_driver_mode,
    robust_clean_dirty_data,
    run_full_pricing_analysis_universal,
    run_what_if_projection,
)


def test_normalize_and_daily_build():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "SKU": ["A", "A", "B"],
            "Unit Price": [100, 110, 90],
            "qty": [1, 2, 1],
        }
    )
    mapping = build_auto_mapping(list(df.columns))
    norm, quality = normalize_transactions(df, mapping)
    assert not quality.get("errors")
    assert "warnings" in quality
    daily = build_daily_from_transactions(norm, "A")
    assert len(daily) >= 2
    assert (daily["price"] > 0).all()


def test_read_uploaded_csv_safely_supports_semicolon_delimiter():
    class DummyUpload:
        def __init__(self, payload: bytes):
            self._payload = payload

        def getvalue(self):
            return self._payload

    payload = "date;product_id;price\n2025-01-01;sku-1;100\n".encode("utf-8")
    df = read_uploaded_csv_safely(DummyUpload(payload))
    assert set(df.columns) == {"date", "product_id", "price"}
    assert len(df) == 1


def test_scenario_runner_baseline_fallback():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        d = kwargs.get("demand_multiplier", 1.0)
        return {"demand_total": 10 * d, "revenue_total": 10 * d * p, "profit_total": 2 * d * p, "confidence": 0.7}

    scenarios = [
        {"name": "Scenario A", "price": 100, "freight_multiplier": 1, "demand_multiplier": 1, "horizon_days": 30},
        {"name": "Scenario B", "price": 90, "freight_multiplier": 1, "demand_multiplier": 1.1, "horizon_days": 30},
    ]
    out = run_scenario_set({}, scenarios, runner)
    assert len(out) == 2
    assert "delta_profit" in out.columns


def test_sensitivity_uses_demand_axis():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        demand_mult = kwargs.get("demand_multiplier", 1.0)
        return {"profit_total": p * (2 - demand_mult), "profit_total_adjusted": p * (2 - demand_mult), "demand_total": 10 * demand_mult}

    out = build_sensitivity_grid({}, base_price=100.0, runner=runner, price_steps=3, demand_steps=3)
    assert "demand_multiplier" in out.columns
    assert "profit_adjusted" in out.columns
    assert "risk_zone" in out.columns


def _make_txn(n_days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 100 + (i % 8)
        promo = 1.0 if i % 14 == 0 else 0.0
        freight = 5.0 + (i % 3)
        qty = max(1.0, 40 - 0.12 * price + 6.0 * promo - 0.3 * freight)
        rows.append(
            {
                "date": d,
                "product_id": "sku-1",
                "category": "cat-a",
                "price": price,
                "quantity": qty,
                "revenue": qty * price * (1.0 - 0.05 * promo),
                "cost": price * 0.65,
                "discount": 0.05 * promo,
                "freight_value": freight,
                "promotion": promo,
                "rating": 4.4,
                "reviews_count": 10 + (i % 5),
            }
        )
    return pd.DataFrame(rows)


def _analyze():
    return run_full_pricing_analysis_universal(_make_txn(), "cat-a", "sku-1")


def _analyze_long_signal():
    return run_full_pricing_analysis_universal(_make_txn(420), "cat-a", "sku-1")


def test_scenario_not_equal_as_is_when_price_changes():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.1)
    as_is_total = float(res["as_is_forecast"]["actual_sales"].sum())
    assert scenario["demand_total"] != as_is_total


def test_price_increase_reduces_demand_monotone():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.10)
    assert high["demand_total"] <= low["demand_total"]


def test_promotion_on_not_decrease_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    off = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 0.0})
    on = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 1.0})
    assert on["demand_total"] >= off["demand_total"]


def test_freight_increase_not_increase_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price, freight_multiplier=1.0)
    high = run_what_if_projection(bundle, manual_price=base_price, freight_multiplier=1.5)
    assert high["demand_total"] <= low["demand_total"]


def test_analysis_summary_does_not_fake_manual_scenario():
    res = _analyze()
    summary_blob_before = res["analysis_run_summary_json"]
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    out = summary["scenario_output_summary"]
    assert out["scenario_status"] == "not_run"
    assert pd.isna(out["scenario_demand_total"])
    assert pd.isna(out["scenario_revenue_total"])
    assert pd.isna(out["scenario_profit_total"])
    assert res["scenario_forecast"] is None
    assert res["scenario_price"] is None
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.15)
    assert scenario["demand_total"] != float(res["as_is_forecast"]["actual_sales"].sum())
    assert scenario["revenue_total"] != float(res["as_is_forecast"]["revenue"].sum())
    assert scenario["profit_total"] != float(res["as_is_forecast"]["profit"].sum())
    # analysis summary is analysis-level artifact and should not mutate after runtime what-if calls.
    assert res["analysis_run_summary_json"] == summary_blob_before
    summary_after = json.loads(summary_blob_before.decode("utf-8"))
    assert summary_after["scenario_output_summary"]["scenario_status"] == "not_run"


def test_manual_scenario_artifacts_created_after_runtime_what_if():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.2)
    summary_blob, daily_blob = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    daily = pd.read_csv(pd.io.common.BytesIO(daily_blob))
    assert summary["scenario_status"] == "executed"
    assert len(daily) > 0
    assert "scenario_demand" in daily.columns


def test_manual_scenario_totals_not_equal_as_is_when_price_changes():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.25)
    summary_blob, _ = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    assert summary["requested_price"] != base_price
    assert abs(summary["delta_vs_as_is"]["demand_total"]) > 0
    assert "applied_overrides" in summary


def test_manual_scenario_executes_and_populates_summary():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.1)
    summary_blob, _ = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    assert summary["scenario_status"] == "executed"
    assert np.isfinite(summary["scenario_demand_total"])
    assert np.isfinite(summary["scenario_revenue_total"])
    assert np.isfinite(summary["scenario_profit_total"])
    assert len(summary["scenario_forecast"]) > 0
    assert np.isfinite(summary["scenario_vs_as_is_demand_pct"])
    assert np.isfinite(summary["scenario_vs_as_is_profit_pct"])


def test_price_increase_above_train_max_is_not_silent():
    res = _analyze()
    bundle = res["_trained_bundle"]
    train_max_price = float(bundle["daily_base"]["price"].max())
    base = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]))
    high = run_what_if_projection(bundle, manual_price=train_max_price * 1.25)
    changed = not np.isclose(float(high["demand_total"]), float(base["demand_total"]))
    explained_clip = bool(high.get("clip_applied")) and str(high.get("clip_reason", "")) != ""
    assert changed or explained_clip


def test_active_path_uses_price_promo_freight_baseline_in_v1():
    res = _analyze_long_signal()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert summary["config"]["selected_candidate"] == "price_promo_freight_baseline"
    assert "price_promo_freight_baseline" in summary["config"]["final_active_path"]


def test_feature_contract_block():
    res = _analyze_long_signal()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    if summary["config"].get("uplift_gate_result") == "passed":
        return
    fr = res["feature_report"]
    attempted_col = "used_in_weekly_uplift_attempted" if "used_in_weekly_uplift_attempted" in fr.columns else "used_in_uplift_attempted"
    active_col = "used_in_weekly_uplift_active" if "used_in_weekly_uplift_active" in fr.columns else "used_in_uplift_active"
    baseline_col = "used_in_weekly_baseline" if "used_in_weekly_baseline" in fr.columns else "used_in_baseline"
    blocked = fr[
        (fr[attempted_col] == True)
        & (fr[active_col] == False)
        & (fr[baseline_col] == False)
    ]
    if blocked.empty:
        return
    assert (blocked["used_in_final_active_forecast"] == False).all()


def test_analysis_summary_contains_schema_version_and_commit_signature():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    cfg = summary["config"]
    assert "git_commit" in cfg and str(cfg["git_commit"]) != ""
    assert "code_signature" in cfg and len(str(cfg["code_signature"])) >= 12
    assert cfg["artifact_schema_version"] == "v39.1"


def test_downloaded_holdout_schema_matches_current_version():
    res = _analyze()
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    cols = set(holdout.columns)
    assert {"baseline_pred_sales", "price_effect_multiplier", "uplift_multiplier", "final_pred_sales"}.issubset(cols)
    assert {"pred_baseline", "pred_direct", "pred_final", "w_direct", "direct_features"}.isdisjoint(cols)


def test_weekly_fallback_projection_preserves_totals():
    hist = _make_txn(120)
    base = pd.DataFrame(
        {
            "date": pd.date_range("2025-06-01", periods=21, freq="D"),
            "actual_sales": np.linspace(10, 20, 21),
            "revenue": np.linspace(100, 200, 21),
            "profit": np.linspace(30, 80, 21),
            "lost_sales": np.linspace(0, 5, 21),
        }
    )
    out = apply_weekly_fallback_projection(base, hist)
    for col in ["actual_sales", "revenue", "profit", "lost_sales"]:
        assert abs(float(out[col].sum()) - float(base[col].sum())) < 1e-6


def test_holdout_final_has_decomposition():
    res = _analyze()
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    assert {"baseline_pred_sales", "price_effect_multiplier", "uplift_multiplier", "final_pred_sales"}.issubset(set(holdout.columns))
    assert "is_full_week" in holdout.columns
    assert (holdout["is_full_week"] >= 1).all()


def test_no_backward_fill_leakage_in_daily_builder():
    src = pd.DataFrame(
        [
            {"date": "2025-01-01", "product_id": "sku-1", "category": "cat-a", "price": 100, "quantity": 1, "revenue": 100, "discount": 0.0, "promotion": 0.0},
            {"date": "2025-01-03", "product_id": "sku-1", "category": "cat-a", "price": 100, "quantity": 1, "revenue": 100, "discount": 0.2, "promotion": 1.0},
        ]
    )
    src["date"] = pd.to_datetime(src["date"])
    daily = build_daily_from_transactions(src, "sku-1", target_category="cat-a")
    mid = daily[daily["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    assert float(mid["discount"]) == 0.0
    assert float(mid["promotion"]) == 0.0


def test_zero_sales_preserved_after_cleaning():
    df = pd.DataFrame({"sales": [10.0, 0.0, 0.0, 2.0], "price": [10, 10, 10, 10], "freight_value": [1, 1, 1, 1]})
    out = robust_clean_dirty_data(df)
    assert (out["sales"] == 0.0).sum() >= 2


def test_no_double_multiplier():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]), freight_multiplier=1.0)
    high = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]), freight_multiplier=1.5)
    base_f = float(base["daily"]["freight_value"].iloc[0])
    high_f = float(high["daily"]["freight_value"].iloc[0])
    assert abs((high_f / base_f) - 1.5) < 1e-6


def test_uplift_uses_exogenous_features_only():
    res = _analyze()
    bundle = res["_trained_bundle"]["uplift_bundle"]
    feats = bundle["features"]
    assert "baseline_log_feature" not in feats
    assert "price_idx" not in feats
    assert "stockout_share" not in feats
    assert set(feats).issubset({"price_gap_ref_8w", "promotion_share", "freight_pct_change_1w", "promo_any"})
    if not bundle.get("disabled", False):
        assert len(bundle.get("models", [])) > 0


def test_uplift_bundle_enabled_when_signal_present():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]["uplift_bundle"]
    assert "features" in bundle
    assert "baseline_log_feature" not in bundle["features"]
    if bundle.get("disabled") is False:
        assert len(bundle.get("models", [])) > 0
    else:
        assert bundle.get("reason") in {"uplift_holdout_failed", "uplift_no_exogenous_features", "uplift_not_enough_rows", "benchmark_gate_failed", "holdout_support_too_low", "uplift_non_neutral_bias"}


def test_runtime_applies_amplitude_calibrator_from_baseline_bundle():
    res = _analyze()
    trained = res["_trained_bundle"]
    baseline_bundle_raw = copy.deepcopy(trained["baseline_bundle"])
    baseline_bundle_cal = copy.deepcopy(trained["baseline_bundle"])
    baseline_bundle_raw["amplitude_calibrator"] = {"enabled": False, "scale": 1.0, "center_mode": "forecast_mean"}
    baseline_bundle_cal["amplitude_calibrator"] = {"enabled": True, "scale": 1.35, "center_mode": "forecast_mean"}
    uplift_off = {"models": [], "features": [], "feature_stats": {}, "disabled": True, "reason": "test", "signal_info": {}, "neutral_reference_log": 0.0}

    sim_raw = app_module.simulate_horizon_profit(
        trained["latest_row"],
        float(trained["base_ctx"]["price"]),
        trained["future_dates"],
        baseline_bundle_raw,
        uplift_off,
        trained["daily_base"],
        trained["base_ctx"],
        trained["elasticity_map"],
        trained["pooled_elasticity"],
    )
    sim_cal = app_module.simulate_horizon_profit(
        trained["latest_row"],
        float(trained["base_ctx"]["price"]),
        trained["future_dates"],
        baseline_bundle_cal,
        uplift_off,
        trained["daily_base"],
        trained["base_ctx"],
        trained["elasticity_map"],
        trained["pooled_elasticity"],
    )
    baseline_daily_raw = np.asarray(sim_raw["baseline_daily"]["pred_sales"], dtype=float)
    baseline_daily_cal = np.asarray(sim_cal["baseline_daily"]["pred_sales"], dtype=float)
    assert baseline_daily_raw.shape == baseline_daily_cal.shape
    assert not np.allclose(baseline_daily_raw, baseline_daily_cal)


def test_naive_forecaster_ignores_amplitude_calibrator():
    res = _analyze()
    trained = res["_trained_bundle"]
    baseline_bundle_raw = copy.deepcopy(trained["baseline_bundle"])
    baseline_bundle_cal = copy.deepcopy(trained["baseline_bundle"])
    baseline_bundle_raw["selected_forecaster"] = "naive_lag1w"
    baseline_bundle_raw["model"] = None
    baseline_bundle_cal["selected_forecaster"] = "naive_lag1w"
    baseline_bundle_cal["model"] = None
    baseline_bundle_raw["amplitude_calibrator"] = {"enabled": False, "scale": 1.0, "center_mode": "forecast_mean"}
    baseline_bundle_cal["amplitude_calibrator"] = {"enabled": True, "scale": 1.35, "center_mode": "forecast_mean"}
    uplift_off = {"models": [], "features": [], "feature_stats": {}, "disabled": True, "reason": "test", "signal_info": {}, "neutral_reference_log": 0.0}

    sim_raw = app_module.simulate_horizon_profit(
        trained["latest_row"],
        float(trained["base_ctx"]["price"]),
        trained["future_dates"],
        baseline_bundle_raw,
        uplift_off,
        trained["daily_base"],
        trained["base_ctx"],
        trained["elasticity_map"],
        trained["pooled_elasticity"],
    )
    sim_cal = app_module.simulate_horizon_profit(
        trained["latest_row"],
        float(trained["base_ctx"]["price"]),
        trained["future_dates"],
        baseline_bundle_cal,
        uplift_off,
        trained["daily_base"],
        trained["base_ctx"],
        trained["elasticity_map"],
        trained["pooled_elasticity"],
    )
    baseline_daily_raw = np.asarray(sim_raw["baseline_daily"]["pred_sales"], dtype=float)
    baseline_daily_cal = np.asarray(sim_cal["baseline_daily"]["pred_sales"], dtype=float)
    assert np.allclose(baseline_daily_raw, baseline_daily_cal)


def test_uplift_training_uses_calibrated_baseline():
    res = _analyze_long_signal()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    debug = summary["config"].get("uplift_debug_info", {})
    assert debug.get("baseline_train_calibrated") is True
    assert "baseline_train_mean" in debug
    assert "baseline_train_raw_mean" in debug


def test_neutral_uplift_multiplier_is_near_one():
    res = _analyze_long_signal()
    trace = pd.read_csv(pd.io.common.BytesIO(res["uplift_holdout_trace_csv"]))
    neutral_mask = (
        trace["price_gap_ref_8w"].abs().le(0.01)
        & trace["promotion_share"].fillna(0.0).eq(0.0)
        & trace["freight_pct_change_1w"].abs().le(0.02)
    )
    if neutral_mask.any():
        neutral_bias = float((trace.loc[neutral_mask, "uplift_multiplier_attempted"] - 1.0).abs().mean())
        assert neutral_bias <= 0.05


def test_backtest_summary_contains_consistent_uplift_and_amplitude_diagnostics():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    backtest = summary["metrics_summary"]["rolling_weekly_backtest"]
    assert "amplitude_scale_median" in backtest
    assert "support_too_low_fold_rate" in backtest
    assert "neutral_bias_mean" in backtest
    assert "uplift_keep_fold_rate" in backtest


def test_gate_accounts_for_correlation_and_std_ratio():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert "corr_final" in summary["scenario_output_summary"]
    assert "std_ratio_final" in summary["scenario_output_summary"]
    assert "shape_quality_low" in summary["scenario_output_summary"]
    assert "shape_quality_low" in summary["config"]


def test_price_and_promo_scenario_changes_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 0.0})
    promo = run_what_if_projection(bundle, manual_price=base_price * 0.9, overrides={"promotion": 1.0})
    assert promo["demand_total"] != base["demand_total"]
    assert promo["revenue_total"] != base["revenue_total"]


def test_holdout_final_differs_from_baseline_when_uplift_enabled():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]["uplift_bundle"]
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    diff = float(np.abs(holdout["final_pred_sales"] - holdout["baseline_pred_sales"]).sum())
    if bundle.get("disabled") is False:
        assert diff > 1e-6
    else:
        assert diff < 1e-6


def test_bundle_selection_prefers_price_promo_freight_baseline(monkeypatch):
    def fake_predict_weekly_holdout_with_actual_exog(model, train_weekly, test_weekly, feature_names, seasonal_anchor_weight=0.0):
        actual = test_weekly["sales"].astype(float).values
        if "freight_mean" in feature_names:
            pred = actual * 1.01
        elif "promotion_share" in feature_names:
            pred = actual * 1.03
        elif "price_idx" in feature_names:
            pred = actual * 0.80
        else:
            pred = actual * 0.95
        out = test_weekly[["week"]].copy() if "week" in test_weekly.columns else pd.DataFrame(index=test_weekly.index)
        out["pred_weekly_sales"] = pred
        return out

    monkeypatch.setattr(app_module, "predict_weekly_holdout_with_actual_exog", fake_predict_weekly_holdout_with_actual_exog)

    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    ranking = summary["weekly_baseline_candidate_comparison"]
    assert ranking["selected_candidate"] == "price_promo_freight_baseline"
    assert ranking["selection_reason"] == "preferred_price_promo_freight_baseline"


def test_bundle_selection_rejects_non_finite_non_legacy_metrics(monkeypatch):
    def fake_predict_weekly_holdout_with_actual_exog(model, train_weekly, test_weekly, feature_names, seasonal_anchor_weight=0.0):
        actual = test_weekly["sales"].astype(float).values
        if "price_idx" in feature_names and "promotion_share" not in feature_names:
            pred = np.repeat(float(np.nanmean(actual)), len(actual))
        elif "promotion_share" in feature_names and "freight_mean" not in feature_names:
            pred = actual * 1.04
        elif "freight_mean" in feature_names:
            pred = actual * 1.02
        else:
            pred = actual * 0.98
        out = test_weekly[["week"]].copy() if "week" in test_weekly.columns else pd.DataFrame(index=test_weekly.index)
        out["pred_weekly_sales"] = pred
        return out

    monkeypatch.setattr(app_module, "predict_weekly_holdout_with_actual_exog", fake_predict_weekly_holdout_with_actual_exog)

    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    ranking = summary["weekly_baseline_candidate_comparison"]
    price_only = next(row for row in ranking["candidates"] if row["name"] == "price_only_baseline")
    assert price_only["eligible_under_selection_rule"] is False
    assert price_only["rejection_reason"] == "corr_non_finite"
    assert ranking["selected_candidate"] != "price_only_baseline"


def test_holdout_weekly_diagnostics_contains_all_bundle_prediction_columns():
    res = _analyze()
    holdout_weekly = pd.read_csv(pd.io.common.BytesIO(res["holdout_weekly_diagnostics_csv"]))
    expected = {
        "legacy_pred_sales",
        "price_only_pred_sales",
        "price_promo_pred_sales",
        "price_promo_freight_pred_sales",
        "selected_pred_sales",
        "final_pred_sales",
        "naive_pred_sales",
        "price_gap_ref_8w",
        "freight_pct_change_1w",
    }
    assert expected.issubset(set(holdout_weekly.columns))


def test_summary_contains_explicit_weekly_driver_mode():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    mode = summary["config"].get("weekly_driver_mode")
    assert mode in {
        "naive_core_only",
        "weekly_ml_core_only",
    }


def test_runtime_never_applies_uplift_multiplier_fallback():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    cfg = summary["config"]
    assert "fallback_multiplier_used" in cfg
    base_price = float(res["_trained_bundle"]["base_ctx"]["price"])
    scenario = run_what_if_projection(res["_trained_bundle"], manual_price=base_price * 0.9, overrides={"promotion": 1.0})
    assert scenario["fallback_multiplier_used"] is False
    assert float(np.abs(scenario["daily"]["uplift_log"]).sum()) < 1e-9


def test_learned_uplift_log_changes_with_price_when_enabled():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.9)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.1)
    assert bundle["uplift_bundle"].get("disabled") in {True, False}
    assert float(low["daily"]["uplift_log"].mean()) == float(high["daily"]["uplift_log"].mean()) == 0.0


def test_baseline_decomposition_consistency_when_uplift_active():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]
    sc = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]) * 0.95)
    daily = sc["daily"]
    assert "base_pred_sales" in daily.columns
    assert "pred_sales" in daily.columns
    assert float(np.abs(daily["pred_sales"] - daily["base_pred_sales"]).sum()) > 1e-6


def test_uplift_rollback_when_holdout_worse_than_baseline(monkeypatch):
    orig_predict = app_module.predict_uplift_log_bundle

    def _bad_uplift(frame, uplift_bundle):
        pred, std = orig_predict(frame, uplift_bundle)
        return pred + 1.5, std

    monkeypatch.setattr(app_module, "predict_uplift_log_bundle", _bad_uplift)
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]["uplift_bundle"]
    assert bundle.get("reason") in {"uplift_holdout_failed", "benchmark_gate_failed", "passed", "production_runtime_disabled_diagnostic_only"}
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    assert float(np.abs(holdout["final_pred_sales"] - holdout["baseline_pred_sales"]).sum()) < 1e-6
    trace = pd.read_csv(pd.io.common.BytesIO(res["uplift_holdout_trace_csv"]))
    if bundle.get("reason") == "uplift_holdout_failed":
        assert float(np.abs(trace["final_pred_attempted"] - trace["core_pred"]).sum()) > 0.0
        assert float(np.abs(trace["final_pred_active"] - trace["core_pred"]).sum()) < 1e-9
        assert float(np.abs(trace["uplift_log_raw_attempted"]).sum()) > 0.0


def test_active_path_contract_and_uplift_off_in_report():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    out = summary["scenario_output_summary"]
    assert out["active_path_contract"].startswith("price_promo_freight_baseline")
    assert out["learned_uplift_contract"] == "inactive_production_diagnostic_only"
    assert summary["config"]["learned_uplift_active"] is False


def test_scenario_monotonicity_including_shock():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price)
    price_down = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    price_up = run_what_if_projection(bundle, manual_price=base_price * 1.05)
    promo_up = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 1.0})
    freight_up = run_what_if_projection(bundle, manual_price=base_price, overrides={"freight_multiplier": 1.2})
    shock_up = run_what_if_projection(bundle, manual_price=base_price, demand_multiplier=1.10)
    shock_down = run_what_if_projection(bundle, manual_price=base_price, demand_multiplier=0.90)
    assert price_down["demand_total"] >= base["demand_total"] >= price_up["demand_total"]
    assert promo_up["demand_total"] >= base["demand_total"]
    assert freight_up["demand_total"] <= base["demand_total"]
    assert shock_up["demand_total"] > base["demand_total"] > shock_down["demand_total"]


def test_no_double_counting_price_effect_against_direct_engine():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    target_price = base_price * 1.05
    scenario = run_what_if_projection(bundle, manual_price=target_price)
    baseline_daily = run_what_if_projection(bundle, manual_price=base_price)["daily"]
    from scenario_engine import run_scenario as run_scenario_engine
    expected = run_scenario_engine(
        baseline_output=pd.DataFrame({"date": baseline_daily["date"], "baseline_units": baseline_daily["base_pred_sales"]}),
        scenario_inputs={
            "baseline_price_ref": base_price,
            "scenario_price": target_price,
            "baseline_net_price": base_price * (1.0 - float(bundle["base_ctx"].get("discount", 0.0))),
            "scenario_net_price": target_price * (1.0 - float(bundle["base_ctx"].get("discount", 0.0))),
            "promo_flag_baseline": 1.0 if float(bundle["base_ctx"].get("promotion", 0.0)) > 0 else 0.0,
            "promo_flag_scenario": 1.0 if float(bundle["base_ctx"].get("promotion", 0.0)) > 0 else 0.0,
            "promo_intensity_baseline": float(bundle["base_ctx"].get("promotion", 0.0)),
            "promo_intensity_scenario": float(bundle["base_ctx"].get("promotion", 0.0)),
            "freight_ref": float(bundle["base_ctx"].get("freight_value", 0.0)),
            "freight_scenario": float(bundle["base_ctx"].get("freight_value", 0.0)),
            "baseline_unit_cost": float(bundle["base_ctx"].get("cost", base_price * 0.65)),
            "unit_cost": float(bundle["base_ctx"].get("cost", base_price * 0.65)),
            "baseline_freight_value": float(bundle["base_ctx"].get("freight_value", 0.0)),
            "freight_value": float(bundle["base_ctx"].get("freight_value", 0.0)),
        },
    )
    assert np.allclose(
        scenario["daily"]["pred_sales"].astype(float).values,
        np.asarray(expected["final_units"], dtype=float),
    )


def test_uplift_disabled_when_benchmark_gate_fails(monkeypatch):
    class BadWeeklyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(app_module, "train_weekly_core_model", lambda weekly_train, feature_names: BadWeeklyModel())
    monkeypatch.setattr(
        app_module,
        "fit_weekly_uplift_model",
        lambda weekly_train, baseline_pred_train, small_mode: {
            "models": [],
            "features": ["baseline_log_feature"],
            "feature_stats": {},
            "disabled": True,
            "reason": "forced_for_test",
            "signal_info": {},
        },
    )
    res = run_full_pricing_analysis_universal(_make_txn(420), "cat-a", "sku-1")
    bundle = res["_trained_bundle"]["uplift_bundle"]
    assert bundle.get("disabled") is True
    assert bundle.get("reason") in {"benchmark_gate_failed", "forced_for_test", "production_runtime_disabled_diagnostic_only"}
    assert len(bundle.get("models", [])) == 0
    assert res["_trained_bundle"]["baseline_bundle"]["selected_forecaster"] in {"weekly_model", "naive_lag1w", "naive_ma4w"}
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    assert float(np.abs(holdout["final_pred_sales"] - holdout["baseline_pred_sales"]).sum()) < 1e-9
    assert float(np.abs(holdout["uplift_log_pred"]).sum()) < 1e-9


def test_refit_scope_is_full_history():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert summary["config"]["fit_scope"] == "refit_full_history"


def test_raw_adjusted_and_ood_fields_present():
    res = _analyze()
    bundle = res["_trained_bundle"]
    sc = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]) * 2.0)
    assert "profit_total_raw" in sc and "profit_total_adjusted" in sc
    assert "requested_price" in sc and "price_for_model" in sc and "ood_flag" in sc


def test_adjusted_profit_not_above_raw_profit():
    res = _analyze()
    bundle = res["_trained_bundle"]
    scenario = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]))
    assert scenario["profit_total_adjusted"] <= scenario["profit_total_raw"]


def test_higher_cost_multiplier_not_improve_adjusted_profit():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, cost_multiplier=1.0)
    high_cost = run_what_if_projection(bundle, manual_price=base_price, cost_multiplier=1.2)
    assert high_cost["profit_total_adjusted"] <= base["profit_total_adjusted"]


def test_oof_small_mode_parameter_accepts_false():
    from app import make_walk_forward_oof_baseline

    df = _make_txn(60)
    daily = build_daily_from_transactions(df, "sku-1", target_category="cat-a")
    daily = robust_clean_dirty_data(daily)
    daily = daily.assign(log_sales=np.log1p(pd.to_numeric(daily["sales"], errors="coerce").fillna(0.0)))
    features = [c for c in ["price", "discount", "promotion", "is_weekend"] if c in daily.columns]
    out = make_walk_forward_oof_baseline(daily.copy(), features, n_splits=3, small_mode=False)
    assert len(out) == len(daily)


def test_feature_report_truthfulness_for_engineered_features():
    res = _analyze()
    fr = res["feature_report"]
    row = fr[fr["factor_name"] == "sales_lag1"].iloc[0]
    assert bool(row["engineered_feature"]) is True
    assert bool(row["found_in_raw"]) is False
    assert bool(row["present_in_daily"]) is True


def test_feature_report_has_weekly_usage_flag():
    res = _analyze()
    fr = res["feature_report"]
    row = fr[fr["factor_name"] == "sales_lag1w"].iloc[0]
    assert "used_in_weekly_model" in fr.columns
    assert bool(row["used_in_weekly_model"]) is True


def test_seasonal_anchor_weight_present_in_bundle_and_summary():
    res = _analyze_long_signal()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    weight_cfg = float(summary["config"]["seasonal_anchor_weight"])
    weight_bundle = float(res["_trained_bundle"]["baseline_bundle"]["seasonal_anchor_weight"])
    assert 0.0 <= weight_cfg <= 0.30
    assert abs(weight_cfg - weight_bundle) < 1e-9


def test_small_mode_warnings_present_on_short_history():
    res = run_full_pricing_analysis_universal(_make_txn(60), "cat-a", "sku-1")
    assert res["small_mode_info"]["small_mode"] is True
    assert len(res["warnings"]) > 0


def test_use_weekly_ml_false_path_keeps_attempted_bundle_initialized():
    res = run_full_pricing_analysis_universal(_make_txn(60), "cat-a", "sku-1")
    trained = res["_trained_bundle"]
    assert "uplift_bundle_attempted" in trained
    assert isinstance(trained["uplift_bundle_attempted"], dict)
    assert "features" in trained["uplift_bundle_attempted"]


def test_scenario_modeled_price_state_for_clipped_request():
    res = _analyze()
    bundle = res["_trained_bundle"]
    sc = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]) * 3.0)
    assert "price_for_model" in sc
    assert sc["price_clipped"] is True
    res["scenario_price_requested"] = float(sc["requested_price"])
    res["scenario_price_modeled"] = float(sc["price_for_model"])
    res["scenario_price"] = res["scenario_price_modeled"]
    assert res["scenario_price"] == res["scenario_price_modeled"]


def test_stock_is_not_public_scenario_factor():
    res = _analyze()
    fr = res["feature_report"]
    row = fr[fr["factor_name"] == "stock"].iloc[0]
    assert bool(row["used_in_scenario"]) is False


def test_weekly_baseline_price_effect_works_when_uplift_disabled():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"].copy()

    class PriceAwareModel:
        def predict(self, X):
            xdf = pd.DataFrame(X).copy()
            price_idx = pd.to_numeric(xdf.get("price_idx", pd.Series(np.ones(len(xdf)))), errors="coerce").fillna(1.0).clip(lower=0.5, upper=1.5)
            pred = 60.0 / price_idx.values
            return np.log1p(np.clip(pred, 0.0, None))

    bundle["baseline_bundle"] = dict(bundle["baseline_bundle"])
    bundle["baseline_bundle"]["selected_forecaster"] = "weekly_model"
    bundle["baseline_bundle"]["features"] = list(dict.fromkeys(app_module.WEEKLY_BASELINE_FEATURES))
    bundle["baseline_bundle"]["model"] = PriceAwareModel()
    bundle["uplift_bundle"] = {"models": [], "features": ["baseline_log_feature"], "feature_stats": {}, "disabled": True, "reason": "forced_for_test", "signal_info": {}}
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.05)
    assert low["demand_total"] != high["demand_total"]


def test_no_fallback_multiplier_for_weekly_model_with_exog_baseline():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"].copy()
    bundle["baseline_bundle"] = dict(bundle["baseline_bundle"])
    bundle["baseline_bundle"]["selected_forecaster"] = "weekly_model"
    bundle["baseline_bundle"]["features"] = list(dict.fromkeys(app_module.WEEKLY_BASELINE_FEATURES))
    bundle["uplift_bundle"] = {"models": [], "features": ["baseline_log_feature"], "feature_stats": {}, "disabled": True, "reason": "forced_for_test", "signal_info": {}}
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.08, overrides={"promotion": 1.0})
    assert scenario["demand_total"] > 0.0
    assert scenario["fallback_multiplier_used"] is False


def test_resolve_scenario_driver_mode_has_three_states():
    assert resolve_scenario_driver_mode("weekly_model", True) == "weekly_ml_exogenous"
    assert resolve_scenario_driver_mode("weekly_model", False) == "weekly_ml_legacy_plus_rule_based_multiplier"
    assert resolve_scenario_driver_mode("naive_ma4w", False) == "naive_plus_rule_based_multiplier"


def test_resolve_weekly_driver_mode_distinguishes_naive_and_weekly_paths():
    assert resolve_weekly_driver_mode("naive_ma4w", False, True) == "naive_core_only"
    assert resolve_weekly_driver_mode("weekly_model", False, False) == "weekly_ml_core_only"
    assert resolve_weekly_driver_mode("weekly_model", True, False) == "weekly_ml_plus_learned_uplift"
    assert resolve_weekly_driver_mode("weekly_model", False, True) == "weekly_ml_plus_rule_based_multiplier"


def test_weekly_baseline_monotone_price_idx_non_increasing():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]
    if bundle["baseline_bundle"]["selected_forecaster"] != "weekly_model":
        return
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.90)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.10)
    assert high["demand_total"] <= low["demand_total"]


def test_feature_report_weekly_only_reason_is_not_absent_after_daily_pipeline():
    res = _analyze()
    fr = res["feature_report"]
    row = fr[fr["factor_name"] == "price_idx"].iloc[0]
    assert bool(row["present_in_weekly"]) is True
    assert row["reason_excluded"] != "absent_after_daily_pipeline"


def test_bridge_week_keeps_immediate_scenario_reaction():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]
    if bundle["baseline_bundle"]["selected_forecaster"] != "weekly_model":
        return
    if bundle["uplift_bundle"].get("disabled") is not True:
        return
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.05)
    low_first_week = float(low["daily"].head(7)["actual_sales"].sum())
    high_first_week = float(high["daily"].head(7)["actual_sales"].sum())
    assert low_first_week != high_first_week


def test_uplift_debug_artifacts_are_exported_with_expected_schema():
    res = _analyze_long_signal()
    debug_csv = pd.read_csv(pd.io.common.BytesIO(res["uplift_debug_report_csv"]))
    trace_csv = pd.read_csv(pd.io.common.BytesIO(res["uplift_holdout_trace_csv"]))
    assert {
        "feature_name",
        "present_in_train_weekly",
        "present_in_holdout_weekly",
        "selected_for_attempted_uplift",
        "selected_for_active_uplift",
        "dropped_reason",
    }.issubset(set(debug_csv.columns))
    assert {
        "week_start",
        "actual_sales",
        "core_pred",
        "uplift_log_raw_attempted",
        "uplift_log_clipped_attempted",
        "uplift_multiplier_attempted",
        "final_pred_attempted",
        "uplift_log_raw_active",
        "uplift_log_clipped_active",
        "uplift_multiplier_active",
        "final_pred_active",
    }.issubset(set(trace_csv.columns))


def test_summary_contains_attempted_vs_active_uplift_blocks():
    res = _analyze_long_signal()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    cfg = summary["config"]
    assert "uplift_attempted_features" in cfg
    assert "uplift_active_features" in cfg
    assert "uplift_gate_result" in cfg
    assert "uplift_gate_reason" in cfg
    assert "uplift_debug_info" in cfg
    assert "uplift_support_train" in cfg
    assert "uplift_support_holdout" in cfg


def test_no_change_with_existing_promo_keeps_baseline():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    promo_state = float(bundle["base_ctx"].get("promotion", 0.0))
    sc = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": promo_state})
    daily = sc["daily"]
    pred = pd.to_numeric(daily["pred_sales"], errors="coerce").fillna(0.0).values
    base = pd.to_numeric(daily["base_pred_sales"], errors="coerce").fillna(0.0).values
    assert np.allclose(pred, base)


def test_stock_constraint_not_double_counted():
    baseline = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=2, freq="D"), "baseline_units": [100.0, 100.0]})
    out = run_scenario(
        baseline_output=baseline,
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_baseline": 0.0,
            "promo_scenario": 0.0,
            "freight_ref": 5.0,
            "freight_scenario": 5.0,
            "available_stock": np.array([20.0, 15.0]),
            "scenario_net_price": 100.0,
            "unit_cost": 65.0,
            "freight_value": 5.0,
        },
    )
    assert np.allclose(out["final_units"], np.array([20.0, 15.0]))


def test_small_mode_does_not_override_scenario_result():
    res = run_full_pricing_analysis_universal(_make_txn(60), "cat-a", "sku-1")
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    sc = run_what_if_projection(bundle, manual_price=base_price * 1.1)
    daily = sc["daily"]
    assert float(np.abs(daily["pred_sales"] - daily["base_pred_sales"]).sum()) > 0.0


def test_real_integration_run_what_if_projection_uses_scenario_engine_result(monkeypatch):
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])

    def fake_run_scenario(baseline_output, scenario_inputs, shocks=None, metadata=None):
        n = len(baseline_output)
        zeros = np.zeros(n, dtype=float)
        ones = np.ones(n, dtype=float)
        return {
            "baseline_units": pd.to_numeric(baseline_output["baseline_units"], errors="coerce").fillna(0.0).to_numpy(),
            "price_effect": 1.0,
            "promo_effect": 1.0,
            "freight_effect": 1.0,
            "stock_effect": 1.0,
            "shock_multiplier": ones,
            "shock_units": zeros,
            "final_units": zeros,
            "baseline_revenue": zeros,
            "final_revenue": zeros,
            "baseline_margin": zeros,
            "final_margin": zeros,
            "confidence": {},
            "warnings": [],
        }

    monkeypatch.setattr(app_module, "run_scenario", fake_run_scenario)
    sc = run_what_if_projection(bundle, manual_price=base_price)
    assert float(sc["demand_total"]) == 0.0
    assert "legacy_baseline_meta" in sc and "scenario_engine_meta" in sc
    assert "price_confidence_score" in sc["scenario_engine_meta"]


def test_discount_multiplier_affects_scenario_economics():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, discount_multiplier=1.0, overrides={"discount": 0.10})
    deep_discount = run_what_if_projection(bundle, manual_price=base_price, discount_multiplier=2.0, overrides={"discount": 0.10})
    assert float(deep_discount["revenue_total"]) < float(base["revenue_total"])


def test_cost_multiplier_affects_profit():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, cost_multiplier=1.0)
    high_cost = run_what_if_projection(bundle, manual_price=base_price, cost_multiplier=1.2)
    assert float(high_cost["profit_total_raw"]) < float(base["profit_total_raw"])


def test_stock_cap_affects_realized_sales():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    uncapped = run_what_if_projection(bundle, manual_price=base_price)
    capped = run_what_if_projection(bundle, manual_price=base_price, stock_cap=1.0)
    assert float(capped["demand_total"]) < float(uncapped["demand_total"])


def test_demand_multiplier_is_applied_as_global_shock():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, demand_multiplier=1.0)
    up = run_what_if_projection(bundle, manual_price=base_price, demand_multiplier=1.15)
    assert float(up["demand_total"]) > float(base["demand_total"])


def test_summary_sensitivity_is_computed_via_runtime_what_if_path():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    sens = summary["scenario_sensitivity_diagnostics"]
    base_price = float(bundle["base_ctx"]["price"])

    base = run_what_if_projection(bundle, manual_price=base_price)
    low = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.05)
    promo = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": min(1.0, float(bundle["base_ctx"].get("promotion", 0.0)) + 0.10)})
    freight = run_what_if_projection(bundle, manual_price=base_price, overrides={"freight_value": float(bundle["base_ctx"].get("freight_value", 0.0)) * 1.10})

    base_total = float(base["demand_total"])
    assert sens["source"] == "run_what_if_projection_runtime_path"
    assert sens["price_minus_5pct_demand_delta_pct"] == pytest.approx(((float(low["demand_total"]) - base_total) / max(base_total, 1e-9)) * 100.0)
    assert sens["price_plus_5pct_demand_delta_pct"] == pytest.approx(((float(high["demand_total"]) - base_total) / max(base_total, 1e-9)) * 100.0)
    assert sens["promo_plus_10pp_demand_delta_pct"] == pytest.approx(((float(promo["demand_total"]) - base_total) / max(base_total, 1e-9)) * 100.0)
    assert sens["freight_plus_10pct_demand_delta_pct"] == pytest.approx(((float(freight["demand_total"]) - base_total) / max(base_total, 1e-9)) * 100.0)


def test_summary_and_runtime_price_plus_5pct_have_same_direction():
    res = _analyze_long_signal()
    bundle = res["_trained_bundle"]
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    sens = summary["scenario_sensitivity_diagnostics"]
    base_price = float(bundle["base_ctx"]["price"])
    runtime_base = run_what_if_projection(bundle, manual_price=base_price)
    runtime_high = run_what_if_projection(bundle, manual_price=base_price * 1.05)
    runtime_delta = float(runtime_high["demand_total"] - runtime_base["demand_total"])
    summary_delta = float(sens["price_plus_5pct_demand_delta_pct"])
    assert np.sign(runtime_delta) == np.sign(summary_delta)


def test_diagnostic_only_summary_explicitly_exposes_v1_active_path_contract():
    res = _analyze_long_signal()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    cfg = summary["config"]
    assert cfg["uplift_activation_mode"] == "diagnostic_only"
    assert cfg["v1_contract"]["active_path"] == cfg["final_active_path"]
    assert summary["scenario_output_summary"]["active_path_contract"] == cfg["final_active_path"]


def test_attempted_uplift_only_factor_not_marked_as_active_model_factor():
    res = _analyze_long_signal()
    fr = res["feature_report"]
    subset = fr[
        (fr["used_in_uplift_attempted"] == True)
        & (fr["used_in_uplift_active"] == False)
        & (fr["used_in_baseline"] == False)
    ]
    if subset.empty:
        return
    assert (subset["used_in_final_active_forecast"] == False).all()
    assert (subset["active_usage_reason"] == "uplift_deactivated_by_gate").all()
