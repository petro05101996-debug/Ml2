import copy
import io
import json

import numpy as np
import pandas as pd

import app as app_module
from app import (
    build_segment_paths,
    build_scenario_support_info,
    build_scenario_support_info_from_paths,
    build_applied_scenario_snapshot,
    build_business_report_payload,
    build_manual_scenario_artifacts,
    build_saved_scenario_metrics,
    build_trust_block,
    get_user_scenario_status,
    resolve_final_active_path,
    run_full_pricing_analysis_universal,
    run_what_if_projection,
    scenario_mode_label,
)
from test_smoke_mvp import _make_txn


def _analyze_bundle():
    return run_full_pricing_analysis_universal(_make_txn(240), "cat-a", "sku-1")


def test_baseline_forecast_uses_legacy_baseline_but_what_if_default_is_enhanced():
    res = _analyze_bundle()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert summary["config"]["selected_candidate"] == "legacy_baseline"
    assert summary["config"]["production_selected_candidate"] == "legacy_baseline"
    assert "diagnostic_selected_candidate" in summary["config"]
    assert summary["config"]["selection_mode"] == "diagnostic_comparison_runtime_frozen_to_legacy"
    assert summary["config"]["production_selection_reason"] == "v1_contract_runtime_frozen_to_legacy"
    assert summary["config"]["baseline_forecast_path"] == "weekly_ml_baseline"
    assert summary["config"]["scenario_calculation_path"] == "enhanced_local_factor_layer"
    assert summary["config"]["learned_uplift_path"] == "inactive_production_diagnostic_only"
    assert summary["config"]["final_user_visible_path"] == "weekly_ml_baseline + enhanced_local_factor_layer"


def test_uplift_not_used_in_production():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])

    base = run_what_if_projection(bundle, manual_price=base_price * 1.03)

    mutated = copy.deepcopy(bundle)
    mutated["uplift_bundle"] = {
        "models": [object()],
        "features": ["promotion", "freight_value"],
        "feature_stats": {},
        "disabled": False,
        "reason": "force_test",
        "signal_info": {},
        "neutral_reference_log": 3.0,
    }
    altered = run_what_if_projection(mutated, manual_price=base_price * 1.03)

    assert base["uplift_used_in_production"] is False
    assert altered["uplift_used_in_production"] is False
    assert np.isclose(float(base["demand_total"]), float(altered["demand_total"]))


def test_scenario_without_changes_equals_as_is():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])

    scenario = run_what_if_projection(bundle, manual_price=base_price)
    as_is_units = float(res["as_is_forecast"]["actual_sales"].sum())

    assert scenario["scenario_status"] == "as_is"
    assert np.isclose(float(scenario["demand_total"]), as_is_units)
    assert np.isclose(float(scenario["demand_total"] - as_is_units), 0.0)


def test_scenario_with_changes_returns_finite_values():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])

    scenario = run_what_if_projection(
        bundle,
        manual_price=base_price * 1.08,
        freight_multiplier=1.1,
        overrides={"promotion": 1.0},
    )

    assert scenario["scenario_status"] == "computed"
    assert np.isfinite(float(scenario["demand_total"]))
    assert np.isfinite(float(scenario["revenue_total"]))
    assert np.isfinite(float(scenario["profit_total"]))


def test_business_report_has_no_nan_in_core_blocks():
    res = _analyze_bundle()
    payload = build_business_report_payload(res, None, {})

    keys = [
        payload["baseline_forecast"]["units"],
        payload["as_is_forecast"]["units"],
        payload["scenario_forecast"]["units"],
        payload["delta_vs_as_is"]["units"],
        payload["baseline_forecast"]["revenue"],
        payload["as_is_forecast"]["revenue"],
        payload["scenario_forecast"]["revenue"],
        payload["delta_vs_as_is"]["revenue"],
    ]
    assert all(np.isfinite(float(v)) for v in keys)
    assert payload["scenario_status"] == "as_is"
    assert np.isclose(float(payload["scenario_forecast"]["units"]), float(payload["as_is_forecast"]["units"]))
    assert np.isclose(float(payload["delta_vs_as_is"]["units"]), 0.0)


def test_business_report_scenario_status_computed_when_manual_scenario_present():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 1.0})
    payload = build_business_report_payload(res, scenario, {})
    assert payload["scenario_status"] == "computed"


def test_active_path_helper_contract():
    assert resolve_final_active_path("weekly_model", "legacy_baseline") == "legacy_baseline+scenario_recompute"
    assert resolve_final_active_path("naive_lag1w", "legacy_baseline") == "naive_lag1w+scenario_recompute"
    assert resolve_final_active_path("naive_ma4w", "legacy_baseline") == "naive_ma4w+scenario_recompute"
    assert resolve_final_active_path("unknown", "legacy_baseline") == "deterministic_fallback+scenario_recompute"


def test_profit_is_none_when_cost_missing():
    res = _analyze_bundle()
    res_no_cost = dict(res)
    res_no_cost["cost_input_available"] = False
    payload = build_business_report_payload(res_no_cost, None, {})
    assert payload["as_is_forecast"]["profit"] is None
    assert payload["scenario_forecast"]["profit"] is None


def test_active_path_summary_contains_v1_contract_fields():
    res = _analyze_bundle()
    payload = build_business_report_payload(res, None, {})
    summary = payload["active_path_summary"]
    assert "production_selected_candidate" in summary
    assert "diagnostic_selected_candidate" in summary
    assert "selection_mode" in summary
    assert summary["uplift_mode"] == "diagnostic_only"
    assert summary["uplift_used_in_production"] is False
    assert summary["scenario_status"] == "as_is"
    assert {"requested_price", "model_price", "price_clipped", "clip_reason"}.issubset(summary["price_clip"].keys())


def test_price_clip_contract_visible():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    high_price = float(bundle["daily_base"]["price"].max()) * 1.5

    scenario = run_what_if_projection(bundle, manual_price=high_price)

    assert "requested_price" in scenario
    assert "model_price" in scenario
    assert scenario["price_clipped"] is True
    assert str(scenario["clip_reason"]) != ""


def test_promo_down_reduces_demand():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base_promo = float(bundle["base_ctx"].get("promotion", 0.0))

    higher = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": min(1.0, base_promo + 0.4)})
    lower = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": max(0.0, base_promo - 0.4)})

    assert float(lower["demand_total"]) <= float(higher["demand_total"])


def test_discount_changes_demand_not_only_economics():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low_discount = run_what_if_projection(bundle, manual_price=base_price, overrides={"discount": 0.05})
    high_discount = run_what_if_projection(bundle, manual_price=base_price, overrides={"discount": 0.45})
    assert not np.isclose(float(low_discount["demand_total"]), float(high_discount["demand_total"]))


def test_net_price_support_warning_and_confidence_drop():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, overrides={"discount": 0.05})
    deep = run_what_if_projection(bundle, manual_price=base_price, overrides={"discount": 0.95})
    trust = build_trust_block(res, deep)
    assert any("Итоговая цена для клиента после скидки" in str(w) for w in trust["warnings"])
    assert float(deep["confidence_scenario"]) <= float(base["confidence_scenario"])


def test_flat_history_warnings_price_promo_freight():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "price": [100.0] * 10,
            "discount": [0.10] * 10,
            "promotion": [0.2] * 10,
            "freight_value": [5.0] * 10,
            "sales": [10.0] * 10,
        }
    )
    results = {"history_daily": history, "analysis_run_summary_json": b"{}"}
    wr = {
        "effective_scenario": {"applied_price_gross": 120.0, "applied_price_net": 96.0, "promotion": 0.5, "freight_value": 8.0},
        "net_price_support": {},
    }
    trust = build_trust_block(results, wr)
    text = " | ".join([str(x) for x in trust["warnings"]])
    assert "Цена вышла за пределы исторически наблюдаемого уровня" in text
    assert "Промо вышло за пределы исторически наблюдаемого уровня" in text
    assert "Логистика вышла за пределы исторически наблюдаемого уровня" in text


def test_report_confidence_consistency():
    res = _analyze_bundle()
    payload = build_business_report_payload(res, {"confidence_scenario": 0.12, "confidence_label": "Низкая"}, {})
    assert payload["scenario_confidence"]["label"] == "Низкая"


def test_applied_vs_requested_after_clip_status_as_is():
    base_form = {"manual_price": 100.0, "discount": 0.1, "promo_value": 0.2, "freight_mult": 1.0, "demand_mult": 1.0, "hdays": 30}
    current_form = dict(base_form)
    snapshot = {"manual_price_requested": 500.0, "manual_price_applied": 100.0, "scenario_status": "as_is"}
    status = get_user_scenario_status(current_form, base_form, snapshot, "applied")
    assert status == "as_is"


def test_export_consistency_uses_applied_values():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.5, overrides={"discount": 0.33, "promotion": 0.4})
    _, daily_blob = build_manual_scenario_artifacts(res, wr)
    daily = pd.read_csv(io.BytesIO(daily_blob))
    eff = wr["effective_scenario"]
    assert np.isclose(float(daily["scenario_price_gross"].iloc[0]), float(eff["applied_price_gross"]))
    assert np.isclose(float(daily["scenario_price_net"].iloc[0]), float(eff["applied_price_net"]))
    assert np.isclose(float(daily["scenario_discount"].iloc[0]), float(eff["applied_discount"]))
    assert np.isclose(float(daily["scenario_promotion"].iloc[0]), float(eff["promotion"]))


def test_saved_scenario_snapshot_contains_effective_fields():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    wr = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]) * 1.4)
    saved = build_saved_scenario_metrics(res["as_is_forecast"], wr["daily"], wr)
    for key in ["requested_price_gross", "applied_price_gross", "applied_price_net", "applied_discount", "clip_reason"]:
        assert key in saved


def test_mode_dispatch_and_unknown_mode_error():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    legacy = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="legacy_current")
    enhanced = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="enhanced_local_factors")
    assert legacy["scenario_calc_mode"] == "legacy_current"
    assert enhanced["scenario_calc_mode"] == "enhanced_local_factors"
    with np.testing.assert_raises(ValueError):
        run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="unknown")


def test_mode_participates_in_dirty_state():
    base_form = {
        "manual_price": 100.0,
        "discount": 0.1,
        "promo_value": 0.2,
        "freight_mult": 1.0,
        "demand_mult": 1.0,
        "hdays": 30,
        "scenario_calc_mode": "legacy_current",
    }
    current_form = dict(base_form)
    current_form["scenario_calc_mode"] = "enhanced_local_factors"
    snapshot = {"scenario_status": "computed", "scenario_calc_mode": "legacy_current"}
    status = get_user_scenario_status(current_form, base_form, snapshot, "applied")
    assert status == "dirty"


def test_saved_scenario_keeps_mode():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wa = run_what_if_projection(bundle, manual_price=base_price * 1.1, scenario_calc_mode="enhanced_local_factors")
    saved = build_saved_scenario_metrics(res["as_is_forecast"], wa["daily"], wa)
    assert saved["scenario_calc_mode"] == "enhanced_local_factors"


def test_trained_bundle_contains_small_mode_info_for_enhanced_path():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    assert "small_mode_info" in bundle
    assert isinstance(bundle["small_mode_info"], dict)
    assert "small_mode" in bundle["small_mode_info"]


def test_enhanced_result_explicit_path_metadata_not_legacy():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    enhanced = run_what_if_projection(bundle, manual_price=base_price * 1.05, scenario_calc_mode="enhanced_local_factors")
    assert enhanced["active_path_contract"] == "weekly_ml_baseline + enhanced_local_factor_layer"
    assert enhanced["scenario_calculation_path"] == "enhanced_local_factor_layer"
    assert enhanced["scenario_driver_mode"] == "baseline_daily_plus_local_factor_layer"
    assert enhanced["legacy_or_enhanced_label"] == "enhanced"


def test_enhanced_path_uses_legacy_baseline_path_contract(monkeypatch):
    res = _analyze_bundle()
    bundle = copy.deepcopy(res["_trained_bundle"])
    base_price = float(bundle["base_ctx"]["price"])

    def _fake_legacy(*args, **kwargs):
        future_dates = bundle["future_dates"].copy()
        n = len(future_dates)
        return {
            "effective_scenario": {"applied_discount": 0.0, "cost": 10.0, "freight_value": 1.0, "promotion": 0.0},
            "model_price": base_price,
            "confidence": 0.6,
            "baseline_forecast_path": "naive_lag1w_baseline",
            "scenario_calculation_path": "scenario_recompute",
            "learned_uplift_path": "inactive_production_diagnostic_only",
            "final_user_visible_path": "naive_lag1w_baseline + scenario_recompute",
            "active_path_contract": "naive_lag1w_baseline + scenario_recompute",
            "daily": pd.DataFrame({"date": future_dates["date"], "base_pred_sales": np.full(n, 10.0)}),
        }

    def _fake_simulate(*args, **kwargs):
        future_dates = (kwargs.get("future_dates") if "future_dates" in kwargs else args[2]).copy()
        return {"daily": pd.DataFrame({"date": future_dates["date"], "base_pred_sales": np.full(len(future_dates), 10.0)})}

    def _fake_enhanced(**kwargs):
        future_dates = pd.to_datetime(kwargs["future_dates"]["date"])
        n = len(future_dates)
        profile = pd.DataFrame(
            {
                "date": future_dates,
                "scenario_demand_raw": np.full(n, 11.0),
                "actual_sales": np.full(n, 11.0),
                "lost_sales": np.zeros(n),
                "revenue": np.full(n, 110.0),
                "profit": np.full(n, 44.0),
                "scenario_price_gross": np.full(n, base_price),
                "scenario_discount": np.zeros(n),
                "scenario_price_net": np.full(n, base_price),
                "scenario_promotion": np.zeros(n),
                "scenario_freight_value": np.full(n, 1.0),
                "scenario_cost": np.full(n, 10.0),
                "available_stock": np.full(n, 999.0),
                "price_effect": np.ones(n),
                "promo_effect": np.ones(n),
                "freight_effect": np.ones(n),
                "standard_multiplier": np.ones(n),
                "shock_multiplier": np.ones(n),
                "shock_units": np.zeros(n),
            }
        )
        return {
            "scenario_profile": profile,
            "confidence": {"price": {"score": 0.6}, "promo": {"score": 0.6}, "freight": {"score": 0.6}},
            "price_effect_vector": np.ones(n),
            "promo_effect_vector": np.ones(n),
            "freight_effect_vector": np.ones(n),
            "shock_multiplier": np.ones(n),
            "effect_breakdown": {},
            "warnings": [],
        }

    monkeypatch.setattr(app_module, "_run_what_if_projection_legacy", _fake_legacy)
    monkeypatch.setattr(app_module, "simulate_horizon_profit", _fake_simulate)
    monkeypatch.setattr(app_module, "run_enhanced_scenario", _fake_enhanced)

    enhanced = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="enhanced_local_factors")
    assert enhanced["baseline_forecast_path"] == "naive_lag1w_baseline"
    assert enhanced["scenario_calculation_path"] == "enhanced_local_factor_layer"
    assert enhanced["final_user_visible_path"] == "naive_lag1w_baseline + enhanced_local_factor_layer"


def test_enhanced_path_falls_back_to_small_mode_when_info_missing():
    res = _analyze_bundle()
    bundle = copy.deepcopy(res["_trained_bundle"])
    bundle.pop("small_mode_info", None)
    base_price = float(bundle["base_ctx"]["price"])
    enhanced = run_what_if_projection(bundle, manual_price=base_price * 1.02, scenario_calc_mode="enhanced_local_factors")
    assert enhanced["scenario_calc_mode"] == "enhanced_local_factors"
    assert "price" in enhanced.get("confidence_factors", {})


def test_enhanced_differs_from_legacy_with_time_varying_paths():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    horizon = 14
    dates = pd.date_range(pd.to_datetime(bundle["future_dates"]["date"]).min(), periods=horizon, freq="D")
    price_path = [{ "date": str(d.date()), "value": base_price * (0.95 if i < 7 else 1.10)} for i, d in enumerate(dates)]
    promo_path = [{ "date": str(d.date()), "value": 0.8 if i < 7 else 0.0} for i, d in enumerate(dates)]

    legacy = run_what_if_projection(
        bundle,
        manual_price=base_price,
        horizon_days=horizon,
        overrides={"promotion": 0.2},
        scenario_calc_mode="legacy_current",
    )
    enhanced = run_what_if_projection(
        bundle,
        manual_price=base_price,
        horizon_days=horizon,
        overrides={"promotion": 0.2, "price_path": price_path, "promo_path": promo_path},
        scenario_calc_mode="enhanced_local_factors",
    )
    assert not np.isclose(float(legacy["demand_total"]), float(enhanced["demand_total"]))
    assert bool((enhanced.get("effect_breakdown") or {}).get("trajectory_inputs_active", False)) is True


def test_support_block_contains_required_fields():
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30, freq="D"),
            "sales": np.random.RandomState(0).randint(0, 20, size=30),
            "price": np.linspace(90, 110, 30),
            "discount": np.linspace(0.0, 0.2, 30),
            "promotion": ([0.0] * 15) + ([0.3] * 15),
            "freight_value": np.linspace(4, 6, 30),
        }
    )
    support = build_scenario_support_info(
        hist,
        {"applied_price_net": 95.0, "promotion": 0.3, "freight_value": 5.0},
        {},
    )
    required = {
        "history_days",
        "recent_history_days",
        "recent_nonzero_sales_days",
        "unique_price_points",
        "price_range_pct",
        "price_changes",
        "price_stability",
        "promo_active_days",
        "promo_change_days",
        "promo_weeks",
        "promo_variability",
        "freight_change_days",
        "freight_changes",
        "freight_variation",
        "discount_unique_count",
        "promotion_positive_share",
        "local_price_support_days",
        "local_promo_support_days",
        "local_freight_support_days",
    }
    assert required.issubset(set(support.keys()))


def test_build_segment_paths_warns_on_overlap_and_tracks_usage():
    future = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10, freq="D")})
    payload, warnings = build_segment_paths(
        future,
        {"price": 100.0, "promotion": 0.0, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "freight_value": 5.0},
        [
            {"start_date": "2025-01-01", "end_date": "2025-01-05", "price": 100.0, "promotion": 0.1, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "shock_units": 0.0},
            {"start_date": "2025-01-04", "end_date": "2025-01-08", "price": 110.0, "promotion": 0.1, "freight_multiplier": 1.1, "demand_multiplier": 1.0, "shock_units": 0.0},
        ],
    )
    assert any("пересекается" in w for w in warnings)
    assert len(payload.get("price_path", [])) == 10


def test_enhanced_single_segment_matches_scalar_behavior():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    horizon = 10
    scalar = run_what_if_projection(
        bundle,
        manual_price=base_price * 1.02,
        horizon_days=horizon,
        overrides={"promotion": 0.2},
        scenario_calc_mode="enhanced_local_factors",
    )
    future = bundle["future_dates"].head(horizon)
    payload, _ = build_segment_paths(
        future,
        {"price": base_price * 1.02, "promotion": 0.2, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "freight_value": float(bundle["base_ctx"].get("freight_value", 0.0))},
        [{"start_date": str(pd.to_datetime(future["date"]).min().date()), "end_date": str(pd.to_datetime(future["date"]).max().date()), "price": base_price * 1.02, "promotion": 0.2, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "shock_units": 0.0}],
    )
    segmented = run_what_if_projection(
        bundle,
        manual_price=base_price * 1.02,
        horizon_days=horizon,
        overrides={"promotion": 0.2, "price_path": payload["price_path"], "promo_path": payload["promo_path"]},
        scenario_calc_mode="enhanced_local_factors",
    )
    assert np.isclose(float(scalar["demand_total"]), float(segmented["demand_total"]), rtol=0.03)


def test_mode_label_helper_human_readable():
    assert "Legacy" in scenario_mode_label("legacy_current")


def test_no_double_count_when_demand_path_and_global_multiplier_present():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    horizon = 12
    dates = pd.date_range(pd.to_datetime(bundle["future_dates"]["date"]).min(), periods=horizon, freq="D")
    dpath = [{"date": str(d.date()), "value": 1.1} for d in dates]
    with_global = run_what_if_projection(
        bundle,
        manual_price=base_price,
        demand_multiplier=1.1,
        horizon_days=horizon,
        overrides={"demand_multiplier_path": dpath},
        scenario_calc_mode="enhanced_local_factors",
    )
    path_only = run_what_if_projection(
        bundle,
        manual_price=base_price,
        demand_multiplier=1.0,
        horizon_days=horizon,
        overrides={"demand_multiplier_path": dpath},
        scenario_calc_mode="enhanced_local_factors",
    )
    assert np.isclose(float(with_global["demand_total"]), float(path_only["demand_total"]))


def test_support_label_changes_with_path_extremity():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    horizon = 14
    dates = pd.date_range(pd.to_datetime(bundle["future_dates"]["date"]).min(), periods=horizon, freq="D")
    mild = run_what_if_projection(
        bundle,
        manual_price=base_price,
        horizon_days=horizon,
        overrides={
            "price_path": [{"date": str(d.date()), "value": base_price * 0.98} for d in dates],
            "promo_path": [{"date": str(d.date()), "value": 0.1} for d in dates],
        },
        scenario_calc_mode="enhanced_local_factors",
    )
    extreme = run_what_if_projection(
        bundle,
        manual_price=base_price,
        horizon_days=horizon,
        overrides={
            "price_path": [{"date": str(d.date()), "value": base_price * (1.45 if i % 2 == 0 else 0.55)} for i, d in enumerate(dates)],
            "promo_path": [{"date": str(d.date()), "value": (1.0 if i % 2 == 0 else 0.0)} for i, d in enumerate(dates)],
            "freight_path": [{"date": str(d.date()), "value": float(bundle["base_ctx"].get("freight_value", 0.0)) * (2.0 if i % 2 == 0 else 0.3)} for i, d in enumerate(dates)],
        },
        scenario_calc_mode="enhanced_local_factors",
    )
    mild_score = float((mild.get("scenario_support_info", {}) or {}).get("support_score", 0.0))
    extreme_score = float((extreme.get("scenario_support_info", {}) or {}).get("support_score", 0.0))
    assert extreme_score <= mild_score


def test_path_guardrails_clip_extreme_values_and_emit_warnings():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    horizon = 7
    dates = pd.date_range(pd.to_datetime(bundle["future_dates"]["date"]).min(), periods=horizon, freq="D")
    scenario = run_what_if_projection(
        bundle,
        manual_price=base_price,
        horizon_days=horizon,
        overrides={
            "price_path": [{"date": str(d.date()), "value": base_price * 3.0} for d in dates],
            "promo_path": [{"date": str(d.date()), "value": 0.99} for d in dates],
            "demand_multiplier_path": [{"date": str(d.date()), "value": 2.5} for d in dates],
        },
        scenario_calc_mode="enhanced_local_factors",
    )
    txt = " | ".join([str(x) for x in scenario.get("warnings", [])])
    assert "guardrails" in txt
    assert float(pd.to_numeric(scenario["daily"]["promotion"], errors="coerce").max()) <= 0.70 + 1e-9


def test_path_support_warnings_are_path_based():
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "sales": np.linspace(1, 2, 20),
            "price": np.linspace(100, 101, 20),
            "discount": np.zeros(20),
            "promotion": np.zeros(20),
            "freight_value": np.ones(20) * 5.0,
        }
    )
    scenario_daily = pd.DataFrame(
        {
            "scenario_price_net": np.linspace(150, 180, 10),
            "scenario_promotion": np.ones(10),
            "scenario_freight_value": np.ones(10) * 12.0,
            "shock_multiplier": np.ones(10),
        }
    )
    info = build_scenario_support_info_from_paths(hist, scenario_daily, {})
    assert len(info.get("warnings", [])) > 0


def test_price_guardrail_mode_change_marks_scenario_dirty():
    import app

    base_form = app.collect_current_form_values(100.0, 0.0, 0.0, 1.0, 1.0, 30, "catboost_full_factors", "safe_clip")
    applied_snapshot = {
        "manual_price_requested": 100.0,
        "discount_requested": 0.0,
        "promo_requested": 0.0,
        "freight_mult": 1.0,
        "demand_mult": 1.0,
        "horizon_days": 30,
        "scenario_calc_mode": "catboost_full_factors",
        "price_guardrail_mode": "safe_clip",
        "scenario_status": "computed",
    }
    current_form = app.collect_current_form_values(100.0, 0.0, 0.0, 1.0, 1.0, 30, "catboost_full_factors", "economic_extrapolation")
    status = app.get_user_scenario_status(current_form, base_form, applied_snapshot, "applied")
    assert status == "dirty"


def test_enhanced_economic_extrapolation_preserves_price_effect_source():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    horizon = 7
    train_max = float(pd.to_numeric(bundle["daily_base"]["price"], errors="coerce").max())
    requested_price = train_max * 1.4
    dates = pd.date_range(pd.to_datetime(bundle["future_dates"]["date"]).min(), periods=horizon, freq="D")
    wr = run_what_if_projection(
        bundle,
        manual_price=requested_price,
        horizon_days=horizon,
        overrides={"price_path": [{"date": str(d.date()), "value": requested_price} for d in dates]},
        scenario_calc_mode="enhanced_local_factors",
        price_guardrail_mode="economic_extrapolation",
    )
    daily = wr["daily"]
    assert wr["scenario_price_effect_source"] == "boundary_plus_elasticity_tail"
    assert wr["extrapolation_applied"] is True
    assert np.isclose(float(wr["requested_price"]), requested_price)
    assert np.isclose(float(wr["model_price"]), train_max)
    assert np.isclose(float(wr["financial_price"]), requested_price)
    assert wr["price_policy"]["is_path"] is True
    assert np.isclose(float(wr["price_policy"]["requested_price_min"]), requested_price)
    assert np.isclose(float(pd.to_numeric(daily["requested_price_gross"], errors="coerce").iloc[0]), requested_price)
    assert np.isclose(float(pd.to_numeric(daily["model_price_gross"], errors="coerce").iloc[0]), train_max)
    assert np.isclose(float(pd.to_numeric(daily["applied_price_gross"], errors="coerce").iloc[0]), requested_price)
    assert "calculation_trace" in wr and "price_policy" in wr and "factor_policy" in wr
    assert float(pd.to_numeric(daily["actual_sales"], errors="coerce").sum()) < float(pd.to_numeric(daily["base_pred_sales"], errors="coerce").sum())
    first = daily.iloc[0]
    assert np.isclose(float(first["revenue"]), float(first["actual_sales"]) * float(first["applied_price_net"]))


def test_legacy_and_enhanced_share_v1_result_contract_fields():
    res = _analyze_bundle()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    required = {
        "daily",
        "effective_scenario",
        "price_policy",
        "factor_policy",
        "calculation_trace",
        "guardrail_warnings",
        "calculation_gate",
        "recommendation_gate",
        "scenario_calc_mode",
        "scenario_engine_version",
    }
    for mode in ["legacy_current", "enhanced_local_factors"]:
        wr = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode=mode)
        assert required.issubset(wr.keys())
        assert {"mode", "is_path", "requested_price_min", "model_price_min", "financial_price_min"}.issubset(wr["price_policy"].keys())
