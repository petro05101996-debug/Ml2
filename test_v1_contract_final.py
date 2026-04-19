import copy
import json

import numpy as np

from app import (
    build_business_report_payload,
    resolve_final_active_path,
    run_full_pricing_analysis_universal,
    run_what_if_projection,
)
from test_smoke_mvp import _make_txn


def _analyze_bundle():
    return run_full_pricing_analysis_universal(_make_txn(240), "cat-a", "sku-1")


def test_active_path_is_frozen_to_legacy_baseline():
    res = _analyze_bundle()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert summary["config"]["selected_candidate"] == "legacy_baseline"
    assert summary["config"]["production_selected_candidate"] == "legacy_baseline"
    assert "diagnostic_selected_candidate" in summary["config"]
    assert summary["config"]["selection_mode"] == "diagnostic_comparison_runtime_frozen_to_legacy"
    assert summary["config"]["production_selection_reason"] == "v1_contract_runtime_frozen_to_legacy"
    assert summary["config"]["final_active_path"] == "legacy_baseline+scenario_recompute"


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
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.05)
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
