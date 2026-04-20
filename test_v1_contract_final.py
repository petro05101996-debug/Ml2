import copy
import io
import json

import numpy as np
import pandas as pd

from app import (
    build_applied_scenario_snapshot,
    build_business_report_payload,
    build_manual_scenario_artifacts,
    build_saved_scenario_metrics,
    build_trust_block,
    get_user_scenario_status,
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
