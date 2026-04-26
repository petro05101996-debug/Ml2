import numpy as np
import pandas as pd
import pytest

from scenario_engine import run_scenario
from what_if import build_sensitivity_grid
from v1_runtime_helpers import (
    build_backend_warning,
    compute_scenario_price_inputs,
    get_model_backend_status,
    select_weekly_baseline_candidate,
)


def _baseline_candidates() -> tuple[list[dict], dict, dict]:
    bundle_results = [
        {"name": "legacy_baseline", "holdout_wape": 11.0, "corr": 0.72, "std_ratio": 0.80},
        {"name": "price_only_baseline", "holdout_wape": 12.5, "corr": 0.70, "std_ratio": 0.85},
        {"name": "price_promo_freight_baseline", "holdout_wape": 14.5, "corr": 0.60, "std_ratio": 0.82},
    ]
    bundle_models = {row["name"]: object() for row in bundle_results}
    bundle_features_selected = {row["name"]: ["sales_lag1w"] for row in bundle_results}
    return bundle_results, bundle_models, bundle_features_selected


def test_baseline_selection_does_not_force_price_promo_freight():
    bundle_results, bundle_models, bundle_features_selected = _baseline_candidates()
    selected = select_weekly_baseline_candidate(
        bundle_results=bundle_results,
        bundle_models=bundle_models,
        bundle_features_selected=bundle_features_selected,
        baseline_bundle_name="legacy_baseline",
        nonlegacy_mode="active_production",
        wape_tol_pp=0.5,
        corr_tol=0.05,
        std_ratio_floor=0.02,
        std_ratio_cap=0.80,
    )
    assert selected["selected_candidate_name"] == "legacy_baseline"
    assert "preferred" not in selected["selection_reason"]
    final_active_path = f"{selected['selected_candidate_name']}+scenario_recompute"
    assert final_active_path == "legacy_baseline+scenario_recompute"


def test_baseline_selection_can_choose_nonlegacy_when_rule_passed():
    bundle_results = [
        {"name": "legacy_baseline", "holdout_wape": 11.0, "corr": 0.72, "std_ratio": 0.80},
        {"name": "price_promo_baseline", "holdout_wape": 10.2, "corr": 0.71, "std_ratio": 0.84},
    ]
    bundle_models = {row["name"]: object() for row in bundle_results}
    bundle_features_selected = {row["name"]: ["sales_lag1w"] for row in bundle_results}
    selected = select_weekly_baseline_candidate(
        bundle_results=bundle_results,
        bundle_models=bundle_models,
        bundle_features_selected=bundle_features_selected,
        baseline_bundle_name="legacy_baseline",
        nonlegacy_mode="active_production",
        wape_tol_pp=0.5,
        corr_tol=0.05,
        std_ratio_floor=0.02,
        std_ratio_cap=0.80,
    )
    assert selected["selected_candidate_name"] == "price_promo_baseline"


def test_price_clip_is_applied_to_model_price():
    clip = compute_scenario_price_inputs(requested_price=140.0, train_min=90.0, train_max=110.0)
    assert clip["requested_price"] != clip["model_price"]
    assert clip["price_clipped"] is True
    baseline = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3, freq="D"), "baseline_units": [100.0, 100.0, 100.0]})
    result = run_scenario(
        baseline_output=baseline,
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": clip["model_price"],
            "baseline_net_price": 100.0,
            "scenario_net_price": clip["model_price"],
            "price_elasticity": -1.0,
            "price_elasticity_prior": -1.0,
        },
    )
    expected_effect = np.exp(np.clip(-1.0 * np.log(clip["model_price"] / 100.0), -0.35, 0.35))
    assert np.isclose(float(result["price_effect"]), float(expected_effect))


def test_promo_decrease_reduces_demand():
    baseline = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=2, freq="D"), "baseline_units": [100.0, 100.0]})
    result = run_scenario(
        baseline_output=baseline,
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_intensity_baseline": 0.5,
            "promo_intensity_scenario": 0.1,
        },
    )
    assert float(result["promo_effect"]) < 1.0
    assert float(np.sum(result["final_units"])) < float(np.sum(result["baseline_units"]))


def test_promo_increase_raises_demand():
    baseline = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=2, freq="D"), "baseline_units": [100.0, 100.0]})
    result = run_scenario(
        baseline_output=baseline,
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_intensity_baseline": 0.1,
            "promo_intensity_scenario": 0.5,
        },
    )
    assert float(result["promo_effect"]) > 1.0
    assert float(np.sum(result["final_units"])) > float(np.sum(result["baseline_units"]))


def test_backend_status_is_explicit():
    class DeterministicWeeklyModel:
        pass

    model = DeterministicWeeklyModel()
    status = get_model_backend_status(model)
    warning = build_backend_warning(status["model_backend"], status["backend_reason"])
    assert status["model_backend"] == "deterministic_fallback"
    assert warning != ""


def test_legacy_mode_regression_identical_to_default_path():
    from app import run_full_pricing_analysis_universal, run_what_if_projection
    from test_smoke_mvp import _make_txn

    res = run_full_pricing_analysis_universal(_make_txn(240), "cat-a", "sku-1")
    bundle = res["_trained_bundle"]
    price = float(bundle["base_ctx"]["price"]) * 1.03
    default_run = run_what_if_projection(bundle, manual_price=price, overrides={"promotion": 0.2})
    explicit_legacy = run_what_if_projection(bundle, manual_price=price, overrides={"promotion": 0.2}, scenario_calc_mode="legacy_current")
    assert np.isclose(float(default_run["demand_total"]), float(explicit_legacy["demand_total"]))
    assert np.isclose(float(default_run["revenue_total"]), float(explicit_legacy["revenue_total"]))
    assert np.isclose(float(default_run["profit_total"]), float(explicit_legacy["profit_total"]))
    assert np.isclose(float(default_run["daily"]["actual_sales"].sum()), float(explicit_legacy["daily"]["actual_sales"].sum()))


def test_sensitivity_grid_propagates_runner_kwargs():
    seen = []

    def _runner(_bundle, manual_price, horizon_days=30, demand_multiplier=1.0, scenario_calc_mode="legacy_current"):
        seen.append(scenario_calc_mode)
        return {"profit_total_adjusted": 1.0, "demand_total": 1.0}

    grid = build_sensitivity_grid({}, base_price=100.0, runner=_runner, price_steps=2, demand_steps=2, runner_kwargs={"scenario_calc_mode": "enhanced_local_factors"})
    assert len(grid) == 4
    assert set(seen) == {"enhanced_local_factors"}


def test_manual_scenario_contract_regression_profit_down_recommendation_not_recommended():
    import json

    from app import build_manual_scenario_artifacts, classify_economic_verdict, run_full_pricing_analysis_universal, run_what_if_projection
    from test_smoke_mvp import _make_txn

    res = run_full_pricing_analysis_universal(_make_txn(240), "cat-a", "sku-1")
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(
        bundle,
        manual_price=base_price * 0.95,
        scenario_calc_mode="enhanced_local_factors",
        overrides={"promotion": min(1.0, float(bundle["base_ctx"].get("promotion", 0.0)) + 0.35), "discount": 0.20},
    )
    res["scenario_forecast"] = wr["daily"].copy()
    summary_blob, daily_blob = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    daily = pd.read_csv(pd.io.common.BytesIO(daily_blob))
    assert float(summary["scenario_demand_total"]) > float(summary["as_is_demand_total"])
    assert float(summary["scenario_revenue_total"]) > float(summary["as_is_revenue_total"])
    assert float(summary["scenario_profit_total"]) < float(summary["as_is_profit_total"])
    expected_label, _, _ = classify_economic_verdict(
        float(summary["scenario_vs_as_is_profit_pct"]),
        float(summary["scenario_vs_as_is_demand_pct"]),
        float(summary["scenario_vs_as_is_revenue_pct"]),
    )
    assert expected_label in {"Не рекомендуется", "Невыгоден"}
    assert str(summary["economic_verdict"]) != ""
    assert str(summary["reliability_verdict"]) != ""
    assert "scenario_vs_as_is_revenue_pct" in summary
    run_summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert str(summary["scenario_calculation_path"]) == str(run_summary["config"]["scenario_calculation_path"])
    assert "final_multiplier" in daily.columns


def test_manual_scenario_exports_effect_multipliers_and_verdict_consistently():
    import json

    from app import build_excel_export_buffer, build_manual_scenario_artifacts, run_full_pricing_analysis_universal, run_what_if_projection
    from test_smoke_mvp import _make_txn

    res = run_full_pricing_analysis_universal(_make_txn(240), "cat-a", "sku-1")
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(
        bundle,
        manual_price=base_price,
        scenario_calc_mode="enhanced_local_factors",
        demand_multiplier=0.82,
        overrides={
            "promotion": 0.20,
            "discount": 0.0,
            "shocks": [],
        },
    )
    res["scenario_forecast"] = wr["daily"].copy()
    summary_blob, daily_blob = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    daily = pd.read_csv(pd.io.common.BytesIO(daily_blob))
    assert pd.to_numeric(daily["promo_effect"], errors="coerce").mean() == pytest.approx(1.12, rel=1e-2)
    assert pd.to_numeric(daily["shock_multiplier"], errors="coerce").mean() == pytest.approx(0.82, rel=1e-2)
    assert pd.to_numeric(daily["final_multiplier"], errors="coerce").mean() == pytest.approx(0.9184, rel=1e-2)
    baseline_total = float(pd.to_numeric(daily["baseline_demand"], errors="coerce").sum())
    scenario_total = float(pd.to_numeric(daily["scenario_demand"], errors="coerce").sum())
    assert scenario_total == pytest.approx(baseline_total * 0.9184, rel=3e-2)

    excel_blob = build_excel_export_buffer(res, wr)
    user_summary = pd.read_excel(excel_blob, sheet_name="User Scenario Summary")
    assert "Сценарий не применён" not in str(user_summary.loc[0, "Вывод"])
    scenario_summary = pd.read_excel(excel_blob, sheet_name="Scenario Summary")
    assert int(scenario_summary.loc[0, "manual_shocks"]) >= 1
    assert summary["manual_scenario_present"] is True
