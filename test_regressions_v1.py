import numpy as np
import pandas as pd

from scenario_engine import run_scenario
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
