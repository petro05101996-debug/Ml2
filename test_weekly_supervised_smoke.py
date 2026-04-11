import numpy as np
import pandas as pd

from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2


def _txn(n=220, weak_controls=False):
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    q = 20 + (pd.Series(range(n)) % 6)
    if weak_controls:
        p = np.full(n, 12.0)
        promo = np.zeros(n)
    else:
        p = 10 + (pd.Series(range(n)) % 9)
        promo = (pd.Series(range(n)) % 14 == 0).astype(float)
    return pd.DataFrame(
        {
            "date": d,
            "product_id": "sku-1",
            "category": "cat",
            "quantity": q,
            "revenue": q * p,
            "price": p,
            "discount_rate": 0.03 + 0.02 * promo,
            "promotion": promo,
            "stock": 100.0,
            "cost": 7.0,
            "freight_value": 1.0,
            "region": "US",
            "channel": "online",
            "segment": "retail",
        }
    )


def test_no_overrides_equals_as_is():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14)
    assert abs(float(out["as_is_forecast"]["actual_sales"].sum()) - float(out["scenario_forecast"]["actual_sales"].sum())) < 1e-6
    diag = out["diagnostic_summary"].iloc[0].to_dict()
    assert bool(diag["manual_scenario_fallback_applied"]) is False
    assert bool(diag["scenario_unchanged_but_forecast_changed"]) is False


def test_price_increase_reduces_or_not_increases_demand():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14, scenario_overrides={"price": 15.0})
    diag = out["diagnostic_summary"].iloc[0].to_dict()
    as_is = float(out["as_is_forecast"]["actual_sales"].sum())
    scn = float(out["scenario_forecast"]["actual_sales"].sum())
    if scn > as_is * 1.01:
        assert bool(diag["price_direction_suspicious"]) is True
    else:
        assert bool(diag["price_monotonicity_violation"]) is False


def test_promo_on_changes_forecast_when_supported():
    base = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14)
    alt = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14, scenario_overrides={"promotion": 1.0})
    assert float(base["scenario_forecast"]["actual_sales"].sum()) != float(alt["scenario_forecast"]["actual_sales"].sum())
    diag = alt["diagnostic_summary"].iloc[0].to_dict()
    assert bool(diag["promo_sensitivity_missing"]) is False


def test_manual_fallback_only_when_model_flat_to_controls():
    out = run_full_pricing_analysis_v2(_txn(weak_controls=True), "cat", "sku-1", horizon_days=14, scenario_overrides={"price": 14.0})
    diag = out["diagnostic_summary"].iloc[0].to_dict()
    assert bool(diag["manual_scenario_fallback_applied"]) is True
    issues = set((out["model_diagnostics"] or {}).get("issues", []))
    assert "scenario_effect_from_manual_fallback" in issues


def test_model_direct_path_detected_when_controls_have_signal():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14, scenario_overrides={"price": 13.0})
    diag = out["diagnostic_summary"].iloc[0].to_dict()
    assert out["scenario_effect_source"] in {"model_direct", "manual_fallback"}
    assert bool(diag["model_direct_sensitivity_present"]) is True or bool(diag["manual_scenario_fallback_applied"]) is True


def test_identical_controls_do_not_trigger_fallback():
    base = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14)
    cur_price = float(base["as_is_forecast"]["price"].iloc[0])
    cur_promo = float(base["as_is_forecast"]["discount"].iloc[0])
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14, scenario_overrides={"price": cur_price, "discount": cur_promo})
    diag = out["diagnostic_summary"].iloc[0].to_dict()
    assert bool(diag["scenario_controls_changed"]) is False
    assert bool(diag["manual_scenario_fallback_applied"]) is False
    assert abs(float(out["as_is_forecast"]["actual_sales"].sum()) - float(out["scenario_forecast"]["actual_sales"].sum())) < 1e-6


def test_stock_cap_generates_lost_sales():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=14, scenario_overrides={"use_stock_cap": True, "stock_total_horizon": 30.0})
    sf = out["scenario_forecast"]
    assert float(sf["lost_sales"].sum()) > 0.0
    assert abs(float((sf["actual_sales"] + sf["lost_sales"] - sf["demand_raw"]).abs().sum())) < 1e-6
