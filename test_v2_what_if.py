import pandas as pd

from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2
from pricing_core.v2_what_if import run_v2_what_if_projection


def _txn(n=220):
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    q = 20 + (pd.Series(range(n)) % 5)
    p = 10 + (pd.Series(range(n)) % 9)
    return pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "quantity": q, "revenue": q * p, "price": p, "discount_rate": 0.05, "promotion": 0.0, "stock": 100.0, "cost": 7.0, "freight_value": 1.0, "region": "US", "channel": "online", "segment": "retail"})


def test_what_if_compares_scenario_vs_as_is():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    r = run_v2_what_if_projection(out["_trained_bundle"], manual_price=10.0, horizon_days=7)
    assert "as_is_demand_total" in r
    assert "scenario_demand_total" in r
    assert r["demand_delta_abs"] == r["scenario_demand_total"] - r["as_is_demand_total"]


def test_what_if_stock_cap_applies_and_economics_consistent():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    b = out["_trained_bundle"]
    w0 = run_v2_what_if_projection(b, manual_price=10.0, horizon_days=7, stock_cap=0.0)
    assert w0["scenario_demand_total"] == 0.0
    w1 = run_v2_what_if_projection(b, manual_price=10.0, horizon_days=7, stock_cap=None)
    assert w1["scenario_profit_total"] == w1["profit_total"]


def test_what_if_as_is_side_respects_stock_lost_logic():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    b = out["_trained_bundle"]
    w = run_v2_what_if_projection(b, manual_price=10.0, horizon_days=7, stock_cap=0.0)
    assert w["lost_sales_total"] >= 0.0
    assert w["as_is_demand_total"] >= 0.0


def test_what_if_confidence_propagation_and_factor_role():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    b = out["_trained_bundle"]
    r = run_v2_what_if_projection(b, manual_price=10.0, horizon_days=7)
    assert r["confidence_label"] == out["confidence"]["overall_confidence"]
    assert "factor_role" in r
    assert r["scenario_mode"] == "weekly_native"
