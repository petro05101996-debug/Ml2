import pandas as pd

from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2


def _txn(n=180, with_user_cat=False):
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    q = 20 + (pd.Series(range(n)) % 5)
    p = 10 + (pd.Series(range(n)) % 9)
    df = pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "quantity": q, "revenue": q * p, "price": p, "discount_rate": 0.05, "promotion": 0.0, "stock": 100.0, "cost": 7.0, "freight_value": 1.0, "review_score": 4.2, "reviews_count": 20, "region": "US", "channel": "online", "segment": "retail"})
    if with_user_cat:
        df["user_factor_cat__campaign"] = "A"
    return df


def test_v2_pipeline_runs_end_to_end():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    assert out["analysis_engine"] == "v2_decomposed_baseline_factor_shock"


def test_v2_returns_baseline_and_scenario_outputs():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    assert len(out["baseline_forecast"]) == 7
    assert len(out["scenario_forecast"]) == 7


def test_v2_confidence_has_all_layers():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    c = out["confidence"]
    assert {"baseline_confidence", "factor_confidence", "shock_confidence", "overall_confidence", "intervals_available"}.issubset(c.keys())


def test_v2_does_not_require_recommendation_block():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    assert "recommendation" not in out


def test_v2_excel_contains_required_sheets():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    xls = pd.ExcelFile(out["excel_buffer"])
    required = {"history", "baseline_forecast", "scenario_forecast", "baseline_economics", "scenario_economics", "delta_summary", "baseline_rolling_metrics", "baseline_rolling_diag", "factor_backtest", "factor_contributions", "confidence"}
    assert required.issubset(set(xls.sheet_names))


def test_v2_factor_fallback_to_multiplier_one():
    out = run_full_pricing_analysis_v2(_txn(30), "cat", "sku-1", horizon_days=5)
    assert (out["scenario_forecast"]["factor_multiplier"] == 1.0).all()


def test_v2_uses_final_baseline_model_for_forecast_not_backtest_model():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    b = out["_trained_bundle"]
    assert b.get("trained_baseline_bt") is not None
    assert b.get("trained_baseline_final") is not None
    assert b["trained_baseline_bt"]["training_profile"] == "backtest"
    assert b["trained_baseline_final"]["training_profile"] == "final"


def test_tiny_history_fallback_consistent_between_baseline_and_scenario():
    out = run_full_pricing_analysis_v2(_txn(10), "cat", "sku-1", horizon_days=5)
    b = out["baseline_forecast"]["baseline_pred"].reset_index(drop=True)
    s = out["scenario_forecast"]["baseline_pred"].reset_index(drop=True)
    assert b.equals(s)


def test_factor_ood_flags_reduce_confidence():
    out = run_full_pricing_analysis_v2(_txn(220, with_user_cat=True), "cat", "sku-1", horizon_days=5, scenario_overrides={"price": 9999, "user_factor_cat__campaign": "NEW"})
    issues = out["confidence"]["factor_confidence"]["issues"]
    assert any("ood" in x for x in issues)
