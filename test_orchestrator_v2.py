import numpy as np
import pandas as pd

from data_adapter import build_auto_mapping, normalize_transactions
from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2
from pricing_core.v2_presenter import build_v2_result_contract
from pricing_core.v2_what_if import run_v2_what_if_projection


def _txn(n=180, with_user_cat=False, two_skus=False):
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    q = 20 + (pd.Series(range(n)) % 5)
    p = 10 + (pd.Series(range(n)) % 9)
    df = pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "quantity": q, "revenue": q * p, "price": p, "discount_rate": 0.05, "promotion": 0.0, "stock": 100.0, "cost": 7.0, "freight_value": 1.0, "review_score": 4.2, "reviews_count": 20, "region": "US", "channel": "online", "segment": "retail"})
    if with_user_cat:
        df["user_factor_cat__campaign"] = "A"
    if two_skus:
        df2 = df.copy()
        df2["product_id"] = "sku-2"
        df2["quantity"] = df2["quantity"] + 2
        df2["revenue"] = df2["quantity"] * (df2["price"] + 2)
        return pd.concat([df, df2], ignore_index=True)
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


def test_v2_excel_contains_required_sheets():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    xls = pd.ExcelFile(out["excel_buffer"])
    required = {"history", "baseline_forecast", "scenario_forecast", "baseline_economics", "scenario_economics", "delta_summary", "baseline_rolling_metrics", "baseline_rolling_diag", "factor_backtest", "factor_contributions", "confidence"}
    assert required.issubset(set(xls.sheet_names))


def test_v2_uses_final_baseline_model_for_forecast():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    b = out["_trained_bundle"]
    assert b.get("baseline_strategy") in {"xgb_recursive", "median7", "mean28", "dow_median8w", "weekly_median4w", "weekly_recent4_avg", "weekly_mean8w"}
    if b.get("baseline_strategy") == "xgb_recursive" and b.get("baseline_granularity") != "weekly":
        assert b.get("trained_baseline_bt") is not None
        assert b.get("trained_baseline_final") is not None
        assert b["trained_baseline_bt"]["training_profile"] == "backtest"
        assert b["trained_baseline_final"]["training_profile"] == "final"
    else:
        assert b.get("trained_baseline_bt") is None
        assert b.get("trained_baseline_final") is None
    assert b["baseline_feature_spec_train"] is not None
    assert b["baseline_feature_spec_full"] is not None


def test_tiny_history_fallback_consistent_between_baseline_and_scenario():
    out = run_full_pricing_analysis_v2(_txn(10), "cat", "sku-1", horizon_days=5)
    b = out["baseline_forecast"]["baseline_pred"].reset_index(drop=True)
    s = out["scenario_forecast"]["baseline_pred"].reset_index(drop=True)
    assert b.equals(s)


def test_factor_model_can_train_after_dense_oof():
    out = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=7)
    assert out["factor_train_rows"] >= 45
    assert out["factor_train_scope"] in {"target", "pooled"}


def test_factor_ood_flags_reduce_confidence():
    out = run_full_pricing_analysis_v2(_txn(220, with_user_cat=True), "cat", "sku-1", horizon_days=5, scenario_overrides={"price": 9999, "user_factor_cat__campaign": "NEW"})
    issues = out["confidence"]["factor_confidence"]["issues"]
    assert len(out["ood_flags"]) >= 1
    assert any("ood" in x for x in issues)


def test_price_override_changes_multiplier_with_trained_factor():
    base = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=5)
    alt = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=5, scenario_overrides={"price": 14.0})
    assert np.allclose(base["baseline_forecast"]["baseline_pred"].values, alt["baseline_forecast"]["baseline_pred"].values)
    if base["factor_model_trained"] and alt["factor_model_trained"]:
        assert float(alt["_trained_bundle"]["base_ctx"].get("price", 0.0)) >= 0.0
        assert float(alt["scenario_economics"]["unit_price"].iloc[0]) == 14.0


def test_price_override_changes_scenario_outputs_with_trained_factor():
    base = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=7)
    alt = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=7, scenario_overrides={"price": 15.0})
    assert np.allclose(base["baseline_forecast"]["baseline_pred"].values, alt["baseline_forecast"]["baseline_pred"].values)
    if base["factor_model_trained"] and alt["factor_model_trained"]:
        assert float(alt["scenario_economics"]["unit_price"].iloc[0]) == 15.0
        b = base["delta_summary"].iloc[0].to_dict()
        a = alt["delta_summary"].iloc[0].to_dict()
        assert (b["revenue_delta_pct"] != a["revenue_delta_pct"]) or (b["profit_delta_pct"] != a["profit_delta_pct"])


def test_v2_result_contract_not_legacy_recommendation():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    contract = build_v2_result_contract(out)
    assert "headline_action" in contract
    assert "best_price" not in contract
    assert "current_price" not in contract
    assert contract["mode"] in {"baseline_only", "baseline_plus_scenario", "fallback_elasticity"}


def test_tiny_mode_what_if_does_not_fail():
    out = run_full_pricing_analysis_v2(_txn(10), "cat", "sku-1", horizon_days=5)
    from pricing_core.v2_what_if import run_v2_what_if_projection
    r = run_v2_what_if_projection(out["_trained_bundle"], manual_price=10.0, horizon_days=3)
    assert "profit_total" in r


def test_orchestrator_targets_single_series_id():
    tx = _txn(180)
    tx2 = tx.copy()
    tx2["region"] = "EU"
    tx2["quantity"] = tx2["quantity"] + 7
    tx2["revenue"] = tx2["quantity"] * tx2["price"]
    all_tx = pd.concat([tx, tx2], ignore_index=True)
    target_series_id = "sku-1|EU|online|retail"
    out = run_full_pricing_analysis_v2(all_tx, "cat", "sku-1", target_series_id=target_series_id, horizon_days=7)
    assert out["target_series_id"] == target_series_id
    assert out["_trained_bundle"]["target_history"]["series_id"].astype(str).nunique() == 1
    assert out["_trained_bundle"]["target_history"]["series_id"].astype(str).iloc[0] == target_series_id


def test_economics_mode_propagates_end_to_end():
    out = run_full_pricing_analysis_v2(_txn(180), "cat", "sku-1", horizon_days=5, unit_price_input_type="list", economics_mode="list_less_discount")
    assert out["economics_mode"] == "list_less_discount"
    assert out["unit_price_input_type"] == "list"
    assert out["_trained_bundle"]["economics_mode"] == "list_less_discount"


def test_v2_bundle_contains_base_ctx_and_scenario_feature_spec():
    out = run_full_pricing_analysis_v2(_txn(20), "cat", "sku-1", horizon_days=5)
    b = out["_trained_bundle"]
    assert "base_ctx" in b
    assert "scenario_feature_spec" in b


def test_cost_and_freight_do_not_change_demand_v2():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    bundle = out["_trained_bundle"]
    base = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, cost_multiplier=1.0, freight_multiplier=1.0)
    stressed = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, cost_multiplier=1.3, freight_multiplier=1.3)
    assert base["demand_total"] == stressed["demand_total"]
    assert base["actual_sales_total"] == stressed["actual_sales_total"]
    assert base["lost_sales_total"] == stressed["lost_sales_total"]
    assert base["profit_total"] != stressed["profit_total"]


def test_stock_cap_none_means_unlimited():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    bundle = out["_trained_bundle"]
    w = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, stock_cap=None)
    assert w["lost_sales_total"] == 0.0
    assert w["actual_sales_total"] == w["demand_total"]


def test_stock_cap_zero_means_zero_inventory():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    bundle = out["_trained_bundle"]
    w = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, stock_cap=0.0)
    assert w["actual_sales_total"] == 0.0
    assert w["lost_sales_total"] == w["demand_total"]


def test_sample_transactions_end_to_end_v2_runs():
    tx = pd.read_csv("sample_transactions.csv")
    mapping = build_auto_mapping(list(tx.columns))
    normalized, _ = normalize_transactions(tx, mapping)
    target_row = normalized.dropna(subset=["category", "product_id"]).iloc[0]
    out = run_full_pricing_analysis_v2(normalized, str(target_row["category"]), str(target_row["product_id"]), horizon_days=7)
    assert "baseline_forecast" in out
    assert "scenario_forecast" in out
    assert "overall_confidence" in out["confidence"]


def test_weekly_baseline_plan_returns_daily_forecast_contract():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=9)
    bf = out["baseline_forecast"]
    assert {"date", "baseline_pred"}.issubset(bf.columns)
    assert len(bf) == 9


def test_weekly_baseline_oof_is_daily_and_non_empty():
    out = run_full_pricing_analysis_v2(_txn(280, two_skus=True), "cat", "sku-1", horizon_days=7)
    bundle = out["_trained_bundle"]
    oof = bundle.get("baseline_oof", pd.DataFrame())
    assert {"date", "baseline_oof"}.issubset(oof.columns)
    if str(out.get("baseline_granularity", "daily")) == "weekly":
        assert oof["baseline_oof"].notna().sum() > 0
        assert oof["date"].nunique() > 0
        assert bundle.get("factor_train_rows", 0) >= 0


def test_end_to_end_v2_runs_with_weekly_baseline_selection():
    out = run_full_pricing_analysis_v2(_txn(300, two_skus=True), "cat", "sku-1", horizon_days=7)
    assert "baseline_forecast" in out
    assert "scenario_forecast" in out
    assert out.get("baseline_granularity") in {"daily", "weekly"}
