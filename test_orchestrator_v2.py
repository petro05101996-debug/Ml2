import numpy as np
import pandas as pd

from data_adapter import build_auto_mapping, normalize_transactions
import pricing_core.orchestrator_v2 as orch
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
    assert len(out["neutral_baseline_forecast"]) == 7
    assert len(out["as_is_forecast"]) == 7
    assert len(out["scenario_forecast"]) == 7


def test_v2_confidence_has_all_layers():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    c = out["confidence"]
    assert {"baseline_confidence", "factor_confidence", "shock_confidence", "overall_confidence", "intervals_available"}.issubset(c.keys())


def test_v2_excel_contains_required_sheets():
    out = run_full_pricing_analysis_v2(_txn(), "cat", "sku-1", horizon_days=7)
    xls = pd.ExcelFile(out["excel_buffer"])
    required = {"history", "neutral_baseline_forecast", "as_is_forecast", "scenario_forecast", "neutral_baseline_economics", "as_is_economics", "scenario_economics", "delta_summary_current_vs_scenario", "delta_summary_neutral_vs_current", "baseline_rolling_metrics", "baseline_rolling_diag", "baseline_benchmark_suite", "baseline_quality_summary", "baseline_data_quality", "scenario_inputs_echo", "diagnostic_summary", "factor_backtest", "current_state_contributions", "scenario_delta_contributions", "confidence", "confidence_flat"}
    assert required.issubset(set(xls.sheet_names))


def test_v2_uses_final_baseline_model_for_forecast():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    b = out["_trained_bundle"]
    assert b.get("baseline_strategy") in {"xgb_recursive", "median7", "mean28", "dow_median8w", "recent_level_dow_profile", "recent_level_dow_trend", "rolling_dow_regression", "ets_seasonal7", "weekly_median4w", "weekly_recent4_avg", "weekly_mean8w"}
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
    b = out["neutral_baseline_forecast"]["baseline_pred"].reset_index(drop=True)
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
    assert np.allclose(base["neutral_baseline_forecast"]["baseline_pred"].values, alt["neutral_baseline_forecast"]["baseline_pred"].values)
    if base["factor_model_trained"] and alt["factor_model_trained"]:
        assert float(alt["_trained_bundle"]["current_ctx"].get("price", 0.0)) >= 0.0
        assert float(alt["scenario_economics"]["unit_price"].iloc[0]) == 14.0


def test_price_override_changes_scenario_outputs_with_trained_factor():
    base = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=7)
    alt = run_full_pricing_analysis_v2(_txn(260, two_skus=True), "cat", "sku-1", horizon_days=7, scenario_overrides={"price": 15.0})
    assert np.allclose(base["neutral_baseline_forecast"]["baseline_pred"].values, alt["neutral_baseline_forecast"]["baseline_pred"].values)
    if base["factor_model_trained"] and alt["factor_model_trained"]:
        assert float(alt["scenario_economics"]["unit_price"].iloc[0]) == 15.0
        b = base["delta_summary_current_vs_scenario"].iloc[0].to_dict()
        a = alt["delta_summary_current_vs_scenario"].iloc[0].to_dict()
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
    assert "neutral_ctx" in b
    assert "current_ctx" in b
    assert "scenario_feature_spec" in b


def test_v2_outputs_baseline_benchmark_and_quality_gate():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    assert not out["baseline_benchmark_suite"].empty
    assert {"granularity", "acceptance_pass"}.issubset(out["baseline_benchmark_suite"].columns)
    assert {
        "baseline_meets_quality_gate",
        "baseline_goal_wape_median_le_25",
        "baseline_goal_wape_max_le_35",
        "baseline_goal_abs_bias_le_7pct",
        "baseline_goal_sum_ratio_in_range",
        "baseline_goal_std_ratio_ge_055",
    }.issubset(out["baseline_quality_gate"].keys())


def test_final_baseline_selection_is_consistent_across_outputs():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    plan = out["baseline_plan_selection"]
    assert plan["final_selected_strategy"] == out["final_baseline_strategy"]
    assert plan["final_selected_granularity"] == out["final_baseline_granularity"]
    assert out["final_baseline_source"] == "benchmark_suite_selection"
    xls = pd.ExcelFile(out["excel_buffer"])
    summary = pd.read_excel(xls, sheet_name="baseline_quality_summary").iloc[0]
    assert str(summary["baseline_strategy"]) == str(out["final_baseline_strategy"])
    assert str(summary["baseline_granularity"]) == str(out["final_baseline_granularity"])


def test_final_selection_prefers_production_candidate_over_benchmark_only_in_outputs(monkeypatch):
    def _fake_plan(*args, **kwargs):
        return {
            "granularity": "daily",
            "selected_strategy": "recent_level_dow_profile",
            "selector_reason": "plan picked benchmark-only",
            "daily_selection": {"best_strategy": "recent_level_dow_profile", "strategy_metrics": pd.DataFrame(), "strategy_summary": pd.DataFrame()},
            "weekly_selection": {"best_strategy": "weekly_median4w", "strategy_metrics": pd.DataFrame(), "strategy_summary": pd.DataFrame()},
            "best_daily_strategy": "recent_level_dow_profile",
            "best_weekly_strategy": "weekly_median4w",
        }

    def _fake_suite(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "strategy": "recent_level_dow_profile",
                    "granularity": "daily",
                    "median_wape": 12.0,
                    "max_wape": 18.0,
                    "median_bias_pct": 0.01,
                    "median_sum_ratio": 1.0,
                    "median_std_ratio": 0.9,
                    "flat_window_share": 0.1,
                    "median_weekday_shape_error": 0.1,
                    "composite_score": 1.0,
                    "backend_available": True,
                    "fallback_used": False,
                    "guardrail_reject": False,
                    "goal_wape_median_le_25": True,
                    "goal_wape_max_le_35": True,
                    "goal_abs_bias_le_7pct": True,
                    "goal_sum_ratio_in_range": True,
                    "goal_std_ratio_ge_055": True,
                    "acceptance_pass": True,
                    "candidate_tier": "benchmark_only",
                    "winner_scope": "rejected",
                },
                {
                    "strategy": "rolling_dow_regression",
                    "granularity": "daily",
                    "median_wape": 13.0,
                    "max_wape": 19.0,
                    "median_bias_pct": 0.01,
                    "median_sum_ratio": 1.0,
                    "median_std_ratio": 0.9,
                    "flat_window_share": 0.1,
                    "median_weekday_shape_error": 0.1,
                    "composite_score": 1.2,
                    "backend_available": True,
                    "fallback_used": False,
                    "guardrail_reject": False,
                    "goal_wape_median_le_25": True,
                    "goal_wape_max_le_35": True,
                    "goal_abs_bias_le_7pct": True,
                    "goal_sum_ratio_in_range": True,
                    "goal_std_ratio_ge_055": True,
                    "acceptance_pass": True,
                    "candidate_tier": "production_candidate",
                    "winner_scope": "best_available",
                },
            ]
        )

    monkeypatch.setattr(orch, "select_best_baseline_plan", _fake_plan)
    monkeypatch.setattr(orch, "run_baseline_benchmark_suite", _fake_suite)

    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    contract = build_v2_result_contract(out)
    assert out["baseline_selection_result"]["best_available_strategy"] == "rolling_dow_regression"
    assert out["final_baseline_strategy"] == "rolling_dow_regression"
    assert contract["final_baseline_strategy"] == "rolling_dow_regression"
    xls = pd.ExcelFile(out["excel_buffer"])
    summary = pd.read_excel(xls, sheet_name="baseline_quality_summary").iloc[0]
    assert str(summary["baseline_strategy"]) == "rolling_dow_regression"


def test_v2_rolling_export_contains_required_columns():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=5)
    xls = pd.ExcelFile(out["excel_buffer"])
    roll = pd.read_excel(xls, sheet_name="baseline_rolling_metrics")
    diag = pd.read_excel(xls, sheet_name="baseline_rolling_diag")
    assert {"window_id", "window_start", "window_end", "forecast_wape", "mae", "rmse", "bias_pct", "sum_ratio", "pred_std", "actual_std", "std_ratio", "pred_nunique", "actual_nunique", "is_flat_forecast", "weekday_shape_error"}.issubset(roll.columns)
    assert {"date", "sales", "baseline_pred", "window_id", "window_start", "window_end"}.issubset(diag.columns)


def test_cost_and_freight_do_not_change_demand_v2():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    bundle = out["_trained_bundle"]
    base = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, cost_multiplier=1.0, freight_multiplier=1.0)
    stressed = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, cost_multiplier=1.3, freight_multiplier=1.3)
    assert base["scenario_demand_total"] == stressed["scenario_demand_total"]
    assert base["actual_sales_total"] == stressed["actual_sales_total"]
    assert base["lost_sales_total"] == stressed["lost_sales_total"]
    assert base["profit_total"] != stressed["profit_total"]


def test_stock_cap_none_means_unlimited():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    bundle = out["_trained_bundle"]
    w = run_v2_what_if_projection(bundle, manual_price=10.0, horizon_days=7, stock_cap=None)
    assert w["lost_sales_total"] == 0.0
    assert w["actual_sales_total"] == w["scenario_demand_total"]


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
    bf = out["neutral_baseline_forecast"]
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


def test_contract_contains_three_forecast_layers_and_contributions():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    assert "neutral_baseline_forecast" in out
    assert "as_is_forecast" in out
    assert "scenario_forecast" in out
    assert "current_state_contributions" in out
    assert "scenario_delta_contributions" in out
    assert "delta_summary_current_vs_scenario" in out


def test_no_overrides_scenario_equals_as_is():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    a = out["as_is_forecast"]["actual_sales"].reset_index(drop=True)
    s = out["scenario_forecast"]["actual_sales"].reset_index(drop=True)
    assert a.equals(s)


def test_scenario_inputs_echo_marks_no_change_when_no_overrides():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7)
    echo = out["scenario_inputs_echo"]
    if not echo.empty:
        assert (echo["changed_flag"] == False).all()  # noqa: E712
        row = out["delta_summary_current_vs_scenario"].iloc[0]
        assert abs(float(row["demand_delta_pct"])) <= 1e-9
        assert out["scenario_controls_changed"] is False
        assert out["scenario_delta_zero_reason"] == "no_overrides"


def test_delta_summary_current_vs_scenario_pct_is_based_on_as_is():
    out = run_full_pricing_analysis_v2(_txn(220), "cat", "sku-1", horizon_days=7, scenario_overrides={"price": 15.0})
    row = out["delta_summary_current_vs_scenario"].iloc[0]
    expected = float(row["demand_delta_abs"]) / max(float(row["as_is_total_demand"]), 1e-9)
    assert abs(float(row["demand_delta_pct"]) - expected) < 1e-9
