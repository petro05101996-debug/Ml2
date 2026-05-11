import pytest

from recommendation_gate import evaluate_price_monotonic_sanity, resolve_recommendation_gate


def test_competitor_price_not_constrained_as_own_price():
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    from catboost_full_factor_engine import _build_monotone_constraints

    features = ["price", "net_unit_price", "competitor_price", "market_price", "factor__benchmark_price"]
    assert _build_monotone_constraints(features) == [-1, -1, 0, 0, 0]


def test_recommendation_gate_blocks_high_wape_and_cost_missing_profit():
    gate = resolve_recommendation_gate(
        model_quality={"wape": 65.0},
        economic_significance={"profit_delta_pct": 10.0, "profit_action": True, "cost_missing": True},
        cost_policy={"cost_missing": True, "profit_action": True},
        decision_reliability={"status_namespace": "decision", "base_status": "recommended"},
    )
    assert gate["decision_status"] == "not_recommended"
    assert gate["usage_policy"]["can_recommend_action"] is False
    assert "cost_missing_blocks_profit_recommendation" in gate["reasons"]


def test_stockout_and_naive_baseline_limit_recommendation():
    gate = resolve_recommendation_gate(
        model_quality={"wape": 12.0, "stockout_share": 0.2, "naive_improvement_pct": 2.0},
        economic_significance={"profit_delta_pct": 8.0, "profit_action": True},
        decision_reliability={"status_namespace": "decision", "base_status": "recommended"},
    )
    assert gate["decision_status"] == "experimental_only"
    assert "stockout_share_15_30_experimental" in gate["reasons"]
    assert "model_not_better_than_naive_baseline" in gate["reasons"]


def test_price_monotonicity_pure_price_violation_detected():
    out = evaluate_price_monotonic_sanity(100.0, 110.0, 1000.0, 1100.0)
    assert out["status"] == "failed"
    assert "price_monotonicity_failed" in out["blockers"]


def test_recommendation_gate_blocks_unknown_wape():
    gate = resolve_recommendation_gate(
        model_quality={"wape": float("nan")},
        economic_significance={"profit_delta_pct": 8.0, "profit_action": True},
        decision_reliability={"status_namespace": "decision", "base_status": "recommended"},
    )
    assert gate["decision_status"] == "not_recommended"
    assert "wape_unknown_not_recommended" in gate["reasons"]


def test_stockout_above_thirty_blocks_recommendation():
    gate = resolve_recommendation_gate(
        model_quality={"wape": 12.0, "stockout_share": 0.31, "naive_improvement_pct": 20.0},
        economic_significance={"profit_delta_pct": 8.0, "profit_action": True},
        decision_reliability={"status_namespace": "decision", "base_status": "recommended"},
    )
    assert gate["decision_status"] == "not_recommended"
    assert "stockout_share_above_30_not_recommended" in gate["reasons"]


def test_recommended_mode_status_requires_finite_quality_and_stable_rolling():
    pytest.importorskip("numpy")
    from app import build_recommended_mode_status, CATBOOST_FULL_FACTOR_MODE

    unknown = build_recommended_mode_status(
        CATBOOST_FULL_FACTOR_MODE,
        {"enabled": True, "holdout_metrics": {"wape": float("nan"), "naive_improvement_pct": 50.0}, "rolling_retrain_backtest": {"verdict": "stable"}},
    )
    assert unknown["status"] == "quality_unknown"
    assert unknown["recommended_mode"] == "enhanced_local_factors"

    moderate = build_recommended_mode_status(
        CATBOOST_FULL_FACTOR_MODE,
        {"enabled": True, "holdout_metrics": {"wape": 12.0, "naive_improvement_pct": 20.0}, "rolling_retrain_backtest": {"verdict": "moderately_stable"}},
    )
    assert moderate["status"] == "recommended"
    assert moderate["recommended_mode"] == CATBOOST_FULL_FACTOR_MODE

    unstable = build_recommended_mode_status(
        CATBOOST_FULL_FACTOR_MODE,
        {"enabled": True, "holdout_metrics": {"wape": 12.0, "naive_improvement_pct": 20.0}, "rolling_retrain_backtest": {"verdict": "unstable_test_only"}},
    )
    assert unstable["status"] == "test_recommended"
    assert unstable["recommended_mode"] == "enhanced_local_factors"


def test_leakage_features_are_excluded_from_catboost_factor_catalog():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from catboost_full_factor_engine import _infer_feature_columns, _build_model_frame

    frame = _build_model_frame(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8),
                "sales": np.arange(8) + 10,
                "price": [100, 101, 102, 103, 104, 105, 106, 107],
                "factor__future_sales": np.arange(8) + 20,
                "factor__weather_index": [1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
    )
    features, _cats, report = _infer_feature_columns(frame)
    assert "factor__future_sales" not in features
    row = report.loc[report["feature"] == "factor__future_sales"].iloc[0]
    assert bool(row["used_in_active_model"]) is False
    assert row["reason"] == "target_leakage_risk"


def test_recommendation_gate_allows_unknown_wape_for_decision_support_test_only():
    gate = resolve_recommendation_gate(
        model_quality={"wape": float("nan")},
        economic_significance={"profit_delta_pct": 8.0, "profit_action": True},
        decision_reliability={
            "status_namespace": "decision",
            "base_status": "recommended",
            "allow_unknown_wape_for_test_recommendation": True,
        },
    )
    assert gate["decision_status"] == "test_recommended"
    assert "wape_unknown_test_only" in gate["reasons"]


def test_high_wape_blocks_recommendation_but_keeps_what_if_visible():
    gate = resolve_recommendation_gate(
        model_quality={"wape": 65.0},
        economic_significance={"profit_delta_pct": 8.0, "profit_action": True},
        decision_reliability={"status_namespace": "decision", "base_status": "recommended"},
    )
    assert gate["decision_status"] == "not_recommended"
    assert gate["usage_policy"]["can_recommend_action"] is False
    assert gate["usage_policy"]["can_show_what_if"] is True


def test_scenario_audit_handles_none_catboost_bundle():
    from scenario_audit import build_scenario_reproducibility_id

    out = build_scenario_reproducibility_id(
        {"daily_base": "minimal", "catboost_full_factor_bundle": None},
        {"manual_price": 100},
        "enhanced_local_factors",
        "safe_clip",
        "v1",
        "code",
    )
    assert out["scenario_run_id"]
    assert out["config_hash"]


def test_catboost_prepare_base_columns_tolerates_missing_economic_columns():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from catboost_full_factor_engine import _prepare_base_columns, _infer_feature_columns

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12),
            "sales": np.arange(12) + 10,
            "price": np.linspace(90, 110, 12),
            "discount": [0.0, 0.1] * 6,
        }
    )
    out = _prepare_base_columns(frame)
    for col in ["cost", "stock", "revenue", "freight_value"]:
        assert col in out.columns
    features, _cats, report = _infer_feature_columns(out)
    assert "cost" not in features
    assert "stock" not in features
    assert "revenue" not in features
    assert set(report.loc[report["feature"].isin(["cost", "stock"]), "reason"]) == {"excluded_by_feature_registry"}


def test_discount_semantics_preserve_source_column_after_mapping():
    pd = pytest.importorskip("pandas")
    from data_adapter import normalize_transactions

    base = {"date": ["2024-01-01"], "product_id": ["A"], "price": [200.0], "quantity": [1.0]}
    amount_df = pd.DataFrame({**base, "discount_amount": [20.0]})
    amount, amount_q = normalize_transactions(amount_df, {"date": "date", "product_id": "product_id", "price": "price", "quantity": "quantity", "discount": "discount_amount"})
    assert float(amount["discount"].iloc[0]) == pytest.approx(0.10)
    assert float(amount["revenue"].iloc[0]) == pytest.approx(180.0)
    assert amount_q["source_contract"]["discount_source"] == "provided_discount_amount_normalized"

    pct_df = pd.DataFrame({**base, "discount_pct": [20.0]})
    pct, _ = normalize_transactions(pct_df, {"date": "date", "product_id": "product_id", "price": "price", "quantity": "quantity", "discount": "discount_pct"})
    assert float(pct["discount"].iloc[0]) == pytest.approx(0.20)
    assert float(pct["revenue"].iloc[0]) == pytest.approx(160.0)

    rate_df = pd.DataFrame({**base, "discount_rate": [0.2]})
    rate, _ = normalize_transactions(rate_df, {"date": "date", "product_id": "product_id", "price": "price", "quantity": "quantity", "discount": "discount_rate"})
    assert float(rate["discount"].iloc[0]) == pytest.approx(0.20)
    assert float(rate["revenue"].iloc[0]) == pytest.approx(160.0)


def test_combined_candidates_respect_allowed_actions_price_only():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from decision_candidate_engine import generate_decision_candidates

    bundle = {
        "base_ctx": {"price": 100.0, "discount": 0.1, "promotion": 0.0, "freight_value": 10.0},
        "daily_base": pd.DataFrame({"price": np.linspace(90, 110, 30), "sales": 10, "discount": 0.1, "promotion": [0, 1] * 15, "freight_value": 10.0}),
    }
    candidates = generate_decision_candidates(bundle, None, allowed_actions=["price_change"])
    for candidate in candidates:
        if candidate["action_type"] == "combined_change":
            overrides = candidate.get("scenario_params", {}).get("overrides", {})
            assert overrides.get("discount") == bundle["base_ctx"]["discount"]
            assert overrides.get("promotion") == bundle["base_ctx"]["promotion"]
            assert overrides.get("freight_value") == bundle["base_ctx"]["freight_value"]
    assert not any(c["action_type"] == "combined_change" for c in candidates)


def test_business_constraints_block_price_below_cost_and_large_change():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from decision_candidate_engine import evaluate_decision_candidates

    bundle = {
        "base_ctx": {"price": 100.0, "discount": 0.0, "promotion": 0.0, "freight_value": 0.0, "cost": 95.0},
        "daily_base": pd.DataFrame({"price": np.linspace(90, 110, 30), "sales": 10}),
    }
    candidates = [
        {"candidate_id": "baseline", "action_type": "baseline", "current_value": 100.0, "target_value": 100.0, "scenario_params": {"manual_price": 100.0}, "objective": "profit"},
        {"candidate_id": "bad_price", "action_type": "price_change", "current_value": 100.0, "target_value": 80.0, "scenario_params": {"manual_price": 80.0}, "objective": "profit"},
    ]

    def runner(_bundle, manual_price, **kwargs):
        demand = 100.0
        revenue = demand * manual_price
        profit = demand * (manual_price - 95.0)
        return {"demand_total": demand, "revenue_total": revenue, "profit_total": profit, "cost": 95.0, "applied_price_gross": manual_price}

    out = evaluate_decision_candidates(
        {"history_daily": bundle["daily_base"], "quality_report": {"holdout_metrics": {"wape": 10}}, "model_quality_gate": {"status": "production_allowed"}},
        bundle,
        candidates,
        runner,
        "enhanced_local_factors",
        "safe_clip",
        30,
        "profit",
        constraints={"forbid_price_below_cost": True, "max_price_change_pct": 10.0, "min_expected_profit_uplift_pct": 3.0},
    )
    bad = next(e for e in out if e["candidate"]["candidate_id"] == "bad_price")
    assert bad["reliability"]["decision_status"] == "not_recommended"
    assert any("forbid_price_below_cost" in b or "max_price_change_pct" in b for b in bad["blockers"])


def test_safe_lag_features_are_not_removed_as_leakage():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from catboost_full_factor_engine import _build_model_frame, _infer_feature_columns

    frame = _build_model_frame(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=60),
                "sales": 10 + np.arange(60) % 9,
                "price": np.linspace(90, 110, 60),
                "factor__future_sales": np.arange(60) + 100,
            }
        )
    )
    features, _cats, report = _infer_feature_columns(frame)
    assert "sales_lag_1" in features
    assert "sales_roll_7" in features
    assert "sales_ewm_14" in features
    assert "factor__future_sales" not in features
    assert report.loc[report["feature"] == "factor__future_sales", "reason"].iloc[0] == "target_leakage_risk"


def test_combined_candidates_require_single_action_support_for_promo():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from decision_candidate_engine import generate_decision_candidates

    bundle = {
        "base_ctx": {"price": 100.0, "discount": 0.05, "promotion": 0.0, "freight_value": 10.0},
        "daily_base": pd.DataFrame({"price": np.linspace(90, 110, 40), "sales": 10, "discount": [0.0, 0.05, 0.1, 0.15] * 10, "promotion": 0.0, "freight_value": [8, 9, 10, 11] * 10}),
    }
    candidates = generate_decision_candidates(bundle, None, allowed_actions=["price_change", "promotion_change"])
    assert not any(c["action_type"] == "promotion_change" for c in candidates)
    assert not any(c["action_type"] == "combined_change" and (c.get("scenario_params", {}).get("overrides", {}).get("promotion") == 1.0) for c in candidates)


def test_combined_factor_support_uses_weakest_changed_factor():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from decision_reliability import _factor_support

    history = pd.DataFrame({"price": np.linspace(90, 110, 60), "sales": 10 + np.arange(60) % 5, "promotion": 0.0, "discount": 0.0, "freight_value": 10.0})
    candidate = {
        "action_type": "combined_change",
        "scenario_params": {"manual_price": 95.0, "overrides": {"promotion": 1.0, "discount": 0.0, "freight_value": 10.0}},
        "metadata": {"model_features": ["price"], "scenario_layer_features": ["price", "promotion"], "combined_actions": {"price_change_pct": -5.0, "promotion_delta_pp": 100.0, "discount_delta_pp": 0.0, "freight_change_pct": 0.0}},
    }
    score, details, warnings, blockers = _factor_support(candidate, history, {})
    assert score <= 35.0
    assert "promotion" in details["component_support"]
    assert any("promotion" in w for w in warnings)


def test_require_cost_for_profit_blocks_proxy_cost():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from decision_candidate_engine import evaluate_decision_candidates

    bundle = {"base_ctx": {"price": 100.0, "cost": 65.0, "cost_source": "proxy_price_65"}, "daily_base": pd.DataFrame({"price": np.linspace(90, 110, 20), "sales": 10})}
    candidates = [
        {"candidate_id": "baseline", "action_type": "baseline", "current_value": 100.0, "target_value": 100.0, "scenario_params": {"manual_price": 100.0}, "objective": "profit"},
        {"candidate_id": "up", "action_type": "price_change", "current_value": 100.0, "target_value": 105.0, "scenario_params": {"manual_price": 105.0}, "objective": "profit"},
    ]

    def runner(_bundle, manual_price, **kwargs):
        return {"demand_total": 100.0, "revenue_total": manual_price * 100.0, "profit_total": (manual_price - 65.0) * 100.0, "cost": 65.0, "cost_proxied": True, "cost_source": "proxy_price_65", "profit_is_reliable": False}

    out = evaluate_decision_candidates({"history_daily": bundle["daily_base"], "quality_report": {"holdout_metrics": {"wape": 10}}}, bundle, candidates, runner, "enhanced_local_factors", "safe_clip", 30, "profit", constraints={"require_cost_for_profit": True})
    rec = next(e for e in out if e["candidate"]["candidate_id"] == "up")
    assert rec["reliability"]["decision_status"] == "not_recommended"
    assert any("proxy_cost" in b or "non_provided_cost" in b or "unreliable_profit" in b for b in rec["blockers"])


def test_quantity_missing_inferred_or_blocks_production():
    pd = pytest.importorskip("pandas")
    from data_adapter import normalize_transactions

    inferred, q1 = normalize_transactions(pd.DataFrame({"date": ["2024-01-01"], "product_id": ["A"], "price": [100.0], "revenue": [250.0]}), {"date": "date", "product_id": "product_id", "price": "price", "revenue": "revenue"})
    assert float(inferred["quantity"].iloc[0]) == pytest.approx(2.5)
    assert q1["source_contract"]["quantity_source"] == "inferred_revenue_div_net_price"
    assert "quantity_missing_not_inferable" not in q1["blockers"]

    defaulted, q2 = normalize_transactions(pd.DataFrame({"date": ["2024-01-01"], "product_id": ["A"], "price": [100.0]}), {"date": "date", "product_id": "product_id", "price": "price"})
    assert float(defaulted["quantity"].iloc[0]) == pytest.approx(1.0)
    assert q2["source_contract"]["quantity_source"] == "default_1_blocked"
    assert "quantity_missing_not_inferable" in q2["blockers"]


def test_safe_option_requires_eligible_gate():
    from decision_optimizer import select_decision_options

    ranked = [
        {"candidate": {"candidate_id": "exp"}, "expected_effect": {"profit_delta_pct": 20.0, "conservative_profit_delta_pct": 20.0}, "reliability": {"decision_status": "experimental_only", "risk_level": "low", "score": 95, "economic_significance": {"conservative_profit_delta_pct": 20.0}}}
    ]
    out = select_decision_options(ranked, objective="profit")
    assert out["best_action"] is None
    assert out["safe_option"] is None


def test_external_discount_recommendation_requires_target():
    pytest.importorskip("numpy")
    from recommendation_auditor import build_candidate_from_recommendation

    cand = build_candidate_from_recommendation({"action_type": "discount_change", "comment": "increase discount"}, {"base_ctx": {"price": 100.0, "discount": 0.1}}, None, 14)
    assert cand["target_value"] == pytest.approx(0.1)
    assert "target_value_required" in cand["blockers"]
    assert cand["scenario_params"]["overrides"]["discount"] == pytest.approx(0.1)
