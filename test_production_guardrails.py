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
