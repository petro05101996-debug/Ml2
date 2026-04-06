from recommendation import build_business_recommendation


def _base_kwargs():
    return dict(
        current_price=100.0,
        recommended_price=110.0,
        current_profit=10000.0,
        recommended_profit=12000.0,
        confidence=0.9,
        elasticity=-1.0,
        history_days=200,
        current_revenue=50000.0,
        recommended_revenue=51000.0,
        current_volume=1000.0,
        recommended_volume=980.0,
        data_quality={"level": "good"},
        base_ctx={"price": 100.0},
        reason_hints={},
    )


def test_predictive_gate_medium_baseline_forces_no_decision():
    rec = build_business_recommendation(
        **_base_kwargs(),
        predictive_gate={
            "baseline_confidence": "medium",
            "factor_role": "production",
            "scenario_outside_factor_backtest_range": False,
            "scenario_equals_current_but_delta_nonzero": False,
            "baseline_is_flat_forecast": False,
            "explainability_unavailable": False,
        },
    )
    assert rec["decision_layer"]["decision_type"] == "no_decision"
    assert rec["decision_layer"]["implementation_mode"] == "do_not_change"


def test_predictive_gate_empty_allows_normal_decision_path():
    rec = build_business_recommendation(**_base_kwargs(), predictive_gate={})
    assert rec["decision_layer"]["decision_type"] in {"action", "test", "hold", "no_decision"}
    assert "primary_lever" in rec["decision_layer"]


def test_predictive_hard_block_contract_is_full_for_ui():
    rec = build_business_recommendation(**_base_kwargs(), predictive_gate={"baseline_confidence": "low", "factor_role": "advisory_only"})
    d = rec["decision_layer"]
    required = {
        "decision_type",
        "implementation_mode",
        "decision_strength",
        "primary_lever",
        "test_plan",
        "rollback_condition",
        "what_to_monitor",
        "conservative_view",
        "guardrails_triggered",
        "risk_level",
    }
    assert required.issubset(d.keys())
