def test_data_quality_gate_blocks_hard_blockers():
    from production_contract import resolve_data_quality_gate

    q = {
        "errors": [],
        "warnings": [],
        "blockers": ["quantity_missing_not_inferable"],
        "cost_is_proxy": False,
        "cost_source": "provided",
    }

    gate = resolve_data_quality_gate(q)

    assert gate["status"] == "diagnostic_only"
    assert gate["recommendation_status"] == "not_recommended"
    assert gate["usage_policy"]["can_run_calculation"] is False
    assert gate["usage_policy"]["can_recommend_action"] is False


def test_data_quality_gate_cost_proxy_is_test_only():
    from production_contract import resolve_data_quality_gate

    q = {
        "errors": [],
        "warnings": [],
        "blockers": [],
        "cost_is_proxy": True,
        "cost_source": "proxy_price_65",
    }

    gate = resolve_data_quality_gate(q)

    assert gate["status"] == "test_only"
    assert gate["usage_policy"]["can_run_calculation"] is True
    assert gate["usage_policy"]["can_recommend_action"] is False


def test_recommendation_gate_blocks_data_quality_blockers():
    from recommendation_gate import resolve_recommendation_gate

    gate = resolve_recommendation_gate(
        data_quality={"blockers": ["quantity_missing_not_inferable"]},
        decision_reliability={
            "status_namespace": "decision",
            "base_status": "recommended",
        },
    )

    assert gate["decision_status"] == "not_recommended"
    assert gate["severity"] == 3
    assert any("Data quality blocker" in b for b in gate["blockers"])


def test_data_quality_stats_include_expected_keys():
    import pandas as pd
    from data_adapter import run_data_quality_checks

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20),
        "product_id": ["sku1"] * 20,
        "price": [100, 100, 101, 101, 102] * 4,
        "quantity": [1] * 20,
        "promotion": [0, 1] * 10,
    })

    q = run_data_quality_checks(df)
    stats = q["stats"]

    assert "history_days" in stats
    assert "price_changes_count" in stats
    assert "unique_price_count" in stats
    assert "promo_share" in stats


def test_decision_candidate_does_not_deepcopy_uncloneable_model(monkeypatch):
    from decision_candidate_engine import run_decision_candidate

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy forbidden")

    trained_bundle = {
        "model": Uncopyable(),
        "base_ctx": {"price": 100},
        "latest_row": {},
    }

    candidate = {
        "target_value": 100,
        "scenario_params": {"manual_price": 100},
    }

    def runner(bundle, **kwargs):
        assert isinstance(bundle["model"], Uncopyable)
        return {"demand_total": 10, "revenue_total": 1000, "profit_total": 100}

    result = run_decision_candidate(
        trained_bundle,
        candidate,
        runner,
        scenario_calc_mode="enhanced_local_factors",
        price_guardrail_mode="safe_clip",
        horizon_days=30,
    )

    assert result["demand_total"] == 10
