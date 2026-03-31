import pandas as pd
import pytest

from data_adapter import (
    build_auto_mapping,
    build_daily_from_transactions,
    normalize_transactions,
    objective_to_weights,
)
from recommendation import build_business_recommendation
from what_if import build_sensitivity_grid, run_scenario_set
from pricing_core.core import build_feature_matrix, derive_feature_spec


def test_normalize_and_daily_build():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "SKU": ["A", "A", "B"],
            "Unit Price": [100, 110, 90],
            "qty": [1, 2, 1],
        }
    )
    mapping = build_auto_mapping(list(df.columns))
    norm, quality = normalize_transactions(df, mapping)
    assert not quality.get("errors")
    assert "warnings" in quality
    daily = build_daily_from_transactions(norm, "A")
    assert len(daily) >= 2
    assert (daily["price"] > 0).all()


def test_missing_required_columns_returns_error():
    df = pd.DataFrame({"date": ["2025-01-01"], "price": [100]})
    mapping = build_auto_mapping(list(df.columns))
    norm, quality = normalize_transactions(df, mapping)
    assert len(norm) == 1
    assert quality["errors"]


def test_invalid_dates_and_prices_are_dropped():
    df = pd.DataFrame(
        {
            "date": ["bad-date", "2025-01-01", "2025-01-02"],
            "product_id": ["A", "A", "A"],
            "price": [100, "broken", 120],
            "quantity": [1, 1, 1],
        }
    )
    norm, quality = normalize_transactions(df, {"date": "date", "product_id": "product_id", "price": "price", "quantity": "quantity"})
    assert len(norm) == 1
    assert any("невалидной датой" in w for w in quality["warnings"])
    assert any("невалидной ценой" in w for w in quality["warnings"])


def test_normalize_fills_missing_numeric_fields_and_reports_raw_duplicates():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02"],
            "product_id": ["A", "A", "A", "A"],
            "price": [100, 100, 100, 120],
            "quantity": [None, None, 2, None],
            "revenue": [None, None, 200, None],
            "cost": [None, None, 60, None],
        }
    )
    norm, quality = normalize_transactions(df, {"date": "date", "product_id": "product_id", "price": "price", "quantity": "quantity", "revenue": "revenue", "cost": "cost"})
    assert norm["quantity"].isna().sum() == 0
    assert norm["revenue"].isna().sum() == 0
    assert norm["cost"].isna().sum() == 0
    assert quality["raw_stats"]["raw_duplicates"] >= 1


def test_discount_rate_is_canonical_with_legacy_discount_alias():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "product_id": ["A", "A"],
            "price": [100.0, 120.0],
            "discount_amount": [10.0, 0.2],
        }
    )
    mapping = build_auto_mapping(list(df.columns))
    norm, _ = normalize_transactions(df, mapping)
    assert "discount_rate" in norm.columns
    assert "discount" in norm.columns
    assert norm["discount_rate"].between(0.0, 0.95).all()
    assert norm["discount_rate"].equals(norm["discount"])


def test_no_future_leakage_from_backward_fill():
    txn = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "product_id": ["A", "A", "A"],
            "price": [None, 20.0, 40.0],
            "quantity": [1, 1, 1],
            "revenue": [0, 20, 40],
            "cost": [10, 12, 24],
            "category": ["c", "c", "c"],
        }
    )
    daily = build_daily_from_transactions(txn, "A")
    # Первый день не должен заполняться будущим значением 20.0
    assert float(daily.iloc[0]["price"]) != 20.0


def test_objective_presets_are_distinct():
    rev = objective_to_weights("maximize_revenue")
    vol = objective_to_weights("protect_volume")
    margin = objective_to_weights("balanced_mode")
    assert rev != vol
    assert vol != margin
    assert rev != margin


def test_unknown_objective_raises():
    with pytest.raises(ValueError):
        objective_to_weights("custom_objective")


def test_scenario_runner_baseline_fallback_and_deltas():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        d = kwargs.get("demand_multiplier", 1.0)
        c = kwargs.get("cost_multiplier", 1.0)
        f = kwargs.get("freight_multiplier", 1.0)
        demand = 10 * d
        revenue = demand * p
        profit = demand * (p - 20 * c - 2 * f)
        return {"demand_total": demand, "revenue_total": revenue, "profit_total": profit, "confidence": 0.7}

    scenarios = [
        {"name": "Scenario A", "price": 100, "freight_multiplier": 1, "demand_multiplier": 1, "horizon_days": 30, "cost_multiplier": 1},
        {"name": "Scenario B", "price": 90, "freight_multiplier": 1.2, "demand_multiplier": 1.1, "horizon_days": 30, "cost_multiplier": 1.1},
    ]
    out = run_scenario_set({}, scenarios, runner)
    assert len(out) == 2
    assert "delta_profit" in out.columns
    assert out.loc[out["scenario"] == "Scenario B", "delta_profit"].iloc[0] != 0


def test_sensitivity_grid_reacts_to_price_and_demand():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        d = kwargs.get("demand_multiplier", 1.0)
        demand = 100 * d * (100 / max(p, 1e-9))
        profit = (p - 60) * demand
        return {"profit_total": profit}

    grid = build_sensitivity_grid({}, base_price=100, runner=runner, price_steps=4, demand_steps=4)
    assert len(grid) == 16
    assert grid["profit"].nunique() > 1


def test_daily_build_empty_sku_raises():
    txn = pd.DataFrame({"date": ["2025-01-01"], "product_id": ["B"], "price": [10], "quantity": [1], "revenue": [10], "category": ["c"]})
    with pytest.raises(ValueError):
        build_daily_from_transactions(txn, "A")


def test_user_numeric_factors_are_kept_and_used_in_feature_spec():
    txn = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="D"),
            "product_id": ["A"] * 40,
            "category": ["cat"] * 40,
            "price": [100 + (i % 5) for i in range(40)],
            "quantity": [10 + (i % 3) for i in range(40)],
            "revenue": [(100 + (i % 5)) * (10 + (i % 3)) for i in range(40)],
            "external_temp": [20 + (i % 7) for i in range(40)],
            "competitor_index": [1.0 + (i % 4) * 0.1 for i in range(40)],
        }
    )
    daily = build_daily_from_transactions(txn, "A")
    assert "user_factor__external_temp" in daily.columns
    assert "user_factor__competitor_index" in daily.columns
    fm = build_feature_matrix(daily)
    spec = derive_feature_spec(fm)
    assert "user_factor__external_temp" in spec["direct_features"]
    assert "user_factor__competitor_index" in spec["direct_features"]


def test_business_recommendation_has_plain_fields():
    rec = build_business_recommendation(
        current_price=100,
        recommended_price=103,
        current_profit=10000,
        recommended_profit=11200,
        confidence=0.78,
        elasticity=-1.1,
        history_days=120,
    )
    for key in ["plain_action", "plain_reason", "plain_effect", "risk_text", "confidence_text", "conditions_text", "next_steps_text", "risks_text"]:
        assert key in rec
        assert isinstance(rec[key], str)
        assert rec[key]


def test_business_recommendation_low_confidence_has_warning():
    rec = build_business_recommendation(
        current_price=100,
        recommended_price=98,
        current_profit=10000,
        recommended_profit=9800,
        confidence=0.31,
        elasticity=-1.8,
        history_days=30,
    )
    assert rec["warning_text"]


def test_business_recommendation_uses_price_delta_percent_not_cost():
    rec = build_business_recommendation(
        current_price=1261.15,
        recommended_price=1362.795,
        current_profit=100000,
        recommended_profit=108100,
        confidence=0.8,
        elasticity=-1.0,
        history_days=180,
    )
    assert "8.1%" in rec["plain_action"]
    assert "3362" not in rec["plain_action"]


def test_scenario_table_price_column_uses_input_price_not_cost():
    def runner(_bundle, **kwargs):
        price = float(kwargs["manual_price"])
        high_cost = 43229.16
        demand = 10.0
        return {
            "demand_total": demand,
            "actual_sales_total": demand,
            "lost_sales_total": 0.0,
            "revenue_total": price * demand,
            "profit_total": (price - high_cost) * demand,
            "confidence": 0.6,
        }

    scenarios = [
        {"name": "Baseline", "price": 1261.15, "freight_multiplier": 1, "demand_multiplier": 1, "horizon_days": 30, "cost_multiplier": 1},
        {"name": "Scenario A", "price": 1362.795, "freight_multiplier": 1, "demand_multiplier": 1, "horizon_days": 30, "cost_multiplier": 1},
    ]
    out = run_scenario_set({}, scenarios, runner)
    scenario_price = float(out.loc[out["scenario"] == "Scenario A", "price"].iloc[0])
    assert scenario_price == pytest.approx(1362.795)
    assert scenario_price != pytest.approx(43229.16)
