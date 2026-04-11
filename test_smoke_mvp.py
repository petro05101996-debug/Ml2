import pandas as pd

from data_adapter import build_auto_mapping, normalize_transactions, build_daily_from_transactions
from what_if import run_scenario_set


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


def test_scenario_runner_baseline_fallback():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        d = kwargs.get("demand_multiplier", 1.0)
        return {"demand_total": 10 * d, "revenue_total": 10 * d * p, "profit_total": 2 * d * p, "confidence": 0.7}

    scenarios = [
        {"name": "Scenario A", "price": 100, "freight_multiplier": 1, "demand_multiplier": 1, "horizon_days": 30},
        {"name": "Scenario B", "price": 90, "freight_multiplier": 1, "demand_multiplier": 1.1, "horizon_days": 30},
    ]
    out = run_scenario_set({}, scenarios, runner)
    assert len(out) == 2
    assert "delta_profit" in out.columns
