import numpy as np
import pandas as pd

from scenario_engine import run_scenario
from scenario_engine_enhanced import run_enhanced_scenario


def _baseline(days: int = 7, units: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=days, freq="D"),
        "baseline_units": np.full(days, units, dtype=float),
    })


def test_no_change_keeps_baseline_units():
    out = run_scenario(
        baseline_output=_baseline(),
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_flag": 0.0,
            "promo_share": 0.0,
            "freight_ref": 5.0,
            "freight_scenario": 5.0,
            "stockout_share": 0.0,
            "scenario_net_price": 100.0,
            "unit_cost": 65.0,
            "freight_value": 5.0,
        },
    )
    assert np.allclose(out["final_units"], out["baseline_units"])


def test_price_monotonicity():
    base = _baseline()
    common = {
        "baseline_price_ref": 100.0,
        "promo_flag": 0.0,
        "promo_share": 0.0,
        "freight_ref": 5.0,
        "freight_scenario": 5.0,
        "stockout_share": 0.0,
        "scenario_net_price": 100.0,
        "unit_cost": 65.0,
        "freight_value": 5.0,
    }
    down = run_scenario(base, {**common, "scenario_price": 90.0})
    up = run_scenario(base, {**common, "scenario_price": 110.0})
    assert down["final_units"].sum() > up["final_units"].sum()


def test_promo_decrease_lowers_units_and_promo_increase_raises_units():
    base = _baseline(days=3, units=100.0)
    common = {
        "baseline_price_ref": 100.0,
        "scenario_price": 100.0,
        "freight_ref": 0.0,
        "freight_scenario": 0.0,
        "scenario_net_price": 100.0,
        "unit_cost": 65.0,
        "freight_value": 0.0,
    }
    promo_down = run_scenario(
        base,
        {
            **common,
            "promo_flag_baseline": 1.0,
            "promo_flag_scenario": 0.0,
            "promo_intensity_baseline": 1.0,
            "promo_intensity_scenario": 0.0,
        },
    )
    promo_up = run_scenario(
        base,
        {
            **common,
            "promo_flag_baseline": 0.0,
            "promo_flag_scenario": 1.0,
            "promo_intensity_baseline": 0.0,
            "promo_intensity_scenario": 1.0,
        },
    )
    baseline_total = float(np.sum(base["baseline_units"]))
    assert float(np.sum(promo_down["final_units"])) < baseline_total
    assert float(np.sum(promo_up["final_units"])) > baseline_total


def test_shock_multiplier_and_units_applied():
    out = run_scenario(
        baseline_output=_baseline(days=3, units=10.0),
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_flag": 0.0,
            "promo_share": 0.0,
            "freight_ref": 5.0,
            "freight_scenario": 5.0,
            "stockout_share": 0.0,
            "scenario_net_price": 100.0,
            "unit_cost": 65.0,
            "freight_value": 5.0,
        },
        shocks=[
            {"shock_name": "pct", "shock_type": "percent", "shock_value": 0.10, "start_date": "2025-01-01", "end_date": "2025-01-02"},
            {"shock_name": "units", "shock_type": "units", "shock_value": 2.0, "start_date": "2025-01-02", "end_date": "2025-01-03"},
        ],
    )
    # day1: 10*1.1=11 ; day2: 10*1.1+2=13 ; day3: 10+2=12
    assert np.allclose(out["final_units"], np.array([11.0, 13.0, 12.0]))


def test_negative_percent_shock_is_bounded():
    out = run_scenario(
        baseline_output=_baseline(days=1, units=10.0),
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_baseline": 0.0,
            "promo_scenario": 0.0,
            "freight_ref": 5.0,
            "freight_scenario": 5.0,
            "scenario_net_price": 100.0,
            "unit_cost": 65.0,
            "freight_value": 5.0,
        },
        shocks=[
            {"shock_name": "too_low", "shock_type": "percent", "shock_value": -1.5, "start_date": "2025-01-01", "end_date": "2025-01-01"},
        ],
    )
    assert np.allclose(out["final_units"], np.array([2.0]))
    assert any("clipped to bounds" in w for w in out["warnings"])


def test_units_shock_total_over_window_semantics_explicit():
    out = run_scenario(
        baseline_output=_baseline(days=4, units=10.0),
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 100.0,
            "promo_baseline": 0.0,
            "promo_scenario": 0.0,
            "freight_ref": 5.0,
            "freight_scenario": 5.0,
            "scenario_net_price": 100.0,
            "unit_cost": 65.0,
            "freight_value": 5.0,
        },
        shocks=[
            {
                "shock_name": "window_total",
                "shock_type": "units",
                "units_mode": "total_over_window",
                "shock_value": 20.0,
                "start_date": "2025-01-01",
                "end_date": "2025-01-04",
            }
        ],
    )
    # +20 total across 4 days => +5/day
    assert np.allclose(out["final_units"], np.array([15.0, 15.0, 15.0, 15.0]))


def test_baseline_economics_use_baseline_prices_not_scenario_prices():
    out = run_scenario(
        baseline_output=_baseline(days=1, units=10.0),
        scenario_inputs={
            "baseline_price_ref": 100.0,
            "scenario_price": 120.0,
            "baseline_net_price": 100.0,
            "scenario_net_price": 120.0,
            "baseline_unit_cost": 60.0,
            "unit_cost": 70.0,
            "baseline_freight_value": 5.0,
            "freight_value": 8.0,
            "promo_baseline": 0.0,
            "promo_scenario": 0.0,
            "freight_ref": 5.0,
            "freight_scenario": 5.0,
        },
    )
    assert np.allclose(out["baseline_revenue"], np.array([1000.0]))
    assert np.allclose(out["baseline_margin"], np.array([350.0]))


def _enhanced_baseline(days: int = 5, units: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "base_pred_sales": np.full(days, units),
            "stock": np.full(days, 200.0),
            "price": np.full(days, 100.0),
            "discount": np.full(days, 0.1),
            "promotion": np.zeros(days),
            "cost": np.full(days, 60.0),
            "freight_value": np.full(days, 5.0),
            "net_unit_price": np.full(days, 90.0),
        }
    )


def _run_enhanced(price: float, promo: float, freight: float, demand_mult: float = 1.0, stock_cap: float = 0.0):
    baseline = _enhanced_baseline()
    shocks = []
    if demand_mult != 1.0:
        shocks = [{"shock_name": "dm", "shock_type": "percent", "shock_value": demand_mult - 1.0, "start_date": "2025-01-01", "end_date": "2025-01-05"}]
    return run_enhanced_scenario(
        baseline_daily=baseline,
        current_ctx={"price": 100.0, "discount": 0.1, "promotion": 0.0, "cost": 60.0, "freight_value": 5.0},
        future_dates=pd.DataFrame({"date": baseline["date"]}),
        scenario_overrides={"promotion": promo, "stock_cap": stock_cap},
        pooled_elasticity=-1.1,
        small_mode_info={},
        requested_price=price,
        model_price=price,
        baseline_discount=0.1,
        scenario_discount=0.1,
        baseline_cost=60.0,
        scenario_cost=60.0,
        baseline_freight=5.0,
        scenario_freight=freight,
        shocks=shocks,
    )


def test_enhanced_price_monotonicity():
    low = _run_enhanced(price=95.0, promo=0.0, freight=5.0)
    high = _run_enhanced(price=110.0, promo=0.0, freight=5.0)
    assert float(np.sum(low["final_units"])) >= float(np.sum(high["final_units"]))


def test_enhanced_promo_monotonicity():
    low = _run_enhanced(price=100.0, promo=0.0, freight=5.0)
    high = _run_enhanced(price=100.0, promo=0.7, freight=5.0)
    assert float(np.sum(high["final_units"])) >= float(np.sum(low["final_units"]))


def test_enhanced_freight_monotonicity():
    low = _run_enhanced(price=100.0, promo=0.0, freight=5.0)
    high = _run_enhanced(price=100.0, promo=0.0, freight=9.0)
    assert float(np.sum(high["final_units"])) <= float(np.sum(low["final_units"]))


def test_enhanced_shock_and_stock_constraint():
    shocked = _run_enhanced(price=100.0, promo=0.0, freight=5.0, demand_mult=1.2, stock_cap=50.0)
    baseline = _run_enhanced(price=100.0, promo=0.0, freight=5.0, demand_mult=1.0, stock_cap=0.0)
    assert float(np.sum(shocked["unconstrained_units"])) >= float(np.sum(baseline["unconstrained_units"]))
    assert np.all(shocked["final_units"] <= shocked["scenario_profile"]["available_stock"].to_numpy(dtype=float) + 1e-9)
    assert np.all(shocked["lost_sales"] >= -1e-9)
