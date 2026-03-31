import pandas as pd

from calc_engine import compute_daily_unit_economics, sanitize_discount


def test_unit_economics_revenue_and_profit_formula():
    daily = pd.DataFrame({"price": [100.0], "cost": [60.0], "pred_sales": [10.0], "discount": [0.1]})
    out, checks = compute_daily_unit_economics(daily)
    assert checks["sanity_warnings"] == []
    assert float(out.loc[0, "effective_unit_price"]) == 90.0
    assert float(out.loc[0, "total_revenue"]) == 900.0
    assert float(out.loc[0, "total_cost"]) == 600.0
    assert float(out.loc[0, "profit"]) == 300.0


def test_stock_cap_keeps_formula_consistent():
    daily = pd.DataFrame({"price": [120.0], "cost": [70.0], "pred_sales": [25.0], "discount": [0.0]})
    out, _ = compute_daily_unit_economics(daily, stock_cap=8.0)
    assert float(out.loc[0, "pred_quantity"]) == 8.0
    assert float(out.loc[0, "total_revenue"]) == 960.0
    assert float(out.loc[0, "profit"]) == 400.0


def test_discount_sanitized_to_supported_range():
    assert sanitize_discount(-0.5) == 0.0
    assert sanitize_discount(2.0) == 0.95


def test_baseline_equals_scenario_when_inputs_equal():
    daily = pd.DataFrame({"price": [100.0, 100.0], "cost": [60.0, 60.0], "pred_sales": [10.0, 15.0], "discount": [0.05, 0.05]})
    baseline, _ = compute_daily_unit_economics(daily)
    scenario, _ = compute_daily_unit_economics(daily.copy())
    assert baseline["total_revenue"].sum() == scenario["total_revenue"].sum()
    assert baseline["profit"].sum() == scenario["profit"].sum()


def test_discount_raw_out_of_range_emits_warning():
    daily = pd.DataFrame({"price": [100.0], "cost": [60.0], "pred_sales": [10.0], "discount": [10.0]})
    _, checks = compute_daily_unit_economics(daily)
    assert "discount_raw_out_of_range" in checks["sanity_warnings"]


def test_freight_is_included_in_total_cost():
    daily = pd.DataFrame({"price": [100.0], "cost": [60.0], "freight_value": [5.0], "pred_sales": [10.0], "discount": [0.0]})
    out, _ = compute_daily_unit_economics(daily)
    assert float(out.loc[0, "total_cost"]) == 650.0
    assert float(out.loc[0, "profit"]) == 350.0
