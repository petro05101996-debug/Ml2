import pandas as pd
import json

from data_adapter import build_auto_mapping, normalize_transactions, build_daily_from_transactions
from what_if import build_sensitivity_grid, run_scenario_set
from app import run_full_pricing_analysis_universal, run_what_if_projection


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


def test_sensitivity_uses_discount_axis():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        discount_mult = kwargs.get("overrides", {}).get("discount_multiplier", 1.0)
        return {"profit_total": p * (2 - discount_mult)}

    out = build_sensitivity_grid({}, base_price=100.0, runner=runner, price_steps=3, discount_steps=3)
    assert "discount_multiplier" in out.columns
    assert "risk_zone" in out.columns


def _make_txn(n_days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 100 + (i % 8)
        promo = 1.0 if i % 14 == 0 else 0.0
        freight = 5.0 + (i % 3)
        qty = max(1.0, 40 - 0.12 * price + 6.0 * promo - 0.3 * freight)
        rows.append(
            {
                "date": d,
                "product_id": "sku-1",
                "category": "cat-a",
                "price": price,
                "quantity": qty,
                "revenue": qty * price * (1.0 - 0.05 * promo),
                "cost": price * 0.65,
                "discount": 0.05 * promo,
                "freight_value": freight,
                "promotion": promo,
                "rating": 4.4,
                "reviews_count": 10 + (i % 5),
            }
        )
    return pd.DataFrame(rows)


def _analyze():
    run_full_pricing_analysis_universal.clear()
    return run_full_pricing_analysis_universal(_make_txn(), "cat-a", "sku-1")


def test_scenario_not_equal_as_is_when_price_changes():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.1)
    as_is_total = float(res["as_is_forecast"]["actual_sales"].sum())
    assert scenario["demand_total"] != as_is_total


def test_price_increase_reduces_demand_monotone():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.10)
    assert high["demand_total"] <= low["demand_total"]


def test_promotion_on_not_decrease_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    off = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 0.0})
    on = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 1.0})
    assert on["demand_total"] >= off["demand_total"]


def test_freight_increase_not_increase_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price, freight_multiplier=1.0)
    high = run_what_if_projection(bundle, manual_price=base_price, freight_multiplier=1.5)
    assert high["demand_total"] <= low["demand_total"]


def test_run_summary_scenario_totals_are_real():
    res = _analyze()
    summary_blob_before = res["run_summary_json"]
    summary = json.loads(res["run_summary_json"].decode("utf-8"))
    out = summary["scenario_output_summary"]
    assert out["scenario_status"] == "not_run"
    assert pd.isna(out["scenario_demand_total"])
    assert pd.isna(out["scenario_revenue_total"])
    assert pd.isna(out["scenario_profit_total"])
    assert res["scenario_forecast"] is None
    assert res["scenario_price"] is None
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.15)
    assert scenario["demand_total"] != float(res["as_is_forecast"]["actual_sales"].sum())
    assert scenario["revenue_total"] != float(res["as_is_forecast"]["revenue"].sum())
    assert scenario["profit_total"] != float(res["as_is_forecast"]["profit"].sum())
    # run_summary_json is analysis-level artifact and should not mutate after runtime what-if calls.
    assert res["run_summary_json"] == summary_blob_before
    summary_after = json.loads(summary_blob_before.decode("utf-8"))
    assert summary_after["scenario_output_summary"]["scenario_status"] == "not_run"


def test_holdout_final_has_decomposition():
    res = _analyze()
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    assert {"pred_baseline", "pred_price_effect_component", "pred_uplift_component", "pred_final"}.issubset(set(holdout.columns))
