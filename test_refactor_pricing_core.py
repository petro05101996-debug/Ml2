import importlib

import pandas as pd

from pricing_core import assess_data_quality, generate_explanation
import pricing_core.core as core


def test_app_module_importable():
    mod = importlib.import_module("app")
    assert mod is not None


def test_pricing_core_computational_helpers_return_expected_types():
    dq = assess_data_quality(history_days=45, n_points=50, missing_share=0.05, holdout_wape=20.0)
    assert isinstance(dq, dict)
    assert "level" in dq

    explanation = generate_explanation(
        {
            "current_price": 100.0,
            "recommended_price": 105.0,
            "current_profit": 1000.0,
            "best_profit": 1100.0,
            "forecast_current": pd.DataFrame({"pred_sales": [10.0]}),
            "forecast_optimal": pd.DataFrame({"pred_sales": [9.8]}),
            "elasticity_map": {"2025-01": -1.1},
        },
        data_quality=dq,
    )
    assert isinstance(explanation, dict)
    assert isinstance(explanation.get("pros", []), list)


def test_what_if_projection_preserves_result_shape(monkeypatch):
    def fake_simulate(*args, **kwargs):
        dates = args[2]["date"]
        daily = pd.DataFrame(
            {
                "date": dates,
                "price": [100.0] * len(dates),
                "pred_sales": [10.0] * len(dates),
                "cost": [65.0] * len(dates),
                "discount": [0.0] * len(dates),
            }
        )
        return {"daily": daily, "sanity_warnings": []}

    monkeypatch.setattr(core, "simulate_horizon_profit", fake_simulate)

    bundle = {
        "feature_spec": {},
        "daily_base": pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, freq="D"), "sales": [1.0, 1.0, 1.0]}),
        "base_ctx": {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 0.0, "review_score": 4.5, "reviews_count": 0.0},
        "latest_row": {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 0.0, "review_score": 4.5, "reviews_count": 0.0},
        "future_dates": pd.DataFrame({"date": pd.date_range("2025-01-04", periods=2, freq="D")}),
        "direct_models": [],
        "baseline_models": [],
        "elasticity_map": {},
        "pooled_elasticity": -1.1,
        "w_direct": 0.5,
        "model_version": "v1",
        "confidence": 0.6,
    }

    out = core.run_what_if_projection(bundle, manual_price=100.0, include_sensitivity=False)
    assert isinstance(out, dict)
    assert isinstance(out.get("daily"), pd.DataFrame)
    for key in ["profit_total", "revenue_total", "confidence"]:
        assert key in out


def test_what_if_multipliers_materially_change_outputs(monkeypatch):
    def fake_simulate(*args, **kwargs):
        dates = args[2]["date"]
        model_price = float(args[1])
        base_ctx = args[6]
        demand = 10.0 * (100.0 / max(model_price, 1e-9))
        daily = pd.DataFrame(
            {
                "date": dates,
                "price": [model_price] * len(dates),
                "pred_sales": [demand] * len(dates),
                "cost": [float(base_ctx.get("cost", 65.0))] * len(dates),
                "discount": [float(base_ctx.get("discount", 0.0))] * len(dates),
            }
        )
        return {"daily": daily, "sanity_warnings": []}

    monkeypatch.setattr(core, "simulate_horizon_profit", fake_simulate)

    bundle = {
        "feature_spec": {},
        "daily_base": pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, freq="D"), "sales": [1.0, 1.0, 1.0]}),
        "base_ctx": {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 0.0, "review_score": 4.5, "reviews_count": 0.0},
        "latest_row": {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 0.0, "review_score": 4.5, "reviews_count": 0.0},
        "future_dates": pd.DataFrame({"date": pd.date_range("2025-01-04", periods=7, freq="D")}),
        "direct_models": [],
        "baseline_models": [],
        "elasticity_map": {},
        "pooled_elasticity": -1.1,
        "w_direct": 0.5,
        "model_version": "v1",
        "confidence": 0.6,
    }
    base = core.run_what_if_projection(bundle, manual_price=100.0, include_sensitivity=False)
    changed = core.run_what_if_projection(
        bundle,
        manual_price=105.0,
        demand_multiplier=0.9,
        discount_multiplier=1.2,
        cost_multiplier=1.1,
        include_sensitivity=False,
    )
    assert float(base["profit_total"]) != float(changed["profit_total"])


def test_quantize_sales_units_rounds_to_non_negative_integer():
    assert core._quantize_sales_units(72.4) == 72.0
    assert core._quantize_sales_units(72.6) == 73.0
    assert core._quantize_sales_units(-5) == 0.0
