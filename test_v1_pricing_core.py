import numpy as np
import pandas as pd

from pricing_core.v1_features import build_v1_feature_matrix, derive_v1_feature_spec
from pricing_core.v1_forecast import recursive_v1_demand_forecast
from pricing_core.v1_optimizer import recommend_v1_price_horizon, simulate_v1_horizon_profit
from pricing_core.v1_orchestrator import run_v1_recursive_holdout
from pricing_core.v1_scenario import run_v1_what_if_projection


class DummyModel:
    def predict(self, X):
        x = pd.DataFrame(X)
        base = pd.to_numeric(x.get("sales_lag1", 0.0), errors="coerce").fillna(0.0)
        uf = pd.to_numeric(x.get("user_factor__external_temp", 0.0), errors="coerce").fillna(0.0)
        return (0.7 * base + 0.4 * uf).values


def _daily(n=80):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "sales": [20 + (i % 4) for i in range(n)],
        "price": [100 + (i % 5) for i in range(n)],
        "cost": [65 + (i % 5) for i in range(n)],
        "discount": [0.0] * n,
        "promotion": [0.0] * n,
        "stock": [100.0] * n,
        "freight_value": [2.0] * n,
        "review_score": [4.6] * n,
        "reviews_count": [120.0] * n,
        "user_factor__external_temp": [10 + (i % 10) for i in range(n)],
    })


def test_what_if_changes_forecast_by_user_factor():
    fm = build_v1_feature_matrix(_daily())
    spec = derive_v1_feature_spec(fm)
    bundle = {"demand_models": [DummyModel()], "feature_spec": spec, "daily_base": fm, "base_ctx": {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 100.0, "freight_value": 2.0, "review_score": 4.6, "reviews_count": 120.0, "user_factor__external_temp": 10.0}, "latest_row": {"price": 100.0}, "future_dates": pd.DataFrame({"date": pd.date_range("2025-04-01", periods=7, freq="D")}), "confidence": 0.7}
    base = run_v1_what_if_projection(bundle, manual_price=100.0, include_sensitivity=False)
    changed = run_v1_what_if_projection(bundle, manual_price=100.0, scenario={"factors": {"user_factor__external_temp": 30.0}}, include_sensitivity=False)
    assert float(base["actual_sales_total"]) != float(changed["actual_sales_total"])


def test_holdout_is_recursive_path():
    fm = build_v1_feature_matrix(_daily())
    spec = derive_v1_feature_spec(fm)
    train, test = fm.iloc[:60].copy(), fm.iloc[60:65].copy()
    out = run_v1_recursive_holdout(train, test, [DummyModel()], spec)
    assert list(out.columns)[:3] == ["date", "actual_sales", "pred_sales"]
    assert out["pred_sales"].nunique() > 1


def test_optimizer_without_elasticity_fields_and_recommendation_uses_demand_model():
    fm = build_v1_feature_matrix(_daily())
    spec = derive_v1_feature_spec(fm)
    future = pd.DataFrame({"date": pd.date_range("2025-04-01", periods=5, freq="D")})
    sim = simulate_v1_horizon_profit({"price": 100.0}, 102.0, future, [DummyModel()], fm, {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 100.0, "freight_value": 2.0, "review_score": 4.6, "reviews_count": 120.0, "user_factor__external_temp": 10.0}, spec)
    assert {"price_multiplier", "elasticity", "reference_price"}.isdisjoint(set(sim["daily"].columns))
    rec = recommend_v1_price_horizon({"price": 100.0}, [DummyModel()], fm, {"price": 100.0, "cost": 65.0, "discount": 0.0, "promotion": 0.0, "stock": 100.0, "freight_value": 2.0, "review_score": 4.6, "reviews_count": 120.0, "user_factor__external_temp": 10.0}, spec, n_days=5)
    assert "best_price" in rec
