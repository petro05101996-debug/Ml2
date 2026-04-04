import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from unittest.mock import patch

from data_adapter import build_daily_panel_from_transactions
from pricing_core.model_utils import XGBRegressor
from pricing_core.v1_features import build_v1_panel_feature_matrix, derive_v1_feature_spec
from pricing_core.v1_forecast import train_v1_demand_model
from pricing_core.v1_orchestrator import run_full_pricing_analysis_universal_v1
from pricing_core.v1_scenario import run_v1_what_if_projection


class DummyMainModel:
    def predict(self, X):
        x = pd.DataFrame(X)
        return pd.to_numeric(x.get("sales_lag1", 0.0), errors="coerce").fillna(0.0).values


class DummyFallback:
    def predict(self, X):
        x = pd.DataFrame(X)
        return (pd.to_numeric(x.get("sales_lag1", 0.0), errors="coerce").fillna(0.0) * 0.9).values


def _txn(n=120, low_price_var=False):
    days = pd.date_range("2025-01-01", periods=n, freq="D")
    price = [100.0 + ((i % 2) * 0.5 if low_price_var else (i % 7)) for i in range(n)]
    return pd.DataFrame(
        {
            "date": days,
            "product_id": ["sku-1"] * n,
            "category": ["cat"] * n,
            "quantity": [20 + (i % 3) for i in range(n)],
            "revenue": [(20 + (i % 3)) * price[i] for i in range(n)],
            "price": price,
            "cost": [65.0] * n,
            "discount_rate": [0.0] * n,
            "freight_value": [2.0] * n,
            "stock": [100.0] * n,
            "promotion": [0.0] * n,
            "review_score": [4.5] * n,
            "reviews_count": [100.0] * n,
            "region": ["US"] * n,
            "channel": ["online"] * n,
            "segment": ["retail"] * n,
            "user_factor_num__external_temp": [10 + (i % 10) for i in range(n)],
        }
    )


def test_xgboost_main_model_and_objective():
    fm = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(_txn(100)))
    spec = derive_v1_feature_spec(fm.iloc[:80].copy())
    trained = train_v1_demand_model(fm.iloc[:80].copy(), spec, small_mode=True)
    assert isinstance(trained["main_models"][0], XGBRegressor)
    assert trained["main_models"][0].get_params()["objective"] == "count:poisson"
    assert trained["main_models"][0].get_params()["eval_metric"] == "poisson-nloglik"


def test_fallback_is_poisson_regressor():
    fm = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(_txn(100)))
    spec = derive_v1_feature_spec(fm.iloc[:80].copy())
    trained = train_v1_demand_model(fm.iloc[:80].copy(), spec, small_mode=True)
    assert isinstance(trained["fallback_model"], PoissonRegressor)


def test_no_fake_sales_fill_for_missing_days():
    tx = _txn(10)
    tx = tx[tx["date"] != tx["date"].min() + pd.Timedelta(days=3)].copy()
    panel = build_daily_panel_from_transactions(tx)
    gap_day = tx["date"].min() + pd.Timedelta(days=3)
    row = panel[panel["date"] == gap_day].iloc[0]
    assert float(row["sales"]) == 0.0


def test_no_cross_sku_rolling_leakage():
    base = _txn(35)
    b2 = base.copy()
    b2["product_id"] = "sku-2"
    b2["quantity"] = 1000.0
    joined = pd.concat([base, b2], ignore_index=True)
    fm = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(joined))
    sku1 = fm[fm["product_id"] == "sku-1"].sort_values("date").reset_index(drop=True)
    assert float(sku1.iloc[1]["sales_ma7"]) < 100.0


def test_train_only_feature_spec_not_from_whole_panel():
    tx = _txn(120)
    tx["user_factor_num__holdout_only"] = 1.0
    tx.loc[tx.index >= 96, "user_factor_num__holdout_only"] = np.linspace(1.0, 4.0, len(tx.index[96:]))
    result = run_full_pricing_analysis_universal_v1(tx, "cat", "sku-1", horizon_days=7)
    used = result["_trained_bundle"]["feature_spec"]["numeric_demand_features"]
    assert "user_factor_num__holdout_only" not in used


def test_weak_price_signal_gating_and_what_if_available():
    result = run_full_pricing_analysis_universal_v1(_txn(120, low_price_var=True), "cat", "sku-1", horizon_days=7)
    holdout = result["holdout_metrics"].iloc[0].to_dict()
    assert holdout["forecast_mode"] in {"weak_signal", "insufficient_data"}
    assert not bool(holdout["can_recommend_price"])
    w = run_v1_what_if_projection(result["_trained_bundle"], manual_price=float(result["current_price"]))
    assert float(w["actual_sales_total"]) >= 0.0


def test_disagreement_gating_blocks_recommendation():
    tx = _txn(120)
    with patch("pricing_core.v1_orchestrator._run_model_price_effect_sign_test", return_value={"main_sign": 1, "fallback_sign": -1, "agree": False}):
        result = run_full_pricing_analysis_universal_v1(tx, "cat", "sku-1", horizon_days=7)
    assert "main_fallback_disagreement" in result["_trained_bundle"]["recommendation_reasons"]
    assert not bool(result["holdout_metrics"].iloc[0]["can_recommend_price"])
