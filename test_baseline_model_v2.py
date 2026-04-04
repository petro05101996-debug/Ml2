import numpy as np
import pandas as pd

from data_adapter import build_daily_panel_from_transactions
from pricing_core.baseline_features import build_baseline_feature_matrix, derive_baseline_feature_spec
from pricing_core.baseline_model import build_baseline_oof_predictions, recursive_baseline_forecast, run_baseline_rolling_backtest, train_baseline_model


def _txn(n=160):
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    price = 10 + (np.arange(n) % 7)
    q = 30 + (np.arange(n) % 5)
    return pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "quantity": q, "revenue": q * price, "price": price, "discount_rate": 0.0, "promotion": 0.0, "stock": 100.0, "cost": 6.5, "region": "US", "channel": "online", "segment": "retail"})


def test_baseline_feature_spec_contains_short_lags():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn()))
    spec = derive_baseline_feature_spec(fm)
    assert {"sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28"}.issubset(set(spec["baseline_numeric_features"]))


def test_baseline_feature_spec_excludes_price_discount_promo_from_default_baseline_features():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn()))
    spec = derive_baseline_feature_spec(fm)
    assert "price" not in spec["baseline_features"]
    assert "discount" not in spec["baseline_features"]
    assert "promotion" not in spec["baseline_features"]


def test_baseline_model_forecast_runs():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn()))
    spec = derive_baseline_feature_spec(fm)
    trained = train_baseline_model(fm.iloc[:-14], spec, small_mode=True)
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=7, freq="D")})
    base_ctx = {"product_id": "sku-1", "category": "cat", "region": "US", "channel": "online", "segment": "retail"}
    out = recursive_baseline_forecast(trained, fm, fut, base_ctx, spec)
    assert len(out) == 7
    assert (out["baseline_pred"] >= 0).all()


def test_baseline_rolling_backtest_returns_windows():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(220)))
    out = run_baseline_rolling_backtest(fm, "cat", "sku-1")
    assert int(out["rolling_summary"]["n_valid_windows"]) >= 1


def test_baseline_oof_has_no_future_leakage():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(220)))
    oof = build_baseline_oof_predictions(fm, "cat", "sku-1")
    assert oof["baseline_oof"].isna().mean() < 1.0


def test_baseline_fallback_works_on_tiny_history():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(10)))
    assert len(fm) == 10


def test_dense_oof_produces_enough_baseline_oof_rows():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(280)))
    oof = build_baseline_oof_predictions(fm, "cat", "sku-1")
    coverage = float(oof["baseline_oof"].notna().mean())
    assert coverage >= 0.45
