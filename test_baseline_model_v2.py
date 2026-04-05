import numpy as np
import pandas as pd

from data_adapter import build_daily_panel_from_transactions
from pricing_core.baseline_features import build_baseline_feature_matrix, derive_baseline_feature_spec
from pricing_core.baseline_model import (
    aggregate_daily_to_weekly,
    build_baseline_oof_predictions,
    build_weekday_profile,
    disaggregate_weekly_to_daily,
    recursive_baseline_forecast,
    run_weekly_baseline_rolling_backtest,
    select_best_baseline_plan,
    run_baseline_rolling_backtest,
    select_best_baseline_strategy,
    train_baseline_model,
    week_start,
)


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


def test_selected_baseline_is_not_worse_than_xgb():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(260)))
    selected = select_best_baseline_strategy(fm, "cat", "sku-1")
    xgb = run_baseline_rolling_backtest(fm, "cat", "sku-1", strategy="xgb_recursive")
    summary = selected["strategy_summary"].copy()
    picked = summary[summary["strategy"] == selected["best_strategy"]]
    selected_wape = float(picked["median_wape"].iloc[0]) if len(picked) else np.inf
    xgb_wape = float(xgb["rolling_summary"]["median_wape"])
    assert selected_wape <= xgb_wape


def test_weekly_disaggregation_preserves_totals():
    future_dates = pd.DataFrame({"date": pd.date_range("2025-03-06", periods=10, freq="D")})
    weekly_fc = pd.DataFrame(
        {
            "week_start": [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-10")],
            "baseline_pred_weekly": [70.0, 35.0],
        }
    )
    profile = pd.Series([0.1, 0.2, 0.15, 0.1, 0.15, 0.2, 0.1], index=range(7))
    daily_fc = disaggregate_weekly_to_daily(weekly_fc, future_dates, profile)
    daily_fc["week_start"] = week_start(daily_fc["date"])
    by_week = daily_fc.groupby("week_start", as_index=False)["baseline_pred"].sum()
    expected = pd.DataFrame(
        {
            "week_start": [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-10")],
            "expected_total": [70.0, 35.0],
        }
    )
    merged = by_week.merge(expected, on="week_start", how="inner")
    assert np.allclose(merged["baseline_pred"].values, merged["expected_total"].values)


def test_weekly_plan_selected_only_when_margin_beats_daily():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(260)))
    plan = select_best_baseline_plan(fm, "cat", "sku-1")
    daily = plan["daily_selection"]["strategy_summary"]
    weekly = plan["weekly_selection"]["strategy_summary"]
    daily_wape = float(daily[daily["strategy"] == plan["best_daily_strategy"]]["median_wape"].iloc[0])
    weekly_wape = float(weekly[weekly["strategy"] == plan["best_weekly_strategy"]]["median_wape"].iloc[0])
    if weekly_wape + 2.0 < daily_wape:
        assert plan["granularity"] == "weekly"
    else:
        assert plan["granularity"] == "daily"


def test_weekly_backtest_returns_daily_oof_non_empty():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(260)))
    target = fm[(fm["category"] == "cat") & (fm["product_id"] == "sku-1")].copy()
    weekly = run_weekly_baseline_rolling_backtest(target)
    oof = weekly["oof_daily"]
    assert not oof.empty
    assert {"date", "sales", "baseline_oof"}.issubset(oof.columns)
    assert oof["date"].nunique() > 0
    assert oof["baseline_oof"].notna().sum() > 0


def test_weekly_aggregation_and_profile_contract():
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(_txn(90)))
    target = fm[(fm["category"] == "cat") & (fm["product_id"] == "sku-1")].copy()
    weekly = aggregate_daily_to_weekly(target)
    profile = build_weekday_profile(target)
    assert {"week_start", "sales"}.issubset(weekly.columns)
    assert len(profile) == 7
    assert abs(float(profile.sum()) - 1.0) < 1e-9


def test_baseline_lags_are_isolated_by_series_id():
    d = pd.date_range("2025-01-01", periods=5, freq="D")
    tx = pd.DataFrame(
        {
            "date": list(d) + list(d),
            "product_id": ["sku-1"] * 10,
            "category": ["cat"] * 10,
            "region": ["US"] * 5 + ["EU"] * 5,
            "channel": ["online"] * 10,
            "segment": ["retail"] * 10,
            "quantity": [10, 20, 30, 40, 50, 1, 1, 1, 1, 1],
            "price": [10] * 10,
            "cost": [6] * 10,
            "revenue": [100, 200, 300, 400, 500, 10, 10, 10, 10, 10],
            "discount_rate": [0.0] * 10,
            "promotion": [0.0] * 10,
            "stock": [100.0] * 10,
        }
    )
    fm = build_baseline_feature_matrix(build_daily_panel_from_transactions(tx))
    us = fm[(fm["region"] == "US")].sort_values("date").reset_index(drop=True)
    eu = fm[(fm["region"] == "EU")].sort_values("date").reset_index(drop=True)
    assert float(us.loc[1, "sales_lag1"]) == 10.0
    assert float(eu.loc[1, "sales_lag1"]) == 1.0
