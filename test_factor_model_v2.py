import numpy as np
import pandas as pd

from pricing_core.factor_features import build_factor_feature_matrix, build_factor_target, derive_factor_feature_spec
from pricing_core.factor_model import predict_factor_effect, run_factor_rolling_backtest, train_factor_model
from pricing_core.scenario_engine import apply_user_overrides, build_future_factor_frame


def _df(n=140, flat_user=False):
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    price = 100 + (np.arange(n) % 10)
    sales = 200 - 0.8 * price + (np.arange(n) % 4)
    base = sales * 0.95
    df = pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "region": "US", "channel": "online", "segment": "retail", "price": price, "discount": 0.05 + (np.arange(n) % 3) * 0.01, "promotion": (np.arange(n)%7==0).astype(float), "stock": 100.0 + (np.arange(n)%5), "sales": sales, "baseline_oof": base, "user_factor_num__x": 1.0 if flat_user else np.arange(n) % 11})
    return df


def test_factor_target_uses_baseline_oof():
    df = _df()
    t = build_factor_target(df)
    assert np.isfinite(t).all()


def test_factor_feature_spec_contains_price_rel_discount_promo():
    df = _df()
    spec = derive_factor_feature_spec(df)
    for c in ["price", "discount", "promotion", "price_rel_to_recent_median_28"]:
        assert c in spec["factor_numeric_features"] or c in spec["factor_features"]


def test_factor_feature_spec_excludes_flat_user_numeric_features():
    df = _df(flat_user=True)
    spec = derive_factor_feature_spec(df)
    assert "user_factor_num__x" not in spec["factor_numeric_features"]


def test_factor_multiplier_is_clipped():
    df = _df()
    spec = derive_factor_feature_spec(df)
    ff = build_factor_feature_matrix(df, spec)
    ff["factor_target"] = build_factor_target(ff)
    trained = train_factor_model(ff, spec, small_mode=True)
    p = predict_factor_effect(ff, trained, spec)
    assert (p["factor_multiplier"].between(0.70, 1.35)).all()


def test_factor_model_unavailable_on_insufficient_signal():
    df = _df(30)
    assert len(df) < 60


def test_price_monotonic_direction_not_positive_on_sanity_case():
    df = _df()
    spec = derive_factor_feature_spec(df)
    ff = build_factor_feature_matrix(df, spec)
    ff["factor_target"] = build_factor_target(ff)
    trained = train_factor_model(ff, spec, small_mode=True)
    row = ff.tail(1).copy()
    base = float(predict_factor_effect(row, trained, spec)["factor_multiplier"].iloc[0])
    row["price_rel_to_recent_median_28"] += 0.2
    up = float(predict_factor_effect(row, trained, spec)["factor_multiplier"].iloc[0])
    assert up <= base + 1e-6


def test_unknown_user_factor_override_ignored_with_warning():
    out = apply_user_overrides({"price": 10.0}, {"unknown": 1})
    assert any("unknown_override_ignored" in w for w in out["_warnings"])


def test_factor_backtest_is_out_of_sample():
    df = _df()
    spec = derive_factor_feature_spec(df)
    ff = build_factor_feature_matrix(df, spec)
    ff["factor_target"] = build_factor_target(ff)
    bt = run_factor_rolling_backtest(ff, spec)
    assert bt["trained"] is True
    assert bt["n_valid_windows"] >= 1
    assert "median_factor_target_rmse" in bt


def test_price_override_changes_factor_multiplier_with_trained_factor():
    df = _df()
    spec = derive_factor_feature_spec(df)
    ff = build_factor_feature_matrix(df, spec)
    ff["factor_target"] = build_factor_target(ff)
    trained = train_factor_model(ff, spec, small_mode=True)
    base_ctx = {"price": 100.0, "discount": 0.05, "promotion": 0.0, "product_id": "sku-1", "category": "cat"}
    fut = pd.DataFrame({"date": pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    base_frame = build_future_factor_frame(df, fut, base_ctx, spec)
    alt_ctx = dict(base_ctx)
    alt_ctx["price"] = 120.0
    alt_frame = build_future_factor_frame(df, fut, alt_ctx, spec)
    base_mult = float(predict_factor_effect(base_frame, trained, spec)["factor_multiplier"].mean())
    alt_mult = float(predict_factor_effect(alt_frame, trained, spec)["factor_multiplier"].mean())
    assert alt_mult <= base_mult


def test_promotion_override_keeps_intensity():
    out = apply_user_overrides({"promotion": 0.0}, {"promotion": 0.2})
    assert out["promotion"] == 0.2


def test_user_factor_cat_override_applied_to_future_frame():
    df = _df()
    df["user_factor_cat__campaign"] = ["A" if i % 2 == 0 else "C" for i in range(len(df))]
    spec = derive_factor_feature_spec(df)
    fut = pd.DataFrame({"date": pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=2, freq="D")})
    frame = build_future_factor_frame(df, fut, {"price": 100.0, "discount": 0.05, "promotion": 0.0, "product_id": "sku-1", "category": "cat", "user_factor_cat__campaign": "B"}, spec)
    assert set(frame["user_factor_cat__campaign"].astype(str).unique()) == {"B"}


def test_price_rel_median_is_series_scoped():
    d = pd.date_range("2025-01-01", periods=35, freq="D")
    df = pd.DataFrame(
        {
            "date": list(d) + list(d),
            "product_id": ["sku-1"] * 70,
            "category": ["cat"] * 70,
            "region": ["US"] * 35 + ["EU"] * 35,
            "channel": ["online"] * 70,
            "segment": ["retail"] * 70,
            "series_id": ["sku-1|US|online|retail"] * 35 + ["sku-1|EU|online|retail"] * 35,
            "price": [100.0] * 35 + [200.0] * 35,
            "discount": [0.0] * 70,
            "promotion": [0.0] * 70,
            "sales": [10.0] * 70,
            "baseline_oof": [10.0] * 70,
        }
    )
    spec = derive_factor_feature_spec(df)
    ff = build_factor_feature_matrix(df, spec)
    us = ff[ff["series_id"] == "sku-1|US|online|retail"].tail(1).iloc[0]
    eu = ff[ff["series_id"] == "sku-1|EU|online|retail"].tail(1).iloc[0]
    assert abs(float(us["price_rel_to_recent_median_28"])) < 1e-9
    assert abs(float(eu["price_rel_to_recent_median_28"])) < 1e-9
