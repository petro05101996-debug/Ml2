import numpy as np
import pandas as pd

from data_adapter import normalize_transactions
from data_adapter import build_daily_from_transactions_scoped
from pricing_core.v1_elasticity import compute_price_multiplier
from pricing_core.v1_features import V1_BASELINE_FEATURES, build_v1_feature_matrix
from pricing_core.v1_orchestrator import run_full_pricing_analysis_universal_v1
from pricing_core.v1_scenario import run_v1_what_if_projection


def _make_txn(days=120, weak_price=False):
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    base_price = 100.0
    if weak_price:
        prices = [base_price + (0.2 if i % 40 == 0 else 0.0) for i in range(days)]
    else:
        prices = [base_price + ((i % 12) - 6) * 1.8 for i in range(days)]
    qty = [max(1.0, 40 - 0.15 * (p - base_price) + (i % 5) * 0.3) for i, p in enumerate(prices)]
    return pd.DataFrame(
        {
            "date": dates,
            "product_id": ["sku-1"] * days,
            "category": ["cat-a"] * days,
            "price": prices,
            "quantity": qty,
            "revenue": np.array(prices) * np.array(qty),
            "cost": np.array(prices) * 0.65,
            "discount_rate": [0.05] * days,
            "freight_value": [2.0] * days,
            "promotion": [0.0] * days,
            "rating": [4.7] * days,
            "reviews_count": [200] * days,
        }
    )


def test_v1_simulation_changes_demand_when_price_changes():
    low = compute_price_multiplier(90.0, 100.0, -1.0)
    mid = compute_price_multiplier(100.0, 100.0, -1.0)
    high = compute_price_multiplier(110.0, 100.0, -1.0)
    assert low > mid > high


def test_v1_no_recommendation_when_unique_prices_too_low():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=90, weak_price=True), "cat-a", "sku-1")
    assert out["best_price"] == out["current_price"]
    assert out["business_recommendation"]["decision_type"] in {"hold", "no_decision"}


def test_daily_build_scoped_by_category_and_sku():
    df = pd.concat(
        [
            _make_txn(days=10),
            _make_txn(days=10).assign(category="cat-b", quantity=999.0, revenue=99900.0),
        ],
        ignore_index=True,
    )
    daily = build_daily_from_transactions_scoped(df, "sku-1", category="cat-a")
    assert float(daily["sales"].max()) < 100.0
    assert daily["category"].iloc[0] == "cat-a"


def test_v1_scenario_rating_alias_updates_review_score():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=100), "cat-a", "sku-1")
    bundle = out["_trained_bundle"]
    res = run_v1_what_if_projection(bundle, manual_price=100.0, scenario={"factors": {"rating": 4.9}}, include_sensitivity=False)
    assert np.isclose(res["daily"]["review_score"].iloc[0], 4.9)


def test_v1_what_if_does_not_modify_history_sales_columns():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=100), "cat-a", "sku-1")
    bundle = out["_trained_bundle"]
    before = bundle["daily_base"]["sales"].copy()
    _ = run_v1_what_if_projection(bundle, manual_price=101.0, demand_multiplier=0.5, include_sensitivity=False)
    after = bundle["daily_base"]["sales"].copy()
    assert before.equals(after)


def test_v1_holdout_and_recommendation_use_same_baseline_model_family():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=120), "cat-a", "sku-1")
    bundle = out["_trained_bundle"]
    assert "baseline_models" in bundle
    assert isinstance(bundle["baseline_models"], list)
    assert len(bundle["baseline_models"]) > 0


def test_v1_feature_spec_contains_only_projection_safe_features():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=120), "cat-a", "sku-1")
    feats = out["_trained_bundle"]["feature_spec"]["baseline_features"]
    assert set(feats).issubset(set(V1_BASELINE_FEATURES))


def test_v1_feature_fill_has_no_global_median_leakage():
    df = _make_txn(days=35)
    features = build_v1_feature_matrix(build_daily_from_transactions_scoped(df, "sku-1", category="cat-a"))
    first_row = features.iloc[0]
    assert first_row["sales_lag7"] == 0.0
    assert first_row["sales_lag28"] == 0.0
    assert first_row["sales_ma28"] == 0.0


def test_v1_business_recommendation_contains_revenue_and_volume_changes():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=120), "cat-a", "sku-1")
    structured = out["business_recommendation"]["structured"]
    assert structured["expected_revenue_change"] is not None
    assert structured["expected_volume_change"] is not None


def test_v1_user_facing_profit_is_raw_and_adjusted_is_separate():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=120), "cat-a", "sku-1", risk_lambda=1.2)
    total_profit = float(out["forecast_current"]["profit"].sum())
    assert np.isclose(out["current_profit"], total_profit)
    assert "current_profit_adjusted" in out
    assert "best_profit_adjusted" in out
    assert out["current_profit_adjusted"] <= out["current_profit"]


def test_v1_business_decision_not_overridden_and_has_legacy_alias():
    out = run_full_pricing_analysis_universal_v1(_make_txn(days=120), "cat-a", "sku-1")
    biz = out["business_recommendation"]
    structured_decision = biz.get("structured", {}).get("decision_type")
    assert structured_decision in {"action", "test", "hold", "no_decision"}
    assert "legacy_decision_type" in biz
    assert biz["legacy_decision_type"] in {"change_price", "pilot_change", "hold", "no_decision"}


def test_normalize_transactions_warns_on_mixed_discount_rate_scale():
    raw = _make_txn(days=4).drop(columns=["discount_rate"]).copy()
    raw.loc[0, "discount"] = 0.1
    raw.loc[1, "discount"] = 15.0
    raw.loc[2, "discount"] = 0.2
    raw.loc[3, "discount"] = 10.0

    mapping = {
        "date": "date",
        "product_id": "product_id",
        "category": "category",
        "price": "price",
        "quantity": "quantity",
        "revenue": "revenue",
        "cost": "cost",
        "discount": "discount",
    }
    _, quality = normalize_transactions(raw, mapping)
    assert any("смешанный формат discount_rate" in w.lower() for w in quality.get("warnings", []))
    assert quality.get("can_recommend") is False
