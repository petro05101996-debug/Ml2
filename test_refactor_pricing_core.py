import numpy as np
import pandas as pd

from data_adapter import build_daily_panel_from_transactions, normalize_transactions
from pricing_core.model_utils import build_models
from pricing_core.v1_features import (
    build_v1_one_step_features,
    build_v1_panel_feature_matrix,
    derive_v1_feature_spec,
)
from pricing_core.v1_orchestrator import run_full_pricing_analysis_universal_v1, run_v1_recursive_holdout
from pricing_core.v1_scenario import run_v1_what_if_projection


def _dataset(kind: str) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2025-01-01", periods=120 if kind != "sparse" else 28, freq="D")
    skus = ["sku-a", "sku-b", "sku-c"]
    for sku in skus:
        for i, d in enumerate(dates):
            if kind == "strong":
                price = 100 + (i % 10) * 1.5
            elif kind == "weak":
                price = 100 + (i % 2) * 0.2
            else:
                price = 100 + (i % 3)
            temp = 15 + (i % 8)
            campaign = "A" if i % 2 == 0 else "B"
            sales = max(1.0, 120 - 0.6 * price + 0.5 * temp + (5 if sku == "sku-a" else 0))
            if kind == "sparse" and i % 5 == 0:
                sales = np.nan
            rows.append(
                {
                    "date": d,
                    "product_id": sku,
                    "category": "cat-1",
                    "price": price,
                    "quantity": sales,
                    "revenue": sales * price if pd.notna(sales) else np.nan,
                    "cost": 0.65 * price,
                    "discount_rate": 0.05,
                    "freight_value": 2.0,
                    "stock": 100,
                    "promotion": 0,
                    "review_score": 4.3,
                    "reviews_count": 100,
                    "region": "US",
                    "channel": "online",
                    "segment": "retail",
                    "temperature": temp,
                    "campaign_type": campaign,
                }
            )
    return pd.DataFrame(rows)


def _normalized(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c for c in df.columns}
    out, _ = normalize_transactions(df, mapping)
    return out


def test_catboost_build_models_and_cat_features():
    df = _normalized(_dataset("strong"))
    panel = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(df))
    spec = derive_v1_feature_spec(panel)
    train = panel.dropna(subset=["sales"]).head(200)
    models = build_models(
        train[spec["demand_features"]],
        train["sales"],
        spec["demand_features"],
        cat_features=spec["cat_features_demand"],
        loss_function="MAE",
        n_models=1,
    )
    assert models
    assert hasattr(models[0], "get_param")


def test_feature_spec_numeric_and_categorical_user_factors():
    df = _normalized(_dataset("strong"))
    panel = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(df))
    spec = derive_v1_feature_spec(panel)
    assert "user_factor_num__temperature" in spec["demand_features"]
    assert "user_factor_num__temperature" in spec["scenario_features"]
    assert "user_factor_cat__campaign_type" in spec["cat_features_demand"]
    assert "user_factor_cat__campaign_type" not in spec["scenario_features"]


def test_no_duplicate_review_score_user_factor():
    df = _normalized(_dataset("strong"))
    assert "user_factor_num__review_score" not in df.columns


def test_panel_training_and_target_forecast():
    df = _normalized(_dataset("strong"))
    result = run_full_pricing_analysis_universal_v1(df, "cat-1", "sku-a", horizon_days=14)
    spec = result["_trained_bundle"]["feature_spec"]
    assert "product_id" in spec["cat_features_demand"]
    assert len(result["forecast_current"]) == 14


def test_recursive_holdout_uses_predicted_lags():
    df = _normalized(_dataset("strong"))
    panel = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(df))
    spec = derive_v1_feature_spec(panel)
    sku = panel[panel["product_id"] == "sku-a"].copy()
    train, test = sku.iloc[:80], sku.iloc[80:90]

    class Dummy:
        def predict(self, X):
            return pd.to_numeric(X["sales_lag1"], errors="coerce").fillna(0.0).values

    out = run_v1_recursive_holdout(train, test, [Dummy()], spec)
    assert out["pred_sales"].iloc[1] == out["pred_sales"].iloc[0]


def test_weak_price_signal_gating_and_what_if_available():
    df = _normalized(_dataset("weak"))
    result = run_full_pricing_analysis_universal_v1(df, "cat-1", "sku-a", horizon_days=7)
    assert result["_trained_bundle"]["can_recommend_price"] is False
    w = run_v1_what_if_projection(result["_trained_bundle"], manual_price=float(result["current_price"]))
    assert w["actual_sales_total"] >= 0


def test_strong_signal_allows_recommendation():
    df = _normalized(_dataset("strong"))
    result = run_full_pricing_analysis_universal_v1(df, "cat-1", "sku-a", horizon_days=7)
    assert result["_trained_bundle"]["forecast_mode"] in {"strong_signal", "weak_signal"}


def test_ood_scenario_flags_and_confidence_drop():
    df = _normalized(_dataset("strong"))
    result = run_full_pricing_analysis_universal_v1(df, "cat-1", "sku-a", horizon_days=7)
    base = run_v1_what_if_projection(result["_trained_bundle"], manual_price=float(result["current_price"]))
    ood = run_v1_what_if_projection(result["_trained_bundle"], manual_price=float(result["current_price"] * 2.0), scenario={"factors": {"price": float(result["current_price"] * 2.0)}})
    assert ood["ood_flags"]
    assert ood["scenario_confidence"] <= base["scenario_confidence"]


def test_forecast_modes_sparse_dataset():
    df = _normalized(_dataset("sparse"))
    result = run_full_pricing_analysis_universal_v1(df, "cat-1", "sku-a", horizon_days=7)
    assert result["_trained_bundle"]["forecast_mode"] in {"insufficient_data", "weak_signal", "strong_signal"}


def test_one_step_row_has_user_numeric():
    df = _normalized(_dataset("strong"))
    panel = build_v1_panel_feature_matrix(build_daily_panel_from_transactions(df))
    spec = derive_v1_feature_spec(panel)
    sku_hist = panel[panel["product_id"] == "sku-a"].copy()
    row = build_v1_one_step_features(sku_hist, pd.Timestamp("2025-08-01"), {"product_id": "sku-a", "category": "cat-1", "region": "US", "channel": "online", "segment": "retail", "user_factor_num__temperature": 22.0}, 120, spec)
    assert "user_factor_num__temperature" in row.columns
