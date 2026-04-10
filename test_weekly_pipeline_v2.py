import numpy as np
import pandas as pd

from data_adapter import build_weekly_panel_from_daily
from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2


def _mk_dataset(kind: str, n_days: int = 180) -> pd.DataFrame:
    d = pd.date_range("2025-01-01", periods=n_days, freq="D")
    base = 120 + 10 * np.sin(np.arange(n_days) / 7.0)
    if kind == "strong":
        price = 12 + 1.5 * np.sin(np.arange(n_days) / 11.0)
        promo = (np.arange(n_days) % 14 == 0).astype(float)
        sales = base * (price / np.median(price)) ** -1.3 * (1 + 0.25 * promo)
    elif kind == "weak":
        price = np.full(n_days, 12.0)
        promo = (np.arange(n_days) % 90 == 0).astype(float)
        sales = base * (1 + 0.03 * promo)
    else:  # seasonal
        price = 11.5 + 0.3 * np.sin(np.arange(n_days) / 30.0)
        promo = (np.arange(n_days) % 21 == 0).astype(float)
        season = 1 + 0.35 * np.sin(np.arange(n_days) / 20.0)
        sales = base * season * (price / np.median(price)) ** -0.35 * (1 + 0.12 * promo)

    sales = np.clip(sales, 0.0, None)
    return pd.DataFrame(
        {
            "date": d,
            "product_id": "sku-1",
            "category": "cat-1",
            "region": "US",
            "channel": "online",
            "segment": "retail",
            "price": price,
            "discount": 0.05 * promo,
            "promotion": promo,
            "freight_value": 0.5,
            "cost": np.maximum(price * 0.62, 0.01),
            "sales": sales,
            "quantity": sales,
            "revenue": sales * price,
        }
    )


def test_weekly_aggregation_has_required_columns():
    daily = _mk_dataset("seasonal", 84)
    weekly = build_weekly_panel_from_daily(daily)["weekly_panel"]
    assert {"sales_week", "lag_1_week_sales", "rolling_mean_4w", "avg_price_week", "promo_share_week"}.issubset(set(weekly.columns))


def test_weekly_pipeline_outputs_decomposition_keys():
    df = _mk_dataset("strong", 180)
    out = run_full_pricing_analysis_v2(df, target_category="cat-1", target_sku="sku-1", horizon_days=28)
    required = {
        "weekly_history",
        "weekly_baseline_forecast",
        "weekly_factor_effect_as_is",
        "weekly_factor_effect_scenario",
        "weekly_final_forecast_as_is",
        "weekly_final_forecast_scenario",
        "weekly_factor_target",
        "daily_presented_as_is",
        "daily_presented_scenario",
        "factor_effect_source",
    }
    assert required.issubset(set(out.keys()))


def test_daily_presented_preserves_weekly_total():
    df = _mk_dataset("strong", 180)
    out = run_full_pricing_analysis_v2(df, target_category="cat-1", target_sku="sku-1", horizon_days=21)
    daily_sum = float(pd.to_numeric(out["daily_presented_scenario"]["actual_sales"], errors="coerce").fillna(0.0).sum())
    weekly_sum = float(pd.to_numeric(out["weekly_final_forecast_scenario"]["sales"], errors="coerce").fillna(0.0).sum())
    assert abs(daily_sum - weekly_sum) < 1e-6


def test_weak_factor_dataset_uses_bounded_rules_source():
    df = _mk_dataset("weak", 180)
    out = run_full_pricing_analysis_v2(df, target_category="cat-1", target_sku="sku-1", horizon_days=14)
    assert out["factor_effect_source"] == "bounded_rules"


def test_weekly_factor_target_is_ratio_and_clipped():
    df = _mk_dataset("strong", 180)
    out = run_full_pricing_analysis_v2(df, target_category="cat-1", target_sku="sku-1", horizon_days=14)
    wk = out["weekly_factor_target"]
    assert "factor_target_week" in wk.columns
    if len(wk):
        assert float(pd.to_numeric(wk["factor_target_week"], errors="coerce").min()) >= 0.4 - 1e-9
        assert float(pd.to_numeric(wk["factor_target_week"], errors="coerce").max()) <= 1.8 + 1e-9
