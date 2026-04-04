import pandas as pd

from pricing_core.v1_features import build_v1_feature_matrix, build_v1_one_step_features, derive_v1_feature_spec


def _synthetic_daily(n=60):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "sales": [10 + (i % 5) for i in range(n)],
        "price": [100 + (i % 3) for i in range(n)],
        "cost": [65 + (i % 3) for i in range(n)],
        "discount": [0.05] * n,
        "promotion": [0.0] * n,
        "stock": [100.0] * n,
        "freight_value": [3.0] * n,
        "review_score": [4.5] * n,
        "reviews_count": [10.0] * n,
        "product_id": ["sku-1"] * n,
        "category": ["cat"] * n,
        "region": ["US"] * n,
        "channel": ["online"] * n,
        "segment": ["retail"] * n,
        "user_factor_num__external_temp": [20 + (i % 7) for i in range(n)],
        "user_factor_num__sales": [1 + (i % 4) for i in range(n)],
        "user_factor_num__profit": [2 + (i % 4) for i in range(n)],
        "user_factor_num__margin": [3 + (i % 4) for i in range(n)],
    })


def test_user_factors_in_feature_spec():
    fm = build_v1_feature_matrix(_synthetic_daily())
    spec = derive_v1_feature_spec(fm)
    assert "user_factor_num__external_temp" in spec["demand_features"]
    assert "user_factor_num__external_temp" in spec["scenario_features"]
    assert "user_factor_num__external_temp" in spec["user_numeric_features"]


def test_leaky_user_factors_blocked():
    fm = build_v1_feature_matrix(_synthetic_daily())
    spec = derive_v1_feature_spec(fm)
    assert "user_factor_num__sales" not in spec["demand_features"]
    assert "user_factor_num__profit" not in spec["demand_features"]
    assert "user_factor_num__margin" not in spec["demand_features"]


def test_one_step_row_includes_user_factor():
    fm = build_v1_feature_matrix(_synthetic_daily())
    spec = derive_v1_feature_spec(fm)
    row = build_v1_one_step_features(fm, pd.Timestamp("2025-03-15"), {"user_factor_num__external_temp": 33.0, "price": 100.0, "product_id": "sku-1", "category": "cat", "region": "US", "channel": "online", "segment": "retail"}, 120, spec)
    assert "user_factor_num__external_temp" in row.columns
