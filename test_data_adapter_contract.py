import pandas as pd

from data_adapter import build_auto_mapping, build_daily_panel_from_transactions, normalize_transactions


def test_normalize_transactions_missing_optional_columns_does_not_crash():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "product_id": ["sku-1"],
            "category": ["cat"],
            "price": [100.0],
            "quantity": [2.0],
        }
    )
    mapping = build_auto_mapping(df.columns.tolist())
    out, quality = normalize_transactions(df, mapping)
    assert len(out) == 1
    assert quality.get("errors", []) == []


def test_discount_percent_0_100_is_converted_to_rate():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "product_id": ["sku-1"],
            "price": [100.0],
            "quantity": [1.0],
            "discount_rate": [25.0],
        }
    )
    out, _ = normalize_transactions(df, build_auto_mapping(df.columns.tolist()))
    assert abs(float(out.loc[0, "discount_rate"]) - 0.25) < 1e-9


def test_discount_amount_is_not_silently_used_as_rate():
    df = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "product_id": ["sku-1"],
            "price": [100.0],
            "quantity": [1.0],
            "discount_value": [20.0],
        }
    )
    out, _ = normalize_transactions(df, build_auto_mapping(df.columns.tolist()))
    assert float(out.loc[0, "discount_rate"]) == 0.0


def test_weighted_daily_aggregation_price_cost_discount():
    txn = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01"],
            "product_id": ["sku-1", "sku-1"],
            "category": ["cat", "cat"],
            "quantity": [1.0, 3.0],
            "price": [100.0, 200.0],
            "cost": [60.0, 120.0],
            "discount_rate": [0.1, 0.2],
            "revenue": [100.0, 600.0],
        }
    )
    panel = build_daily_panel_from_transactions(txn)
    row = panel.iloc[0]
    assert abs(float(row["price"]) - 175.0) < 1e-9
    assert abs(float(row["cost"]) - 105.0) < 1e-9
    assert abs(float(row["discount_rate"]) - 0.175) < 1e-9


def test_gap_fill_no_backward_fill_leakage():
    txn = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-03"],
            "product_id": ["sku-1", "sku-1"],
            "category": ["cat", "cat"],
            "quantity": [1.0, 1.0],
            "price": [100.0, 110.0],
            "cost": [60.0, 65.0],
            "discount_rate": [0.1, 0.2],
            "revenue": [100.0, 110.0],
        }
    )
    panel = build_daily_panel_from_transactions(txn)
    gap_row = panel[panel["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    assert abs(float(gap_row["price"]) - 100.0) < 1e-9


def test_sample_transactions_csv_normalizes_and_runs_v2():
    from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2

    raw = pd.read_csv("sample_transactions.csv")
    mapping = build_auto_mapping(raw.columns.tolist())
    normalized, quality = normalize_transactions(raw, mapping)

    assert len(normalized) > 0
    assert len(quality.get("errors", [])) == 0

    target_row = normalized.iloc[0]
    out = run_full_pricing_analysis_v2(
        normalized,
        target_category=str(target_row.get("category", "unknown")),
        target_sku=str(target_row["product_id"]),
        horizon_days=7,
    )
    assert out.get("analysis_engine") == "v2_decomposed_baseline_factor_shock"
