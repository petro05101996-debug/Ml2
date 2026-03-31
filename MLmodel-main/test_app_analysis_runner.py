from io import StringIO

import pandas as pd

import app_analysis_runner as runner


def test_read_csv_input_accepts_dataframe():
    src = pd.DataFrame({"order_id": [1], "customer_id": ["c1"]})
    out = runner._read_csv_input(src)
    assert out.equals(src)
    assert out is not src


def test_run_analysis_from_context_handles_non_seekable_inputs(monkeypatch):
    captured = {}

    def fake_run_full_pricing_analysis(orders, items, products, reviews, target_category, target_sku, **kwargs):
        captured["orders"] = orders
        captured["items"] = items
        captured["products"] = products
        captured["reviews"] = reviews
        captured["target_category"] = target_category
        captured["target_sku"] = target_sku
        captured["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(runner, "run_full_pricing_analysis", fake_run_full_pricing_analysis)

    ctx = {
        "load_mode": "Legacy Olist (3 CSV)",
        "target_category": "cat",
        "target_sku": "sku-1",
        "orders_file": pd.DataFrame({"order_id": ["o1"]}),
        "items_file": StringIO("order_id,product_id,price\no1,sku-1,100\n"),
        "products_file": pd.DataFrame({"product_id": ["sku-1"], "product_category_name": ["cat"]}),
        "reviews_file": None,
        "forecast_horizon_days": 14,
        "caution_level": "Средний",
    }

    out = runner.run_analysis_from_context(ctx)

    assert out == {"ok": True}
    assert captured["target_category"] == "cat"
    assert captured["target_sku"] == "sku-1"
    assert captured["kwargs"]["horizon_days"] == 14
    assert "order_item_id" in captured["items"].columns
    assert "freight_value" in captured["items"].columns


def test_run_analysis_from_context_supports_russian_universal_load_mode(monkeypatch):
    captured = {}

    def fake_run_full_pricing_analysis_universal(universal_txn, target_category, target_sku, **kwargs):
        captured["universal_txn"] = universal_txn
        captured["target_category"] = target_category
        captured["target_sku"] = target_sku
        captured["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(runner, "run_full_pricing_analysis_universal", fake_run_full_pricing_analysis_universal)

    universal_df = pd.DataFrame({"product_id": ["sku-1"], "category": ["cat"], "price": [100]})
    ctx = {
        "load_mode": "Универсальный CSV",
        "target_category": "cat",
        "target_sku": "sku-1",
        "universal_txn": universal_df,
        "forecast_horizon_days": 30,
        "caution_level": "Высокий",
    }

    out = runner.run_analysis_from_context(ctx)

    assert out == {"ok": True}
    assert captured["universal_txn"].equals(universal_df)
    assert captured["target_category"] == "cat"
    assert captured["target_sku"] == "sku-1"
    assert captured["kwargs"]["horizon_days"] == 30
