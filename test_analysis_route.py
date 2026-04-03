import pandas as pd

import app_analysis_runner as runner


def test_analysis_route_for_universal_mode(monkeypatch):
    def fake_run_full_pricing_analysis_universal_v1(universal_txn, target_category, target_sku, **kwargs):
        return {"analysis_engine": "v1_universal"}

    monkeypatch.setattr(runner, "run_full_pricing_analysis_universal_v1", fake_run_full_pricing_analysis_universal_v1)

    ctx = {
        "load_mode": "Универсальный CSV",
        "target_category": "cat",
        "target_sku": "sku-1",
        "universal_txn": pd.DataFrame({"product_id": ["sku-1"]}),
    }

    result = runner.run_analysis_from_context(ctx)

    assert result["analysis_engine"] == "v1_universal"
    assert result["analysis_route"] == "runner_to_v1_universal"


def test_analysis_route_for_legacy_mode(monkeypatch):
    def fake_run_full_pricing_analysis(orders, items, products, reviews, target_category, target_sku, **kwargs):
        return {"analysis_engine": "legacy_core"}

    monkeypatch.setattr(runner, "run_full_pricing_analysis", fake_run_full_pricing_analysis)

    ctx = {
        "load_mode": "Legacy Olist (3 CSV)",
        "target_category": "cat",
        "target_sku": "sku-1",
        "orders_file": pd.DataFrame({"order_id": ["o1"]}),
        "items_file": pd.DataFrame({"order_id": ["o1"], "product_id": ["sku-1"], "price": [100]}),
        "products_file": pd.DataFrame({"product_id": ["sku-1"], "product_category_name": ["cat"]}),
        "reviews_file": None,
    }

    result = runner.run_analysis_from_context(ctx)

    assert result["analysis_engine"] == "legacy_core"
    assert result["analysis_route"] == "runner_to_legacy_core"
