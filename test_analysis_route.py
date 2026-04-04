import pandas as pd

import app_analysis_runner as runner


def test_analysis_route_for_universal_mode(monkeypatch):
    def fake_run_full_pricing_analysis_universal_v1(universal_txn, target_category, target_sku, **kwargs):
        return {"analysis_engine": "v1_universal"}

    monkeypatch.setattr(runner, "run_full_pricing_analysis_universal_v1", fake_run_full_pricing_analysis_universal_v1)
    ctx = {"load_mode": "Универсальный CSV", "target_category": "cat", "target_sku": "sku-1", "universal_txn": pd.DataFrame({"product_id": ["sku-1"]})}
    result = runner.run_analysis_from_context(ctx)
    assert result["analysis_engine"] == "v1_universal"
    assert result["analysis_route"] == "runner_to_v1_universal"
