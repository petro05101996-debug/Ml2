import pandas as pd

import app_analysis_runner as runner


def test_analysis_route_for_v2_mode(monkeypatch):
    def fake_run_full_pricing_analysis_v2(universal_txn, target_category, target_sku, **kwargs):
        return {"analysis_engine": "v2_decomposed_baseline_factor_shock"}

    monkeypatch.setattr(runner, "run_full_pricing_analysis_v2", fake_run_full_pricing_analysis_v2)
    ctx = {"load_mode": "Универсальный CSV", "target_category": "cat", "target_sku": "sku-1", "universal_txn": pd.DataFrame({"product_id": ["sku-1"]})}
    result = runner.run_analysis_from_context(ctx)
    assert result["analysis_engine"] == "v2_decomposed_baseline_factor_shock"
    assert result["analysis_route"] == "runner_to_v2_decomposed"
