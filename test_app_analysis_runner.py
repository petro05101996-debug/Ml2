import pandas as pd

import app_analysis_runner as runner


def test_run_analysis_from_context_universal_only(monkeypatch):
    captured = {}

    def fake_run_full_pricing_analysis_universal_v1(universal_txn, target_category, target_sku, **kwargs):
        captured.update({"universal_txn": universal_txn, "target_category": target_category, "target_sku": target_sku, "kwargs": kwargs})
        return {"ok": True, "analysis_engine": "v1_universal"}

    monkeypatch.setattr(runner, "run_full_pricing_analysis_universal_v1", fake_run_full_pricing_analysis_universal_v1)
    universal_df = pd.DataFrame({"product_id": ["sku-1"], "category": ["cat"], "price": [100]})
    ctx = {"load_mode": "Универсальный CSV", "target_category": "cat", "target_sku": "sku-1", "universal_txn": universal_df, "forecast_horizon_days": 30, "caution_level": "Высокий"}

    out = runner.run_analysis_from_context(ctx)
    assert out["ok"] is True
    assert out["analysis_engine"] == "v1_universal"
    assert captured["universal_txn"].equals(universal_df)
