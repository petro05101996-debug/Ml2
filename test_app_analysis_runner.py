import pandas as pd

import app_analysis_runner as runner


def test_runner_invokes_v2(monkeypatch):
    captured = {}

    def fake_run_full_pricing_analysis_v2(universal_txn, target_category, target_sku, **kwargs):
        captured["kwargs"] = kwargs
        return {
            "ok": True,
            "analysis_engine": "v2_decomposed_baseline_factor_shock",
            "delta_summary": pd.DataFrame([{"baseline_total_demand": 10, "scenario_total_demand": 11, "demand_delta_pct": 0.1, "baseline_total_revenue": 100, "scenario_total_revenue": 110, "revenue_delta_pct": 0.1, "baseline_total_profit": 50, "scenario_total_profit": 55, "profit_delta_pct": 0.1}]),
            "factor_model_trained": False,
            "mode": "baseline_only",
            "ood_flags": [],
            "confidence": {"overall_confidence": "low", "issues": []},
        }

    monkeypatch.setattr(runner, "run_full_pricing_analysis_v2", fake_run_full_pricing_analysis_v2)
    ctx = {"load_mode": "Universal CSV", "target_category": "cat", "target_sku": "sku-1", "universal_txn": pd.DataFrame({"product_id": ["sku-1"]}), "forecast_horizon_days": 14}
    out = runner.run_analysis_from_context(ctx)
    assert out["analysis_engine"] == "v2_decomposed_baseline_factor_shock"
    assert out["analysis_route"] == "runner_to_v2_decomposed"
    assert "v2_result_contract" in out
    assert captured["kwargs"]["horizon_days"] == 14
