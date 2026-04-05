from contextlib import contextmanager

import pandas as pd

import app


def test_runtime_no_legacy_imports_in_files():
    assert "from pricing_core.core import *" not in open("app.py", "r", encoding="utf-8").read()
    assert "run_what_if_projection" not in open("pricing_core/__init__.py", "r", encoding="utf-8").read()
    assert "Legacy Olist (3 CSV)" not in open("app_analysis_runner.py", "r", encoding="utf-8").read()


def test_scenario_lab_passes_cost_multiplier_to_projection(monkeypatch):
    captured = {}

    class _SessionState(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    class _ColumnControl:
        def number_input(self, _label, **kwargs): return kwargs.get("value")
        def slider(self, _label, *args, **kwargs): return kwargs.get("value", args[2])
        def select_slider(self, _label, **kwargs): return kwargs.get("value")

    @contextmanager
    def _expander(*_args, **_kwargs):
        yield None

    class _FakeSt:
        def __init__(self): self.session_state = _SessionState()
        def markdown(self, *_a, **_k): return None
        def warning(self, *_a, **_k): return None
        def selectbox(self, _label, options, **_kwargs): return options[0]
        def slider(self, _label, *args, **kwargs): return kwargs.get("value", args[2])
        def number_input(self, _label, **kwargs): return kwargs.get("value")
        def columns(self, n): return [_ColumnControl() for _ in range(n)]
        def expander(self, *_args, **_kwargs): return _expander()

    monkeypatch.setattr(app, "st", _FakeSt())
    monkeypatch.setattr(app, "build_seller_scenario_presets", lambda *_a, **_k: {"Keep current price": {"price": 100.0, "demand_multiplier": 1.0, "freight_multiplier": 1.0, "horizon_days": 30, "cost_multiplier": 1.15, "stock_cap": 50.0}})
    monkeypatch.setattr(app, "run_v2_what_if_projection", lambda *_a, **kwargs: captured.update(kwargs) or {"profit_total": 1, "revenue_total": 1, "demand_total": 1})

    app.render_scenario_lab({"current_price": 100.0, "current_profit": 1000.0, "forecast_horizon_days": 30, "forecast_current": pd.DataFrame({"price": [100.0], "pred_sales": [30.0]}), "_trained_bundle": {"base_ctx": {"stock_total_horizon": 50.0}, "scenario_feature_spec": {"user_numeric_features": []}}})
    assert captured.get("cost_multiplier") == 1.15



def test_v2_bundle_contains_base_ctx_and_scenario_feature_spec():
    from pricing_core.orchestrator_v2 import run_full_pricing_analysis_v2
    d = pd.date_range("2025-01-01", periods=20, freq="D")
    q = pd.Series(range(20)) + 10
    p = pd.Series(range(20)) * 0 + 100
    txn = pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "quantity": q, "price": p, "revenue": q * p, "cost": 60.0})
    out = run_full_pricing_analysis_v2(txn, "cat", "sku-1", horizon_days=5)
    b = out["_trained_bundle"]
    assert "base_ctx" in b
    assert "scenario_feature_spec" in b
