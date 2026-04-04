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
    monkeypatch.setattr(app, "run_v1_what_if_projection", lambda *_a, **kwargs: captured.update(kwargs) or {"profit_total": 1, "revenue_total": 1, "demand_total": 1})

    app.render_scenario_lab({"current_price": 100.0, "current_profit": 1000.0, "forecast_horizon_days": 30, "forecast_current": pd.DataFrame({"price": [100.0], "pred_sales": [30.0]}), "_trained_bundle": {"base_ctx": {"stock": 50.0, "review_score": 4.5, "reviews_count": 100.0}, "feature_spec": {"user_factor_features": []}}})
    assert captured.get("cost_multiplier") == 1.15
