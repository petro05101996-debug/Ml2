from contextlib import contextmanager

import app
from app_domain import (
    PRESET_AGGRESSIVE_INCREASE,
    PRESET_CAUTIOUS_INCREASE,
    PRESET_COST_FREIGHT_STRESS,
    PRESET_KEEP_CURRENT,
    PRESET_LIMITED_STOCK,
    PRESET_LOWER_FOR_VOLUME,
    PRESET_PROMO_PUSH,
    REQUIRED_PRESET_KEYS,
    build_seller_scenario_presets,
)


def test_required_presets_exist_in_domain_builder() -> None:
    presets = build_seller_scenario_presets(base_price=100.0, base_ctx={"stock": 50}, horizon_days=30)
    missing = [key for key in REQUIRED_PRESET_KEYS if key not in presets]
    assert missing == []


def test_required_preset_constants_are_unique() -> None:
    required = [
        PRESET_KEEP_CURRENT,
        PRESET_CAUTIOUS_INCREASE,
        PRESET_AGGRESSIVE_INCREASE,
        PRESET_LOWER_FOR_VOLUME,
        PRESET_PROMO_PUSH,
        PRESET_COST_FREIGHT_STRESS,
        PRESET_LIMITED_STOCK,
    ]
    assert len(required) == len(set(required))


def test_scenario_lab_passes_cost_multiplier_to_projection(monkeypatch) -> None:
    captured = {}

    class _SessionState(dict):
        def __getattr__(self, key):
            return self.get(key)

        def __setattr__(self, key, value):
            self[key] = value

    class _ColumnControl:
        def number_input(self, _label, **kwargs):
            return kwargs.get("value")

        def slider(self, _label, *args, **kwargs):
            if "value" in kwargs:
                return kwargs["value"]
            return args[2]

        def select_slider(self, _label, **kwargs):
            return kwargs.get("value")

        def metric(self, *_args, **_kwargs):
            return None

    @contextmanager
    def _expander(*_args, **_kwargs):
        yield None

    class _FakeSt:
        def __init__(self):
            self.session_state = _SessionState()

        def markdown(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

        def selectbox(self, _label, options, **_kwargs):
            return options[0]

        def slider(self, _label, *args, **kwargs):
            if "value" in kwargs:
                return kwargs["value"]
            return args[2]

        def number_input(self, _label, **kwargs):
            return kwargs.get("value")

        def columns(self, n):
            return [_ColumnControl() for _ in range(n)]

        def expander(self, *_args, **_kwargs):
            return _expander()

        def caption(self, *_args, **_kwargs):
            return None

    def fake_run_what_if_projection(*_args, **kwargs):
        captured.update(kwargs)
        return {"profit_total": 1000.0, "revenue_total": 2000.0, "demand_total": 30.0}

    monkeypatch.setattr(app, "st", _FakeSt())
    monkeypatch.setattr(
        app,
        "build_seller_scenario_presets",
        lambda *_args, **_kwargs: {key: {"price": 100.0, "demand_multiplier": 1.0, "freight_multiplier": 1.0, "horizon_days": 30, "cost_multiplier": 1.15} for key in REQUIRED_PRESET_KEYS},
    )
    monkeypatch.setattr(app, "run_what_if_projection", fake_run_what_if_projection)

    result_bundle = {
        "current_price": 100.0,
        "current_profit": 5000.0,
        "forecast_horizon_days": 30,
        "forecast_current": {"price": [100.0], "pred_sales": [30.0]},
        "_trained_bundle": {"base_ctx": {"stock": 50.0, "review_score": 4.5, "reviews_count": 100.0}},
    }
    import pandas as pd

    result_bundle["forecast_current"] = pd.DataFrame(result_bundle["forecast_current"])
    app.render_scenario_lab(result_bundle)
    assert captured.get("cost_multiplier") == 1.15


def test_results_page_action_buttons_trigger_expected_calculations(monkeypatch) -> None:
    import pandas as pd

    calls = {"scenario": 0, "sensitivity": 0}

    class _SessionState(dict):
        def __getattr__(self, key):
            return self.get(key)

        def __setattr__(self, key, value):
            self[key] = value

    class _Column:
        def markdown(self, *_args, **_kwargs):
            return None

    @contextmanager
    def _expander(*_args, **_kwargs):
        yield None

    class _FakeSt:
        def __init__(self):
            self.session_state = _SessionState({"scenario_table": None, "sensitivity_df": None})

        def markdown(self, *_args, **_kwargs):
            return None

        def columns(self, spec):
            n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
            return [_Column() for _ in range(n)]

        def button(self, label, **_kwargs):
            return label in {
                "Запустить: текущий vs рекомендованный vs консервативный",
                "Запустить карту чувствительности",
            }

        def dataframe(self, *_args, **_kwargs):
            return None

        def plotly_chart(self, *_args, **_kwargs):
            return None

        def expander(self, *_args, **_kwargs):
            return _expander()

        def write(self, *_args, **_kwargs):
            return None

    def fake_run_scenario_set(*_args, **_kwargs):
        calls["scenario"] += 1
        return pd.DataFrame({"scenario": ["Baseline"], "profit": [100.0], "delta_profit": [0.0]})

    def fake_build_sensitivity_grid(*_args, **_kwargs):
        calls["sensitivity"] += 1
        return pd.DataFrame({"price": [100.0], "demand_multiplier": [1.0], "profit": [100.0]})

    monkeypatch.setattr(app, "st", _FakeSt())
    monkeypatch.setattr(app, "run_scenario_set", fake_run_scenario_set)
    monkeypatch.setattr(app, "build_sensitivity_grid", fake_build_sensitivity_grid)
    monkeypatch.setattr(app, "generate_explanation", lambda *_args, **_kwargs: {"summary": "ok"})
    monkeypatch.setattr(
        app,
        "build_default_scenario_inputs",
        lambda *_args, **_kwargs: [
            {"name": "Baseline", "price": 100.0, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "horizon_days": 30, "cost_multiplier": 1.0},
            {"name": "Scenario A", "price": 101.0, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "horizon_days": 30, "cost_multiplier": 1.0},
            {"name": "Scenario B", "price": 99.0, "freight_multiplier": 1.0, "demand_multiplier": 1.0, "horizon_days": 30, "cost_multiplier": 1.0},
        ],
    )

    r = {
        "current_price": 100.0,
        "best_price": 102.0,
        "current_profit": 1000.0,
        "best_profit": 1100.0,
        "current_revenue": 3000.0,
        "best_revenue": 3200.0,
        "current_volume": 30.0,
        "best_volume": 31.0,
        "forecast_horizon_days": 30,
        "forecast_current": pd.DataFrame({"price": [100.0], "pred_sales": [30.0]}),
        "profit_curve": pd.DataFrame({"price": [95.0, 100.0, 105.0], "adjusted_profit": [900.0, 1000.0, 980.0]}),
        "business_recommendation": {"structured": {}, "seller_friendly_reason": "ok"},
        "_trained_bundle": {"confidence": 0.8, "base_ctx": {"stock": 50.0}},
        "data_quality": {},
    }
    app.render_results_page(r)
    assert calls["scenario"] == 1
    assert calls["sensitivity"] == 1
    assert app.st.session_state["scenario_table"] is not None
    assert app.st.session_state["sensitivity_df"] is not None
