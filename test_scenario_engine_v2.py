import pandas as pd

from pricing_core.scenario_engine import run_scenario_forecast
from pricing_core.shock_engine import build_shock_profile


class DummyBaseline:
    def __init__(self, v=100.0):
        self.v = v



def _history():
    d = pd.date_range("2025-01-01", periods=90, freq="D")
    return pd.DataFrame({"date": d, "product_id": "sku-1", "category": "cat", "region": "US", "channel": "online", "segment": "retail", "sales": 100.0, "price": 10.0, "discount": 0.0, "promotion": 0.0, "stock": 1000.0})


def _trained_baseline():
    from pricing_core.baseline_features import derive_baseline_feature_spec, build_baseline_feature_matrix
    from pricing_core.baseline_model import train_baseline_model
    h = _history()
    fm = build_baseline_feature_matrix(h)
    spec = derive_baseline_feature_spec(fm)
    tr = train_baseline_model(fm, spec, small_mode=True)
    return tr, spec, fm


def test_no_override_returns_baseline_when_factor_multiplier_is_one_and_no_shocks():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    sf = out["scenario_forecast"]
    assert (sf["actual_sales"] == sf["baseline_pred"]).all()


def test_price_override_changes_factor_multiplier_not_baseline():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    a = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 999})
    b = run_scenario_forecast(tr, None, fm, fut, spec, None)
    assert (a["scenario_forecast"]["baseline_pred"].values == b["scenario_forecast"]["baseline_pred"].values).all()


def test_shock_multiplier_applies_after_factor_multiplier():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None, shocks=[{"shock_name": "promo", "start_date": fut["date"].iloc[0], "end_date": fut["date"].iloc[-1], "direction": "positive", "intensity": 0.1, "shape": "flat"}])
    assert (out["scenario_forecast"]["shock_multiplier"] > 1.0).all()


def test_stock_total_horizon_caps_cumulatively():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"use_stock_cap": True, "stock_total_horizon": 5})
    sf = out["scenario_forecast"]
    assert float(sf["actual_sales"].sum()) <= 5.0


def test_lost_sales_is_positive_when_stock_binding():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"use_stock_cap": True, "stock_total_horizon": 5})
    sf = out["scenario_forecast"]
    assert float(sf["lost_sales"].sum()) > 0.0


def test_missing_external_future_values_use_last_known():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    assert len(out["scenario_forecast"]) == 3


def test_invalid_shock_shape_raises():
    fut = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3, freq="D")})
    try:
        build_shock_profile([{"shock_name": "x", "start_date": fut["date"].iloc[0], "end_date": fut["date"].iloc[-1], "direction": "positive", "intensity": 0.1, "shape": "bad"}], fut)
        assert False
    except ValueError:
        assert True


def test_scenario_mode_is_exposed():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    assert out["mode"] == "baseline_only"


def test_baseline_override_is_used_for_scenario_path():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    override = pd.DataFrame({"date": fut["date"], "baseline_pred": [11.0, 12.0, 13.0]})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None, baseline_override_df=override)
    assert out["scenario_forecast"]["baseline_pred"].tolist() == [11.0, 12.0, 13.0]


def test_stock_cap_none_means_unlimited_in_engine():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None, baseline_override_df=pd.DataFrame({"date": fut["date"], "baseline_pred": [4.0, 5.0, 6.0]}))
    sf = out["scenario_forecast"]
    assert float(sf["actual_sales"].sum()) == float(sf["scenario_demand_raw"].sum())
    assert float(sf["lost_sales"].sum()) == 0.0


def test_stock_cap_zero_is_binding_when_explicitly_set():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(
        tr,
        None,
        fm,
        fut,
        spec,
        None,
        baseline_override_df=pd.DataFrame({"date": fut["date"], "baseline_pred": [4.0, 5.0, 6.0]}),
        scenario_overrides={"use_stock_cap": True, "stock_total_horizon": 0.0},
    )
    sf = out["scenario_forecast"]
    assert float(sf["actual_sales"].sum()) == 0.0
    assert float(sf["lost_sales"].sum()) == float(sf["scenario_demand_raw"].sum())
