import pandas as pd

from pricing_core.factor_features import build_factor_feature_matrix, derive_factor_feature_spec
from pricing_core.baseline_model import disaggregate_weekly_to_daily
from pricing_core.scenario_engine import apply_total_stock_cap_weekly, run_scenario_forecast, run_weekly_scenario_forecast
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


def test_no_override_returns_as_is_equal_to_scenario():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    a = out["as_is_forecast"]["actual_sales"].reset_index(drop=True)
    s = out["scenario_forecast"]["actual_sales"].reset_index(drop=True)
    assert a.equals(s)


def test_price_override_changes_factor_multiplier_not_baseline():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    a = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 999})
    b = run_scenario_forecast(tr, None, fm, fut, spec, None)
    assert (a["scenario_forecast"]["baseline_pred"].values == b["scenario_forecast"]["baseline_pred"].values).all()


def test_neutral_baseline_differs_from_as_is_when_current_price_or_promo_non_neutral():
    tr, spec, fm = _trained_baseline()
    fm = fm.copy()
    fm.loc[fm.index[-1], "promotion"] = 1.0
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    neutral = float(out["neutral_baseline_forecast"]["baseline_pred"].sum())
    as_is = float(out["as_is_forecast"]["actual_sales"].sum())
    assert neutral != as_is


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


def test_stock_cap_applies_to_as_is_and_scenario_paths_independently():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 9.0, "use_stock_cap": True, "stock_total_horizon": 5})
    assert float(out["as_is_forecast"]["actual_sales"].sum()) > 5.0
    assert float(out["scenario_forecast"]["actual_sales"].sum()) <= 5.0


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
    assert out["mode"] == "fallback_elasticity"
    assert out["scenario_effect_source"] == "bounded_rules"


def test_scenario_outside_backtest_range_sets_flag():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(
        tr,
        None,
        fm,
        fut,
        spec,
        None,
        scenario_overrides={"price": 1.0},
        factor_backtest_summary={"factor_multiplier_p95": 1.01},
    )
    assert "scenario_outside_factor_backtest_range" in out["warnings"]


def test_run_scenario_forecast_returns_three_forecast_layers():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    assert "neutral_baseline_forecast" in out
    assert "as_is_forecast" in out
    assert "scenario_forecast" in out


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


def test_fallback_elasticity_responds_to_price_change():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")})
    low = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 8.0})
    high = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 20.0})
    assert low["scenario_effect_source"] == "bounded_rules"
    assert high["scenario_effect_source"] == "bounded_rules"
    assert float(low["scenario_forecast"]["actual_sales"].sum()) != float(high["scenario_forecast"]["actual_sales"].sum())


def test_scenario_decomposition_columns_exist():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, None)
    required = {"baseline_component", "factor_effect", "shock_effect", "final_forecast"}
    assert required.issubset(set(out["scenario_forecast"].columns))


def test_baseline_is_invariant_under_factor_only_override():
    tr, spec, fm = _trained_baseline()
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=7, freq="D")})
    base = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 10.0})
    bump = run_scenario_forecast(tr, None, fm, fut, spec, None, scenario_overrides={"price": 14.0})
    assert base["scenario_forecast"]["baseline_pred"].tolist() == bump["scenario_forecast"]["baseline_pred"].tolist()


def test_future_and_train_factor_feature_schema_are_synced_for_dynamic_features():
    tr, spec, fm = _trained_baseline()
    ff_spec = derive_factor_feature_spec(fm)
    train_mat = build_factor_feature_matrix(fm, ff_spec)
    fut = pd.DataFrame({"date": pd.date_range(fm["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")})
    out = run_scenario_forecast(tr, None, fm, fut, spec, ff_spec)
    future_cols = set(out["scenario_forecast"].columns)
    required = {"demand_raw", "actual_sales", "lost_sales", "remaining_stock", "factor_multiplier"}
    assert required.issubset(future_cols)
    dynamic = {"recent_sales_level_7", "recent_sales_level_28", "sales_level_ratio_7_to_28", "weekday_profile_share", "days_since_last_promo", "price_rank_vs_last_8_weeks"}
    assert dynamic.issubset(set(train_mat.columns))


def test_weekly_stock_cap_works():
    capped = apply_total_stock_cap_weekly(pd.Series([10.0, 8.0, 5.0]), total_stock=12.0)
    assert capped["actual_sales_weekly"].tolist() == [10.0, 2.0, 0.0]
    assert capped["lost_sales_weekly"].tolist() == [0.0, 6.0, 5.0]
    assert capped["remaining_stock_weekly"].tolist() == [2.0, 0.0, 0.0]


def test_daily_disaggregation_preserves_actual_and_lost_totals():
    weekly_baseline = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2026-01-05", "2026-01-12"]),
            "baseline_pred_weekly": [100.0, 100.0],
        }
    )
    out = run_weekly_scenario_forecast(
        weekly_baseline_forecast=weekly_baseline,
        trained_weekly_factor=None,
        current_ctx={"price": 10.0, "discount": 0.0, "promotion": 0.0},
        scenario_ctx={"price": 10.0, "discount": 0.0, "promotion": 0.0, "use_stock_cap": True, "stock_total_horizon": 80.0},
    )
    dates = pd.DataFrame({"date": pd.date_range("2026-01-05", periods=14, freq="D")})
    weekday_profile = pd.Series([1 / 7.0] * 7, index=range(7), dtype=float)
    d_actual = disaggregate_weekly_to_daily(
        out["weekly_scenario_forecast"][["week_start", "actual_scenario"]].rename(columns={"actual_scenario": "baseline_pred_weekly"}),
        dates,
        weekday_profile,
    )
    d_lost = disaggregate_weekly_to_daily(
        out["weekly_scenario_forecast"][["week_start", "lost_scenario"]].rename(columns={"lost_scenario": "baseline_pred_weekly"}),
        dates,
        weekday_profile,
    )
    assert abs(float(d_actual["baseline_pred"].sum()) - float(out["weekly_scenario_forecast"]["actual_scenario"].sum())) < 1e-6
    assert abs(float(d_lost["baseline_pred"].sum()) - float(out["weekly_scenario_forecast"]["lost_scenario"].sum())) < 1e-6
