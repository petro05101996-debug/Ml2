import pandas as pd
import numpy as np
import json

from data_adapter import build_auto_mapping, normalize_transactions, build_daily_from_transactions
from what_if import build_sensitivity_grid, run_scenario_set
from app import (
    apply_weekly_fallback_projection,
    build_manual_scenario_artifacts,
    read_uploaded_csv_safely,
    robust_clean_dirty_data,
    run_full_pricing_analysis_universal,
    run_what_if_projection,
)


def test_normalize_and_daily_build():
    df = pd.DataFrame(
        {
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "SKU": ["A", "A", "B"],
            "Unit Price": [100, 110, 90],
            "qty": [1, 2, 1],
        }
    )
    mapping = build_auto_mapping(list(df.columns))
    norm, quality = normalize_transactions(df, mapping)
    assert not quality.get("errors")
    assert "warnings" in quality
    daily = build_daily_from_transactions(norm, "A")
    assert len(daily) >= 2
    assert (daily["price"] > 0).all()


def test_read_uploaded_csv_safely_supports_semicolon_delimiter():
    class DummyUpload:
        def __init__(self, payload: bytes):
            self._payload = payload

        def getvalue(self):
            return self._payload

    payload = "date;product_id;price\n2025-01-01;sku-1;100\n".encode("utf-8")
    df = read_uploaded_csv_safely(DummyUpload(payload))
    assert set(df.columns) == {"date", "product_id", "price"}
    assert len(df) == 1


def test_scenario_runner_baseline_fallback():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        d = kwargs.get("demand_multiplier", 1.0)
        return {"demand_total": 10 * d, "revenue_total": 10 * d * p, "profit_total": 2 * d * p, "confidence": 0.7}

    scenarios = [
        {"name": "Scenario A", "price": 100, "freight_multiplier": 1, "demand_multiplier": 1, "horizon_days": 30},
        {"name": "Scenario B", "price": 90, "freight_multiplier": 1, "demand_multiplier": 1.1, "horizon_days": 30},
    ]
    out = run_scenario_set({}, scenarios, runner)
    assert len(out) == 2
    assert "delta_profit" in out.columns


def test_sensitivity_uses_demand_axis():
    def runner(_bundle, **kwargs):
        p = kwargs["manual_price"]
        demand_mult = kwargs.get("demand_multiplier", 1.0)
        return {"profit_total": p * (2 - demand_mult), "profit_total_adjusted": p * (2 - demand_mult), "demand_total": 10 * demand_mult}

    out = build_sensitivity_grid({}, base_price=100.0, runner=runner, price_steps=3, demand_steps=3)
    assert "demand_multiplier" in out.columns
    assert "profit_adjusted" in out.columns
    assert "risk_zone" in out.columns


def _make_txn(n_days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 100 + (i % 8)
        promo = 1.0 if i % 14 == 0 else 0.0
        freight = 5.0 + (i % 3)
        qty = max(1.0, 40 - 0.12 * price + 6.0 * promo - 0.3 * freight)
        rows.append(
            {
                "date": d,
                "product_id": "sku-1",
                "category": "cat-a",
                "price": price,
                "quantity": qty,
                "revenue": qty * price * (1.0 - 0.05 * promo),
                "cost": price * 0.65,
                "discount": 0.05 * promo,
                "freight_value": freight,
                "promotion": promo,
                "rating": 4.4,
                "reviews_count": 10 + (i % 5),
            }
        )
    return pd.DataFrame(rows)


def _analyze():
    return run_full_pricing_analysis_universal(_make_txn(), "cat-a", "sku-1")


def test_scenario_not_equal_as_is_when_price_changes():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.1)
    as_is_total = float(res["as_is_forecast"]["actual_sales"].sum())
    assert scenario["demand_total"] != as_is_total


def test_price_increase_reduces_demand_monotone():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price * 0.95)
    high = run_what_if_projection(bundle, manual_price=base_price * 1.10)
    assert high["demand_total"] <= low["demand_total"]


def test_promotion_on_not_decrease_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    off = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 0.0})
    on = run_what_if_projection(bundle, manual_price=base_price, overrides={"promotion": 1.0})
    assert on["demand_total"] >= off["demand_total"]


def test_freight_increase_not_increase_demand():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    low = run_what_if_projection(bundle, manual_price=base_price, freight_multiplier=1.0)
    high = run_what_if_projection(bundle, manual_price=base_price, freight_multiplier=1.5)
    assert high["demand_total"] <= low["demand_total"]


def test_analysis_summary_does_not_fake_manual_scenario():
    res = _analyze()
    summary_blob_before = res["analysis_run_summary_json"]
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    out = summary["scenario_output_summary"]
    assert out["scenario_status"] == "not_run"
    assert pd.isna(out["scenario_demand_total"])
    assert pd.isna(out["scenario_revenue_total"])
    assert pd.isna(out["scenario_profit_total"])
    assert res["scenario_forecast"] is None
    assert res["scenario_price"] is None
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    scenario = run_what_if_projection(bundle, manual_price=base_price * 1.15)
    assert scenario["demand_total"] != float(res["as_is_forecast"]["actual_sales"].sum())
    assert scenario["revenue_total"] != float(res["as_is_forecast"]["revenue"].sum())
    assert scenario["profit_total"] != float(res["as_is_forecast"]["profit"].sum())
    # analysis summary is analysis-level artifact and should not mutate after runtime what-if calls.
    assert res["analysis_run_summary_json"] == summary_blob_before
    summary_after = json.loads(summary_blob_before.decode("utf-8"))
    assert summary_after["scenario_output_summary"]["scenario_status"] == "not_run"


def test_manual_scenario_artifacts_created_after_runtime_what_if():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.2)
    summary_blob, daily_blob = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    daily = pd.read_csv(pd.io.common.BytesIO(daily_blob))
    assert summary["scenario_status"] == "computed"
    assert len(daily) > 0
    assert "scenario_demand" in daily.columns


def test_manual_scenario_totals_not_equal_as_is_when_price_changes():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.25)
    summary_blob, _ = build_manual_scenario_artifacts(res, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    assert summary["requested_price"] != base_price
    assert abs(summary["delta_vs_as_is"]["demand_total"]) > 0
    assert "applied_overrides" in summary


def test_analysis_summary_contains_schema_version_and_commit_signature():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    cfg = summary["config"]
    assert "git_commit" in cfg and str(cfg["git_commit"]) != ""
    assert "code_signature" in cfg and len(str(cfg["code_signature"])) >= 12
    assert cfg["artifact_schema_version"] == "v39.1"


def test_downloaded_holdout_schema_matches_current_version():
    res = _analyze()
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    cols = set(holdout.columns)
    assert {"baseline_pred_sales", "price_effect_multiplier", "uplift_multiplier", "final_pred_sales"}.issubset(cols)
    assert {"pred_baseline", "pred_direct", "pred_final", "w_direct", "direct_features"}.isdisjoint(cols)


def test_weekly_fallback_projection_preserves_totals():
    hist = _make_txn(120)
    base = pd.DataFrame(
        {
            "date": pd.date_range("2025-06-01", periods=21, freq="D"),
            "actual_sales": np.linspace(10, 20, 21),
            "revenue": np.linspace(100, 200, 21),
            "profit": np.linspace(30, 80, 21),
            "lost_sales": np.linspace(0, 5, 21),
        }
    )
    out = apply_weekly_fallback_projection(base, hist)
    for col in ["actual_sales", "revenue", "profit", "lost_sales"]:
        assert abs(float(out[col].sum()) - float(base[col].sum())) < 1e-6


def test_holdout_final_has_decomposition():
    res = _analyze()
    holdout = pd.read_csv(pd.io.common.BytesIO(res["holdout_predictions_csv"]))
    assert {"baseline_pred_sales", "price_effect_multiplier", "uplift_multiplier", "final_pred_sales"}.issubset(set(holdout.columns))


def test_no_backward_fill_leakage_in_daily_builder():
    src = pd.DataFrame(
        [
            {"date": "2025-01-01", "product_id": "sku-1", "category": "cat-a", "price": 100, "quantity": 1, "revenue": 100, "discount": 0.0, "promotion": 0.0},
            {"date": "2025-01-03", "product_id": "sku-1", "category": "cat-a", "price": 100, "quantity": 1, "revenue": 100, "discount": 0.2, "promotion": 1.0},
        ]
    )
    src["date"] = pd.to_datetime(src["date"])
    daily = build_daily_from_transactions(src, "sku-1", target_category="cat-a")
    mid = daily[daily["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    assert float(mid["discount"]) == 0.0
    assert float(mid["promotion"]) == 0.0


def test_zero_sales_preserved_after_cleaning():
    df = pd.DataFrame({"sales": [10.0, 0.0, 0.0, 2.0], "price": [10, 10, 10, 10], "freight_value": [1, 1, 1, 1]})
    out = robust_clean_dirty_data(df)
    assert (out["sales"] == 0.0).sum() >= 2


def test_no_double_multiplier():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]), freight_multiplier=1.0)
    high = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]), freight_multiplier=1.5)
    base_f = float(base["daily"]["freight_value"].iloc[0])
    high_f = float(high["daily"]["freight_value"].iloc[0])
    assert abs((high_f / base_f) - 1.5) < 1e-6


def test_uplift_uses_baseline_feature():
    res = _analyze()
    feats = res["_trained_bundle"]["uplift_bundle"]["features"]
    assert "baseline_log_feature" in feats


def test_refit_scope_is_full_history():
    res = _analyze()
    summary = json.loads(res["analysis_run_summary_json"].decode("utf-8"))
    assert summary["config"]["fit_scope"] == "refit_full_history"


def test_raw_adjusted_and_ood_fields_present():
    res = _analyze()
    bundle = res["_trained_bundle"]
    sc = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]) * 2.0)
    assert "profit_total_raw" in sc and "profit_total_adjusted" in sc
    assert "requested_price" in sc and "price_for_model" in sc and "ood_flag" in sc


def test_adjusted_profit_not_above_raw_profit():
    res = _analyze()
    bundle = res["_trained_bundle"]
    scenario = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]))
    assert scenario["profit_total_adjusted"] <= scenario["profit_total_raw"]


def test_higher_cost_multiplier_not_improve_adjusted_profit():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, cost_multiplier=1.0)
    high_cost = run_what_if_projection(bundle, manual_price=base_price, cost_multiplier=1.2)
    assert high_cost["profit_total_adjusted"] <= base["profit_total_adjusted"]


def test_oof_small_mode_parameter_accepts_false():
    from app import make_walk_forward_oof_baseline

    df = _make_txn(60)
    daily = build_daily_from_transactions(df, "sku-1", target_category="cat-a")
    daily = robust_clean_dirty_data(daily)
    daily = daily.assign(log_sales=np.log1p(pd.to_numeric(daily["sales"], errors="coerce").fillna(0.0)))
    features = [c for c in ["price", "discount", "promotion", "is_weekend"] if c in daily.columns]
    out = make_walk_forward_oof_baseline(daily.copy(), features, n_splits=3, small_mode=False)
    assert len(out) == len(daily)


def test_feature_report_truthfulness_for_engineered_features():
    res = _analyze()
    fr = res["feature_report"]
    row = fr[fr["factor_name"] == "sales_lag1"].iloc[0]
    assert bool(row["engineered_feature"]) is True
    assert bool(row["found_in_raw"]) is False
    assert bool(row["present_in_daily"]) is True


def test_small_mode_warnings_present_on_short_history():
    res = run_full_pricing_analysis_universal(_make_txn(60), "cat-a", "sku-1")
    assert res["small_mode_info"]["small_mode"] is True
    assert len(res["warnings"]) > 0


def test_scenario_modeled_price_state_for_clipped_request():
    res = _analyze()
    bundle = res["_trained_bundle"]
    sc = run_what_if_projection(bundle, manual_price=float(bundle["base_ctx"]["price"]) * 3.0)
    assert "price_for_model" in sc
    assert sc["price_clipped"] is True
    res["scenario_price_requested"] = float(sc["requested_price"])
    res["scenario_price_modeled"] = float(sc["price_for_model"])
    res["scenario_price"] = res["scenario_price_modeled"]
    assert res["scenario_price"] == res["scenario_price_modeled"]


def test_stock_is_not_public_scenario_factor():
    res = _analyze()
    fr = res["feature_report"]
    row = fr[fr["factor_name"] == "stock"].iloc[0]
    assert bool(row["used_in_scenario"]) is False
