import pandas as pd

from app import build_excel_export_buffer, run_full_pricing_analysis_universal, run_what_if_projection


def _make_txn(n_days: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 100 + (i % 10)
        promo = 1.0 if i % 21 == 0 else 0.0
        freight = 4.0 + (i % 4)
        qty = max(1.0, 45 - 0.15 * price + 7.0 * promo - 0.35 * freight)
        rows.append(
            {
                "date": d,
                "product_id": "sku-contract",
                "category": "cat-contract",
                "price": price,
                "quantity": qty,
                "revenue": qty * price * (1.0 - 0.05 * promo),
                "cost": price * 0.65,
                "discount": 0.05 * promo,
                "freight_value": freight,
                "promotion": promo,
                "rating": 4.5,
                "reviews_count": 10 + (i % 5),
            }
        )
    return pd.DataFrame(rows)


def _analyze():
    return run_full_pricing_analysis_universal(_make_txn(), "cat-contract", "sku-contract")


def test_analysis_only_export_status():
    res = _analyze()
    with pd.ExcelFile(res["excel_buffer"]) as xls:
        summary_df = pd.read_excel(xls, "C_manual_summary")
        assert str(summary_df.loc[0, "artifact_scope"]) == "analysis_only"
        assert bool(summary_df.loc[0, "manual_scenario_present"]) is False
        assert str(summary_df.loc[0, "scenario_calc_mode"]) == "not_applied"
        assert "B_manual_scenario" not in xls.sheet_names
        executive = pd.read_excel(xls, "Executive Summary")
        assert str(executive.loc[0, "manual_scenario_status"]) == "not_applied"


def test_enhanced_scenario_export_status():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.08, scenario_calc_mode="enhanced_local_factors")
    res["scenario_forecast"] = wr["daily"].copy()
    excel_blob = build_excel_export_buffer(res, wr)
    with pd.ExcelFile(excel_blob) as xls:
        summary_df = pd.read_excel(xls, "C_manual_summary")
        assert bool(summary_df.loc[0, "manual_scenario_present"]) is True
        assert str(summary_df.loc[0, "scenario_calc_mode"]) == "enhanced_local_factors"
        assert str(summary_df.loc[0, "active_path_contract"]) == "weekly_ml_baseline + enhanced_local_factor_layer"
        assert str(summary_df.loc[0, "scenario_calculation_path"]) == "enhanced_local_factor_layer"
        assert str(summary_df.loc[0, "final_user_visible_path"]) == "weekly_ml_baseline + enhanced_local_factor_layer"
        assert "B_manual_scenario" in xls.sheet_names
        manual_sheet = pd.read_excel(xls, "B_manual_scenario")
        assert len(manual_sheet) > 0
        assert "Scenario Summary" in xls.sheet_names
        daily_effects = pd.read_excel(xls, "daily_effects_summary")
        assert len(daily_effects) > 0
        effect_breakdown = pd.read_excel(xls, "effect_breakdown")
        assert len(effect_breakdown) > 0


def test_legacy_scenario_still_works():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.02, scenario_calc_mode="legacy_current")
    assert wr["scenario_calc_mode"] == "legacy_current"
    assert wr["active_path_contract"] == "weekly_ml_baseline + scenario_recompute"


def test_neutral_scenario_close_to_baseline():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(
        bundle,
        manual_price=base_price,
        scenario_calc_mode="enhanced_local_factors",
        freight_multiplier=1.0,
        demand_multiplier=1.0,
        overrides={
            "promotion": float(bundle["base_ctx"].get("promotion", 0.0)),
            "discount": float(bundle["base_ctx"].get("discount", 0.0)),
            "shocks": [],
        },
    )
    baseline_units = float(res["neutral_baseline_forecast"]["actual_sales"].sum())
    assert abs(float(wr["demand_total"]) - baseline_units) / max(abs(baseline_units), 1e-9) <= 0.02


def test_price_direction_sanity():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    base = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="enhanced_local_factors")
    up = run_what_if_projection(bundle, manual_price=base_price * 1.10, scenario_calc_mode="enhanced_local_factors")
    down = run_what_if_projection(bundle, manual_price=base_price * 0.90, scenario_calc_mode="enhanced_local_factors")
    up_warn = bool(up.get("ood_flag", False)) or bool(up.get("warnings"))
    down_warn = bool(down.get("ood_flag", False)) or bool(down.get("warnings"))
    if not up_warn:
        assert float(up["demand_total"]) <= float(base["demand_total"])
    if not down_warn:
        assert float(down["demand_total"]) >= float(base["demand_total"])
    assert float(up["revenue_total"]) >= 0.0
    assert float(down["profit_total_adjusted"]) <= float(down["revenue_total"]) + 1e-9


def test_promo_direction_sanity():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    off = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="enhanced_local_factors", overrides={"promotion": 0.0})
    on = run_what_if_projection(bundle, manual_price=base_price, scenario_calc_mode="enhanced_local_factors", overrides={"promotion": 1.0})
    if str(on.get("support_label", "")).lower() in {"high", "medium"} and not on.get("warnings"):
        assert float(on["demand_total"]) >= float(off["demand_total"])
    else:
        assert bool(on.get("warnings")) or str(on.get("support_label", "")).lower() in {"low", "very_low", "unknown"}


def test_shock_units_contract():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    fdates = pd.to_datetime(bundle["future_dates"]["date"])
    start_date = str(fdates.min().date())
    end_date = str(fdates.max().date())
    pos = run_what_if_projection(
        bundle,
        manual_price=base_price,
        scenario_calc_mode="enhanced_local_factors",
        overrides={"shocks": [{"shock_name": "manual_units_pos", "shock_type": "units", "shock_value": 10.0, "start_date": start_date, "end_date": end_date}]},
    )
    neg = run_what_if_projection(
        bundle,
        manual_price=base_price,
        scenario_calc_mode="enhanced_local_factors",
        overrides={"shocks": [{"shock_name": "manual_units_neg", "shock_type": "units", "shock_value": -10.0, "start_date": start_date, "end_date": end_date}]},
    )
    assert float(pos["demand_total"]) >= float(neg["demand_total"])
    assert float(pos["demand_total"]) >= 0.0 and float(neg["demand_total"]) >= 0.0
    daily = pd.DataFrame(pos.get("daily_effects_summary", []))
    assert len(daily) > 0
    assert "shock_units" in daily.columns
    assert pd.to_numeric(daily["shock_units"], errors="coerce").abs().sum() > 0


def test_feature_usage_contract():
    res = _analyze()
    custom_feature_report = pd.DataFrame(
        [
            {"factor_name": "price", "used_in_weekly_model": False, "used_in_scenario": False},
            {"factor_name": "discount", "used_in_weekly_model": False, "used_in_scenario": False},
            {"factor_name": "promotion", "used_in_weekly_model": False, "used_in_scenario": False},
            {"factor_name": "freight_value", "used_in_weekly_model": False, "used_in_scenario": False},
            {"factor_name": "weekday", "used_in_weekly_model": True, "used_in_final_active_forecast": True},
        ]
    )
    res["feature_report_csv"] = custom_feature_report.to_csv(index=False).encode("utf-8")
    excel_blob = build_excel_export_buffer(res, None)
    with pd.ExcelFile(excel_blob) as xls:
        usage = pd.read_excel(xls, "Feature Usage")
        assert not (len(usage) == 1 and str(usage.loc[0, "reason_excluded"]) == "feature_column_missing")
        scenario_rows = usage[usage["group"] == "Scenario-only factors"]["feature"].astype(str).tolist()
        assert {"price", "discount", "promotion", "freight_value"}.issubset(set(scenario_rows))
        baseline_rows = usage[usage["group"] == "Active baseline ML features"]["feature"].astype(str).tolist()
        assert "weekday" in baseline_rows


def test_excel_contains_user_summary():
    res = _analyze()
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(bundle, manual_price=base_price * 1.03, scenario_calc_mode="enhanced_local_factors")
    res["scenario_forecast"] = wr["daily"].copy()
    excel_blob = build_excel_export_buffer(res, wr)
    with pd.ExcelFile(excel_blob) as xls:
        assert "Executive Summary" in xls.sheet_names
        assert "Scenario Summary" in xls.sheet_names
        assert "Feature Usage" in xls.sheet_names
        warnings_df = pd.read_excel(xls, "Executive Summary")
        assert "top_warnings" in warnings_df.columns
