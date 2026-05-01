import app
import pandas as pd
import json


def test_multiplier_pct_roundtrip():
    assert app.pct_to_multiplier(10.0) == 1.1
    assert abs(app.multiplier_to_pct(0.9) + 10.0) < 1e-9


def test_assessment_potentially_profitable():
    label, _ = app.classify_scenario_assessment(
        profit_delta_pct=5.0,
        warnings_list=[],
        ood_flag=False,
        confidence_label="Высокая",
        shape_quality_low=False,
        support_label="high",
    )
    assert label == "Потенциально выгоден"


def test_assessment_risky_from_ood():
    label, _ = app.classify_scenario_assessment(
        profit_delta_pct=2.0,
        warnings_list=[],
        ood_flag=True,
        confidence_label="Средняя",
        shape_quality_low=False,
        support_label="high",
    )
    assert label == "Нейтральный"


def test_assessment_caution_from_shape_quality():
    label, _ = app.classify_scenario_assessment(
        profit_delta_pct=7.0,
        warnings_list=[],
        ood_flag=False,
        confidence_label="Высокая",
        shape_quality_low=True,
        support_label="high",
    )
    assert label == "Потенциально выгоден"


def test_assessment_risky_even_with_shape_quality_low():
    label, _ = app.classify_scenario_assessment(
        profit_delta_pct=-16.9,
        warnings_list=[],
        ood_flag=False,
        confidence_label="Высокая",
        shape_quality_low=True,
        support_label="high",
    )
    assert label == "Рискован"


def test_formatters_price_and_percent_signs():
    assert app.fmt_price(69.99) == "₽ 69.99"
    assert app.fmt_pct_abs(9.0) == "9.0%"
    assert app.fmt_pct_delta(-16.9) == "-16.9%"


def test_safe_signed_pct_handles_negative_base():
    assert app.safe_signed_pct(-200.0, -1000.0) == -20.0
    assert app.safe_signed_pct(200.0, -1000.0) == 20.0


def test_negative_base_profit_verdict_not_profitable():
    pct = app.safe_signed_pct(-200.0, -1000.0)
    label, _, _ = app.classify_economic_verdict(pct, 5.0, 1.0)
    assert pct == -20.0
    assert label == "Невыгоден"


def test_validation_gate_passes_consistent_totals():
    as_is = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "actual_sales": [10.0, 10.0], "revenue": [90.0, 90.0], "profit": [20.0, 20.0]})
    daily = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "actual_sales": [11.0, 11.0],
            "revenue": [100.0, 120.0],
            "profit": [30.0, 35.0],
            "net_unit_price": [9.090909, 10.909091],
            "price": [11.0, 11.0],
        }
    )
    wr = {
        "daily": daily,
        "demand_total": 22.0,
        "revenue_total": 220.0,
        "profit_total": 65.0,
        "ood_flag": False,
        "clip_applied": False,
        "fallback_multiplier_used": False,
    }
    gate = app.validate_scenario_consistency(as_is, wr, expected_hdays=2)
    assert gate["ok"] is True
    assert gate["errors"] == []


def test_align_forecasts_by_scenario_dates_returns_same_horizon():
    base = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "actual_sales": [10.0, 11.0, 12.0],
            "revenue": [100.0, 110.0, 120.0],
            "profit": [30.0, 31.0, 32.0],
        }
    )
    scenario = pd.DataFrame({"date": ["2026-01-02", "2026-01-03"], "actual_sales": [9.0, 10.0]})
    aligned = app.align_forecasts_by_scenario_dates(base, scenario)
    assert len(aligned) == 2
    assert list(aligned["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-03"]


def test_calculate_scenario_deltas_align_to_scenario_horizon():
    as_is = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "actual_sales": [10.0, 20.0, 999.0],
            "revenue": [100.0, 200.0, 9990.0],
            "profit": [30.0, 60.0, 999.0],
        }
    )
    scenario = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "actual_sales": [11.0, 22.0],
            "revenue": [110.0, 220.0],
            "profit": [33.0, 66.0],
        }
    )
    d = app.calculate_scenario_deltas(as_is, scenario)

    assert d["base_units"] == 30.0
    assert d["scenario_units"] == 33.0
    assert d["delta_units"] == 3.0

    assert d["base_revenue"] == 300.0
    assert d["scenario_revenue"] == 330.0
    assert d["delta_revenue"] == 30.0

    assert d["base_profit"] == 90.0
    assert d["scenario_profit"] == 99.0
    assert d["delta_profit"] == 9.0


def test_build_saved_scenario_metrics_uses_aligned_horizon():
    as_is = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "actual_sales": [10.0, 20.0, 500.0],
            "revenue": [100.0, 200.0, 5000.0],
            "profit": [30.0, 60.0, 500.0],
        }
    )
    scenario = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "actual_sales": [11.0, 22.0],
            "revenue": [110.0, 220.0],
            "profit": [33.0, 66.0],
        }
    )
    metrics = app.build_saved_scenario_metrics(as_is, scenario, {})
    assert metrics["delta_units"] == 3.0
    assert metrics["delta_revenue"] == 30.0
    assert metrics["delta_profit"] == 9.0


def test_economic_and_reliability_verdicts_are_independent():
    economic, _, _ = app.classify_economic_verdict(-16.9, 10.9, 0.95)
    reliability, _, _ = app.classify_reliability_verdict(
        ood_flag=False,
        warnings_list=[],
        confidence_label="Высокая",
        support_label="high",
        shape_quality_low=False,
        validation_ok=True,
    )
    assert economic == "Невыгоден"
    assert reliability == "Высокая"


def test_manual_summary_contains_as_is_and_neutral_totals():
    result_dict = {
        "as_is_forecast": pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "actual_sales": [10.0, 20.0],
                "revenue": [100.0, 200.0],
                "profit": [30.0, 50.0],
            }
        ),
        "neutral_baseline_forecast": pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "actual_sales": [9.0, 18.0],
                "revenue": [90.0, 180.0],
                "profit": [27.0, 45.0],
            }
        ),
        "_trained_bundle": {"base_ctx": {"product_id": "sku"}},
    }
    wr = {
        "daily": pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "actual_sales": [11.0, 19.0],
                "revenue": [110.0, 190.0],
                "profit": [33.0, 47.0],
                "price": [70.0, 70.0],
                "discount": [0.09, 0.09],
                "net_unit_price": [63.7, 63.7],
                "promotion": [0.0, 0.0],
                "freight_value": [1.0, 1.0],
                "cost": [40.0, 40.0],
            }
        ),
        "scenario_status": "computed",
    }
    summary_blob, _ = app.build_manual_scenario_artifacts(result_dict, wr)
    summary = json.loads(summary_blob.decode("utf-8"))
    assert summary["as_is_profit_total"] == 80.0
    assert summary["neutral_baseline_profit_total"] == 72.0


def test_excel_scenario_summary_has_freight_value_and_multiplier():
    result_dict = {
        "history_daily": pd.DataFrame({"date": ["2026-01-01"], "sales": [10.0], "price": [70.0]}),
        "as_is_forecast": pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "actual_sales": [10.0, 20.0],
                "revenue": [100.0, 200.0],
                "profit": [30.0, 50.0],
            }
        ),
        "neutral_baseline_forecast": pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "actual_sales": [9.0, 18.0],
                "revenue": [90.0, 180.0],
                "profit": [27.0, 45.0],
            }
        ),
        "scenario_forecast": pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "actual_sales": [11.0, 19.0],
                "revenue": [110.0, 190.0],
                "profit": [33.0, 47.0],
                "price": [70.0, 70.0],
                "discount": [0.09, 0.09],
                "promotion": [0.0, 0.0],
            }
        ),
        "holdout_metrics": pd.DataFrame(),
        "analysis_run_summary_json": b"{}",
    }
    wr = {
        "daily": result_dict["scenario_forecast"].copy(),
        "applied_path_summary": {"avg_freight": 3.83},
        "effective_scenario": {"freight_multiplier": 1.0},
        "confidence_label": "Высокая",
    }
    excel = app.build_excel_export_buffer(result_dict, wr)
    scenario_summary = pd.read_excel(excel, sheet_name="Scenario Summary")
    assert "avg_freight_value" in scenario_summary.columns
    assert "freight_multiplier" in scenario_summary.columns
    assert "avg_freight_multiplier" not in scenario_summary.columns


def test_excel_has_user_scenario_summary_sheet():
    result_dict = {
        "history_daily": pd.DataFrame({"date": ["2026-01-01"], "sales": [10.0], "price": [70.0]}),
        "as_is_forecast": pd.DataFrame({"date": ["2026-01-01"], "actual_sales": [10.0], "revenue": [100.0], "profit": [30.0]}),
        "neutral_baseline_forecast": pd.DataFrame({"date": ["2026-01-01"], "actual_sales": [9.0], "revenue": [90.0], "profit": [27.0]}),
        "scenario_forecast": pd.DataFrame({"date": ["2026-01-01"], "actual_sales": [11.0], "revenue": [110.0], "profit": [25.0]}),
        "holdout_metrics": pd.DataFrame(),
        "analysis_run_summary_json": b"{}",
    }
    wr = {
        "daily": result_dict["scenario_forecast"].copy(),
        "effective_scenario": {"freight_multiplier": 1.0},
        "applied_path_summary": {"avg_freight": 3.83},
        "confidence_label": "Высокая",
    }
    excel = app.build_excel_export_buffer(result_dict, wr)
    user_summary = pd.read_excel(excel, sheet_name="User Scenario Summary")
    assert "Текущий план: прибыль" in user_summary.columns
    assert "Сценарий: прибыль" in user_summary.columns
    assert "Рекомендация" in user_summary.columns
