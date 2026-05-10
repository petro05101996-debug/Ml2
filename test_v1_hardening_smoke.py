import io

import pandas as pd


def _synthetic_txn(n_days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i, date in enumerate(dates):
        price = 100.0 + float(i % 8)
        promo = 1.0 if i % 21 == 0 else 0.0
        freight = 4.0 + float(i % 3)
        quantity = max(1.0, 42.0 - 0.12 * price + 5.0 * promo - 0.25 * freight)
        rows.append(
            {
                "date": date,
                "product_id": "sku-hardening",
                "category": "cat-hardening",
                "price": price,
                "quantity": quantity,
                "revenue": quantity * price * (1.0 - 0.05 * promo),
                "cost": price * 0.65,
                "discount": 0.05 * promo,
                "freight_value": freight,
                "promotion": promo,
                "stock": quantity + 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_enhanced_e2e_analysis_what_if_report_and_excel_smoke():
    from core_analysis import run_full_pricing_analysis_universal
    from what_if_runner import run_what_if_projection
    from report_builder import build_business_report_payload
    from export_builder import build_excel_export_buffer

    res = run_full_pricing_analysis_universal(
        _synthetic_txn(),
        "cat-hardening",
        "sku-hardening",
        scenario_calc_mode="enhanced_local_factors",
        horizon_days=30,
    )
    bundle = res["_trained_bundle"]
    base_price = float(bundle["base_ctx"]["price"])
    wr = run_what_if_projection(
        bundle,
        manual_price=base_price * 1.03,
        scenario_calc_mode="enhanced_local_factors",
        horizon_days=30,
    )

    assert len(wr["daily"]) > 0
    assert wr["demand_total"] > 0
    assert wr["revenue_total"] > 0
    assert "profit_total" in wr
    assert wr.get("scenario_run_id")
    assert wr.get("scenario_reproducibility", {}).get("full_dataset_sha256")
    assert wr.get("scenario_reproducibility", {}).get("rows_count") == len(bundle["daily_base"])

    payload = build_business_report_payload(res, wr, {})
    assert payload["trust_block"]["recommended_mode_status"] == res.get("recommended_mode_status", {})
    assert "scenario_reproducibility" in payload["trust_block"]

    excel_buffer = build_excel_export_buffer(res, wr)
    assert isinstance(excel_buffer, io.BytesIO)
    assert excel_buffer.getbuffer().nbytes > 0
    with pd.ExcelFile(excel_buffer) as xls:
        for sheet in ["Decision Gate", "Scenario Audit", "Model Quality", "Feature Usage", "Limitations"]:
            assert sheet in xls.sheet_names


def test_catboost_unavailable_dispatcher_falls_back_to_enhanced(monkeypatch):
    import app

    def fake_train(*args, **kwargs):
        return {
            "enabled": False,
            "reason": "catboost_unavailable",
            "warnings": ["CatBoost недоступен"],
            "feature_report": pd.DataFrame(),
            "factor_catalog": pd.DataFrame(),
        }

    monkeypatch.setattr(app, "train_catboost_full_factor_bundle", fake_train)
    result = app.run_full_pricing_analysis_universal(
        _synthetic_txn(),
        "cat-hardening",
        "sku-hardening",
        scenario_calc_mode="catboost_full_factors",
        horizon_days=30,
    )

    assert result.get("blocking_error", False) is False
    assert result.get("analysis_scenario_calc_mode") == "enhanced_local_factors"
    assert result.get("catboost_full_factor_attempt", {}).get("enabled") is False
    assert result.get("recommended_mode_status", {}).get("recommended_mode") == "enhanced_local_factors"


def test_scenario_audit_run_id_is_stable_and_sensitive_to_semantics():
    from scenario_audit import build_scenario_reproducibility_id

    df = _synthetic_txn(20)
    bundle_a = {"daily_base": df}
    bundle_b = {"daily_base": df[list(reversed(df.columns))].copy()}
    base_params = {"manual_price": 101.0, "horizon_days": 30}

    run_a = build_scenario_reproducibility_id(bundle_a, base_params, "enhanced_local_factors", "safe_clip", "v1", "code")
    run_b = build_scenario_reproducibility_id(bundle_b, base_params, "enhanced_local_factors", "safe_clip", "v1", "code")
    run_price = build_scenario_reproducibility_id(bundle_a, {**base_params, "manual_price": 102.0}, "enhanced_local_factors", "safe_clip", "v1", "code")
    run_mode = build_scenario_reproducibility_id(bundle_a, base_params, "legacy_current", "safe_clip", "v1", "code")

    assert run_a["scenario_run_id"] == run_b["scenario_run_id"]
    assert run_a["dataset_hash"] == run_b["dataset_hash"]
    assert run_a["scenario_run_id"] != run_price["scenario_run_id"]
    assert run_a["scenario_run_id"] != run_mode["scenario_run_id"]
