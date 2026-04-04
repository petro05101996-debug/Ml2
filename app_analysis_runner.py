from __future__ import annotations

from typing import Any, Dict

from pricing_core import run_full_pricing_analysis_universal_v1


UNIVERSAL_LOAD_MODES = {"Universal CSV", "Универсальный CSV"}


def run_analysis_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    load_mode = ctx.get("load_mode", "Universal CSV")
    target_category = ctx["target_category"]
    target_sku = ctx["target_sku"]
    horizon_days = int(ctx.get("forecast_horizon_days", 30))

    caution_to_risk_lambda = {"Низкий": 0.45, "Средний": 0.7, "Высокий": 1.0}
    risk_lambda = float(caution_to_risk_lambda.get(ctx.get("caution_level", "Средний"), 0.7))

    result = run_full_pricing_analysis_universal_v1(
        ctx["universal_txn"],
        target_category,
        target_sku,
        objective_mode=ctx.get("objective_mode", "maximize_profit"),
        horizon_days=horizon_days,
        risk_lambda=risk_lambda,
        analysis_route="runner_to_v1_universal",
        ui_load_mode=str(load_mode),
    )
    if result.get("analysis_engine") != "v1_universal":
        raise RuntimeError("Expected v1_universal engine but got a different analysis_engine marker.")
    result["analysis_route"] = "runner_to_v1_universal"
    result["ui_load_mode"] = str(load_mode if load_mode in UNIVERSAL_LOAD_MODES else "Universal CSV")
    return result
