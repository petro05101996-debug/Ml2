from __future__ import annotations

from typing import Any, Dict

from pricing_core import build_v2_result_contract, run_full_pricing_analysis_v2


UNIVERSAL_LOAD_MODES = {"Universal CSV", "Универсальный CSV"}


def run_analysis_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    load_mode = ctx.get("load_mode", "Universal CSV")
    target_category = ctx["target_category"]
    target_sku = ctx["target_sku"]
    horizon_days = int(ctx.get("forecast_horizon_days", 30))

    result = run_full_pricing_analysis_v2(
        ctx["universal_txn"],
        target_category,
        target_sku,
        horizon_days=horizon_days,
        scenario_overrides=ctx.get("scenario_overrides"),
        shocks=ctx.get("shocks"),
        objective_mode=str(ctx.get("objective_mode", "maximize_profit")),
    )
    result["analysis_route"] = "runner_to_v2_decomposed"
    result["ui_load_mode"] = str(load_mode if load_mode in UNIVERSAL_LOAD_MODES else "Universal CSV")
    if result.get("analysis_engine") == "v2_decomposed_baseline_factor_shock":
        result["v2_result_contract"] = build_v2_result_contract(result)
    if isinstance(result.get("_trained_bundle"), dict):
        result["_trained_bundle"]["objective_mode"] = str(ctx.get("objective_mode", "maximize_profit"))
    return result
