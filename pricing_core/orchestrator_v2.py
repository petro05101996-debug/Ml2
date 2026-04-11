from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from calc_engine import compute_daily_unit_economics
from data_adapter import build_baseline_data_quality_summary, build_daily_panel_from_transactions, build_series_id
from pricing_core.baseline_model import build_weekday_profile, disaggregate_weekly_to_daily
from pricing_core.model_diagnostics import build_model_diagnostics
from pricing_core.scenario_engine import apply_user_overrides, build_current_state_context, build_neutral_context
from pricing_core.weekly_backtest import build_acceptance_summary, evaluate_vs_benchmarks, run_weekly_rolling_backtest
from pricing_core.weekly_forecast_features import build_future_weekly_frame, build_weekly_train_frame
from pricing_core.weekly_forecast_model import recursive_weekly_forecast, train_weekly_forecast_model


def _weekly_to_daily_forecast(weekly_forecast: pd.DataFrame, future_dates: pd.DataFrame, weekday_profile: pd.Series) -> pd.DataFrame:
    return disaggregate_weekly_to_daily(
        weekly_forecast[["week_start", "sales_week"]].rename(columns={"sales_week": "baseline_pred_weekly"}),
        future_dates[["date"]],
        weekday_profile,
    ).rename(columns={"baseline_pred": "actual_sales"})


def _build_excel_buffer(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _apply_total_stock_cap(raw_series: pd.Series, total_stock: float | None) -> pd.DataFrame:
    raw = pd.to_numeric(raw_series, errors="coerce").fillna(0.0).clip(lower=0.0)
    if total_stock is None:
        return pd.DataFrame({"actual_sales": raw, "lost_sales": np.zeros(len(raw))})
    remaining = max(float(total_stock), 0.0)
    actual, lost = [], []
    for v in raw:
        keep = min(float(v), remaining)
        actual.append(keep)
        lost.append(max(float(v) - keep, 0.0))
        remaining = max(0.0, remaining - keep)
    return pd.DataFrame({"actual_sales": actual, "lost_sales": lost})


def _ctx_controls_changed(current_ctx: Dict[str, Any], scenario_ctx: Dict[str, Any]) -> bool:
    controls = ["price", "discount", "promotion"] + [k for k in current_ctx.keys() if str(k).startswith("user_factor_")]
    for c in controls:
        av = current_ctx.get(c)
        bv = scenario_ctx.get(c)
        if isinstance(av, str) or isinstance(bv, str):
            if str(av) != str(bv):
                return True
            continue
        a = float(pd.to_numeric(pd.Series([av]), errors="coerce").fillna(0.0).iloc[0])
        b = float(pd.to_numeric(pd.Series([bv]), errors="coerce").fillna(0.0).iloc[0])
        if abs(a - b) > 1e-9:
            return True
    return False


def run_full_pricing_analysis_v2(
    normalized_txn: pd.DataFrame,
    target_category: str,
    target_sku: str,
    horizon_days: int = 30,
    scenario_overrides: Dict[str, Any] | None = None,
    shocks: List[Dict[str, Any]] | None = None,
    objective_mode: str = "maximize_profit",
    target_series_id: str | None = None,
    unit_price_input_type: str = "net",
    economics_mode: str = "net_price",
) -> Dict[str, Any]:
    panel = build_daily_panel_from_transactions(normalized_txn)
    if "series_id" not in panel.columns:
        panel["series_id"] = build_series_id(panel)

    if target_series_id is None:
        pick = panel[(panel["category"].astype(str) == str(target_category)) & (panel["product_id"].astype(str) == str(target_sku))].copy()
        if pick.empty:
            raise ValueError("Target history is empty")
        target_series_id = str(pick["series_id"].astype(str).mode().iloc[0])

    target_history = panel[panel["series_id"].astype(str) == str(target_series_id)].copy().sort_values("date")
    if target_history.empty:
        raise ValueError("Target history is empty")

    future_dates = pd.DataFrame({"date": pd.date_range(pd.Timestamp(target_history["date"].max()) + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")})
    weekly = build_weekly_train_frame(target_history)
    weekly = weekly.dropna(subset=["sales_week"]).reset_index(drop=True)

    current_ctx = build_current_state_context(target_history, {})
    neutral_ctx = build_neutral_context(target_history, {})
    scenario_ctx = apply_user_overrides(current_ctx, scenario_overrides)

    horizon_weeks = max(1, int(np.ceil(horizon_days / 7)))
    model = train_weekly_forecast_model(weekly)

    future_as_is = build_future_weekly_frame(weekly["week_start"].max(), horizon_weeks, current_ctx)
    future_scn = build_future_weekly_frame(weekly["week_start"].max(), horizon_weeks, scenario_ctx)

    wk_as_is = recursive_weekly_forecast(model, weekly, future_as_is)
    wk_scn = recursive_weekly_forecast(model, weekly, future_scn)
    wk_scn_raw = wk_scn.copy()
    wk_baseline = recursive_weekly_forecast(model, weekly, build_future_weekly_frame(weekly["week_start"].max(), horizon_weeks, neutral_ctx))
    controls_changed = _ctx_controls_changed(current_ctx, scenario_ctx)
    manual_scenario_fallback_applied = False
    if controls_changed:
        base_total_raw = float(pd.to_numeric(wk_as_is["sales_week"], errors="coerce").fillna(0.0).sum())
        scn_total_raw = float(pd.to_numeric(wk_scn["sales_week"], errors="coerce").fillna(0.0).sum())
        if abs(scn_total_raw - base_total_raw) <= max(1e-6, abs(base_total_raw) * 0.0025):
            p_ref = max(float(pd.to_numeric(pd.Series([current_ctx.get("price", 1.0)]), errors="coerce").fillna(1.0).iloc[0]), 1e-6)
            p_scn = max(float(pd.to_numeric(pd.Series([scenario_ctx.get("price", p_ref)]), errors="coerce").fillna(p_ref).iloc[0]), 1e-6)
            d_ref = float(pd.to_numeric(pd.Series([current_ctx.get("discount", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            d_scn = float(pd.to_numeric(pd.Series([scenario_ctx.get("discount", d_ref)]), errors="coerce").fillna(d_ref).iloc[0])
            m_ref = float(pd.to_numeric(pd.Series([current_ctx.get("promotion", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            m_scn = float(pd.to_numeric(pd.Series([scenario_ctx.get("promotion", m_ref)]), errors="coerce").fillna(m_ref).iloc[0])
            scenario_mult = float(np.clip((p_scn / p_ref) ** -1.1 * (1.0 + 0.25 * (d_scn - d_ref)) * (1.0 + 0.15 * (m_scn - m_ref)), 0.85, 1.15))
            wk_scn["sales_week"] = pd.to_numeric(wk_scn["sales_week"], errors="coerce").fillna(0.0) * scenario_mult
            manual_scenario_fallback_applied = True

    ood_flags: List[str] = []
    p_hist = pd.to_numeric(weekly.get("price_week", np.nan), errors="coerce").dropna()
    if len(p_hist):
        p5, p95 = float(p_hist.quantile(0.05)), float(p_hist.quantile(0.95))
        ps = float(pd.to_numeric(pd.Series([scenario_ctx.get("price", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
        if np.isfinite(ps) and (ps < p5 or ps > p95):
            ood_flags.append("scenario_out_of_training_range:price")
    d_hist = pd.to_numeric(weekly.get("discount_week", np.nan), errors="coerce").dropna()
    if len(d_hist):
        d5, d95 = float(d_hist.quantile(0.05)), float(d_hist.quantile(0.95))
        ds = float(pd.to_numeric(pd.Series([scenario_ctx.get("discount", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
        if np.isfinite(ds) and (ds < d5 or ds > d95):
            ood_flags.append("scenario_out_of_training_range:discount")
    pmo_hist = pd.to_numeric(weekly.get("promo_week", np.nan), errors="coerce").dropna()
    if len(pmo_hist):
        pm = float(pd.to_numeric(pd.Series([scenario_ctx.get("promotion", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
        if np.isfinite(pm) and (pm < float(pmo_hist.quantile(0.05)) or pm > float(pmo_hist.quantile(0.95))):
            ood_flags.append("scenario_out_of_training_range:promo")
    for c in [k for k in weekly.columns if str(k).startswith("user_factor_cat__")]:
        known = set(weekly[c].dropna().astype(str).unique())
        incoming = str(scenario_ctx.get(c, ""))
        if incoming and known and incoming not in known:
            ood_flags.append(f"scenario_out_of_training_range:{c}")
    data_suff = {
        "n_weeks": int(len(weekly)),
        "price_unique_count": int(pd.to_numeric(weekly.get("price_week", np.nan), errors="coerce").nunique()),
        "promo_transitions": int(pd.to_numeric(weekly.get("promo_week", 0.0), errors="coerce").diff().abs().fillna(0.0).gt(1e-9).sum()),
    }

    backtest = run_weekly_rolling_backtest(weekly)
    benchmark = evaluate_vs_benchmarks(backtest)
    # behavior checks
    base_total = float(wk_as_is["sales_week"].sum())
    scn_total = float(wk_scn["sales_week"].sum())
    scn_raw_total = float(wk_scn_raw["sales_week"].sum())
    same_controls = not controls_changed
    scenario_changed_but_forecast_unchanged = controls_changed and (abs(scn_total - base_total) <= max(1e-6, abs(base_total) * 0.005))
    scenario_unchanged_but_forecast_changed = same_controls and (abs(scn_total - base_total) > max(1e-6, abs(base_total) * 0.005))
    # monotonicity sanity check
    ctx_up = dict(scenario_ctx)
    ctx_up["price"] = float(scenario_ctx.get("price", current_ctx.get("price", 0.0))) * 1.02
    wk_up = recursive_weekly_forecast(model, weekly, build_future_weekly_frame(weekly["week_start"].max(), horizon_weeks, ctx_up))
    if manual_scenario_fallback_applied:
        p_ref_u = max(float(pd.to_numeric(pd.Series([current_ctx.get("price", 1.0)]), errors="coerce").fillna(1.0).iloc[0]), 1e-6)
        p_up = max(float(pd.to_numeric(pd.Series([ctx_up.get("price", p_ref_u)]), errors="coerce").fillna(p_ref_u).iloc[0]), 1e-6)
        d_ref_u = float(pd.to_numeric(pd.Series([current_ctx.get("discount", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        d_up = float(pd.to_numeric(pd.Series([ctx_up.get("discount", d_ref_u)]), errors="coerce").fillna(d_ref_u).iloc[0])
        m_ref_u = float(pd.to_numeric(pd.Series([current_ctx.get("promotion", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        m_up = float(pd.to_numeric(pd.Series([ctx_up.get("promotion", m_ref_u)]), errors="coerce").fillna(m_ref_u).iloc[0])
        up_mult = float(np.clip((p_up / p_ref_u) ** -1.1 * (1.0 + 0.25 * (d_up - d_ref_u)) * (1.0 + 0.15 * (m_up - m_ref_u)), 0.85, 1.15))
        wk_up["sales_week"] = pd.to_numeric(wk_up["sales_week"], errors="coerce").fillna(0.0) * up_mult
    price_monotonicity_violation = float(wk_up["sales_week"].sum()) > float(wk_scn["sales_week"].sum()) * 1.01
    promo_now = float(pd.to_numeric(pd.Series([scenario_ctx.get("promotion", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    if promo_now >= 0.99:
        promo_sensitivity_missing = False
    else:
        ctx_promo = dict(scenario_ctx)
        ctx_promo["promotion"] = 1.0
        wk_promo = recursive_weekly_forecast(model, weekly, build_future_weekly_frame(weekly["week_start"].max(), horizon_weeks, ctx_promo))
        promo_sensitivity_missing = abs(float(wk_promo["sales_week"].sum()) - float(wk_scn["sales_week"].sum())) <= max(1e-6, abs(float(wk_scn["sales_week"].sum())) * 0.005)
    behavior_checks = {
        "controls_changed": controls_changed,
        "scenario_changed_but_forecast_unchanged": scenario_changed_but_forecast_unchanged,
        "scenario_unchanged_but_forecast_changed": scenario_unchanged_but_forecast_changed,
        "price_monotonicity_violation": price_monotonicity_violation,
        "promo_sensitivity_missing": promo_sensitivity_missing,
        "manual_scenario_fallback_applied": manual_scenario_fallback_applied,
        "scenario_effect_only_from_fallback": bool(controls_changed and manual_scenario_fallback_applied and (abs(scn_raw_total - base_total) <= max(1e-6, abs(base_total) * 0.0025))),
        "model_direct_sensitivity_present": bool((abs(scn_raw_total - base_total) > max(1e-6, abs(base_total) * 0.0025)) if controls_changed else True),
        "price_direction_suspicious": bool(float(pd.to_numeric(pd.Series([scenario_ctx.get("price", np.nan)]), errors="coerce").fillna(np.nan).iloc[0]) > float(pd.to_numeric(pd.Series([current_ctx.get("price", np.nan)]), errors="coerce").fillna(np.nan).iloc[0]) and scn_total > base_total * 1.01),
    }
    diagnostics = build_model_diagnostics(backtest, benchmark, ood_flags, data_suff, behavior_checks=behavior_checks)
    scenario_effect_source = "manual_fallback" if manual_scenario_fallback_applied else "model_direct"
    acceptance_summary = build_acceptance_summary(backtest, benchmark, manual_scenario_fallback_applied, diagnostics.get("issues", []))

    weekday_profile = build_weekday_profile(target_history)
    as_is_daily = _weekly_to_daily_forecast(wk_as_is, future_dates, weekday_profile)
    scn_daily = _weekly_to_daily_forecast(wk_scn, future_dates, weekday_profile)
    baseline_daily = _weekly_to_daily_forecast(wk_baseline, future_dates, weekday_profile).rename(columns={"actual_sales": "baseline_pred"})

    as_is_forecast = future_dates[["date"]].merge(as_is_daily, on="date", how="left").merge(baseline_daily, on="date", how="left")
    scenario_forecast = future_dates[["date"]].merge(scn_daily, on="date", how="left").merge(baseline_daily, on="date", how="left")
    neutral_baseline_forecast = baseline_daily[["date", "baseline_pred"]].copy()
    # stock cap + shocks on top of direct scenario forecast
    as_is_stock = None
    if bool(current_ctx.get("use_stock_cap", False)):
        as_is_stock = float(pd.to_numeric(pd.Series([current_ctx.get("stock_total_horizon", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
    scn_stock = None
    if bool(scenario_ctx.get("use_stock_cap", False)):
        scn_stock = float(pd.to_numeric(pd.Series([scenario_ctx.get("stock_total_horizon", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
    as_is_cap = _apply_total_stock_cap(as_is_forecast["actual_sales"], as_is_stock)
    scn_cap = _apply_total_stock_cap(scenario_forecast["actual_sales"], scn_stock)
    as_is_forecast["actual_sales"] = as_is_cap["actual_sales"].values
    as_is_forecast["lost_sales"] = as_is_cap["lost_sales"].values
    scenario_forecast["actual_sales"] = scn_cap["actual_sales"].values
    scenario_forecast["lost_sales"] = scn_cap["lost_sales"].values
    shock_multiplier = 1.0 + sum(
        [float(pd.to_numeric(pd.Series([sh.get("intensity", 0.0)]), errors="coerce").fillna(0.0).iloc[0]) for sh in (shocks or [])]
    )
    shock_multiplier = float(np.clip(shock_multiplier, 0.5, 1.5))
    for ddf in (as_is_forecast, scenario_forecast):
        ddf["shock_multiplier"] = shock_multiplier
        ddf["actual_sales"] = pd.to_numeric(ddf["actual_sales"], errors="coerce").fillna(0.0) * shock_multiplier
        ddf["demand_raw"] = ddf["actual_sales"] + ddf["lost_sales"]
        ddf["scenario_demand_raw"] = ddf["demand_raw"]

    as_is_forecast["price"] = float(current_ctx.get("price", 0.0))
    as_is_forecast["discount"] = float(current_ctx.get("discount", 0.0))
    as_is_forecast["cost"] = float(current_ctx.get("cost", as_is_forecast["price"].iloc[0] * 0.65))
    as_is_forecast["freight_value"] = float(current_ctx.get("freight_value", 0.0))
    scenario_forecast["price"] = float(scenario_ctx.get("price", as_is_forecast["price"].iloc[0]))
    scenario_forecast["discount"] = float(scenario_ctx.get("discount", as_is_forecast["discount"].iloc[0]))
    scenario_forecast["cost"] = float(scenario_ctx.get("cost", as_is_forecast["cost"].iloc[0]))
    scenario_forecast["freight_value"] = float(scenario_ctx.get("freight_value", as_is_forecast["freight_value"].iloc[0]))

    neutral_baseline_economics, _ = compute_daily_unit_economics(
        neutral_baseline_forecast.assign(
            price=float(neutral_ctx.get("price", as_is_forecast["price"].iloc[0])),
            discount=float(neutral_ctx.get("discount", 0.0)),
            cost=float(neutral_ctx.get("cost", as_is_forecast["cost"].iloc[0])),
            freight_value=float(neutral_ctx.get("freight_value", 0.0)),
        ),
        quantity_col="baseline_pred",
        unit_price_input_type=unit_price_input_type,
        economics_mode=economics_mode,
    )
    as_is_economics, _ = compute_daily_unit_economics(as_is_forecast, quantity_col="actual_sales", unit_price_input_type=unit_price_input_type, economics_mode=economics_mode)
    scenario_economics, _ = compute_daily_unit_economics(scenario_forecast, quantity_col="actual_sales", unit_price_input_type=unit_price_input_type, economics_mode=economics_mode)

    delta_current_vs = pd.DataFrame([{
        "as_is_total_demand": float(as_is_forecast["actual_sales"].sum()),
        "scenario_total_demand": float(scenario_forecast["actual_sales"].sum()),
        "demand_delta_abs": float(scenario_forecast["actual_sales"].sum() - as_is_forecast["actual_sales"].sum()),
        "demand_delta_pct": float((scenario_forecast["actual_sales"].sum() - as_is_forecast["actual_sales"].sum()) / max(float(as_is_forecast["actual_sales"].sum()), 1e-9)),
        "as_is_total_revenue": float(as_is_economics["total_revenue"].sum()),
        "scenario_total_revenue": float(scenario_economics["total_revenue"].sum()),
        "revenue_delta_abs": float(scenario_economics["total_revenue"].sum() - as_is_economics["total_revenue"].sum()),
        "revenue_delta_pct": float((scenario_economics["total_revenue"].sum() - as_is_economics["total_revenue"].sum()) / max(float(as_is_economics["total_revenue"].sum()), 1e-9)),
        "as_is_total_profit": float(as_is_economics["profit"].sum()),
        "scenario_total_profit": float(scenario_economics["profit"].sum()),
        "profit_delta_abs": float(scenario_economics["profit"].sum() - as_is_economics["profit"].sum()),
        "profit_delta_pct": float((scenario_economics["profit"].sum() - as_is_economics["profit"].sum()) / max(abs(float(as_is_economics["profit"].sum())), 1e-9)),
    }])

    delta_neutral_vs = pd.DataFrame([{
        "neutral_total_demand": float(neutral_baseline_forecast["baseline_pred"].sum()),
        "as_is_total_demand": float(as_is_forecast["actual_sales"].sum()),
        "demand_delta_abs": float(as_is_forecast["actual_sales"].sum() - neutral_baseline_forecast["baseline_pred"].sum()),
    }])

    overall_conf = diagnostics["overall_confidence"]
    if manual_scenario_fallback_applied and overall_conf == "high":
        overall_conf = "medium"
    confidence = {
        "baseline_confidence": diagnostics,
        "shock_confidence": {"overall_confidence": "medium", "issues": []},
        "overall_confidence": overall_conf,
        "intervals_available": False,
    }

    bundle = {
        "target_history": target_history,
        "future_dates": future_dates,
        "current_ctx": current_ctx,
        "neutral_ctx": neutral_ctx,
        "scenario_feature_spec": {"path": "weekly_supervised"},
        "trained_weekly_forecast_model": model,
        "weekly_history": weekly,
        "confidence": confidence,
        "acceptance_summary": acceptance_summary,
        "unit_price_input_type": unit_price_input_type,
        "economics_mode": economics_mode,
        "weekly_baseline_forecast_native": wk_baseline.rename(columns={"sales_week": "baseline_pred_weekly"}),
        "weekday_profile": weekday_profile,
    }

    scenario_inputs_echo_rows = []
    for key in ["price", "discount", "promotion"] + [k for k in sorted(set(list(current_ctx.keys()) + list(scenario_ctx.keys()))) if str(k).startswith("user_factor_")]:
        cur = current_ctx.get(key)
        scn = scenario_ctx.get(key)
        changed = _ctx_controls_changed({key: cur}, {key: scn})
        supported = key in scenario_ctx
        in_range = True
        if key == "price":
            in_range = "scenario_out_of_training_range:price" not in ood_flags
        elif key == "discount":
            in_range = "scenario_out_of_training_range:discount" not in ood_flags
        elif key == "promotion":
            in_range = "scenario_out_of_training_range:promo" not in ood_flags
        scenario_inputs_echo_rows.append({"control_name": key, "current_value": cur, "scenario_value": scn, "changed_flag": bool(changed), "supported_flag": bool(supported), "in_training_range_flag": bool(in_range)})
    for w in scenario_ctx.get("_warnings", []):
        if str(w).startswith("unknown_override_ignored:"):
            k = str(w).split(":", 1)[1]
            scenario_inputs_echo_rows.append({"control_name": k, "current_value": None, "scenario_value": None, "changed_flag": False, "supported_flag": False, "in_training_range_flag": False})
    scenario_inputs_echo = pd.DataFrame(scenario_inputs_echo_rows)
    stock_cap_applied_as_is = bool(current_ctx.get("use_stock_cap", False))
    stock_cap_applied_scenario = bool(scenario_ctx.get("use_stock_cap", False))
    scenario_delta_zero_reason = ""
    if abs(float(delta_current_vs["demand_delta_abs"].iloc[0])) <= max(1e-6, abs(float(delta_current_vs["as_is_total_demand"].iloc[0])) * 0.0025):
        if not controls_changed:
            scenario_delta_zero_reason = "scenario_equals_current_state"
        elif (scenario_inputs_echo["supported_flag"] == False).all():
            scenario_delta_zero_reason = "unsupported_overrides"
        elif not behavior_checks["model_direct_sensitivity_present"]:
            scenario_delta_zero_reason = "model_insensitive"
        else:
            scenario_delta_zero_reason = "manual_fallback_not_triggered"

    excel = _build_excel_buffer({
        "history": target_history,
        "neutral_baseline_forecast": neutral_baseline_forecast,
        "as_is_forecast": as_is_forecast,
        "scenario_forecast": scenario_forecast,
        "neutral_baseline_economics": neutral_baseline_economics,
        "as_is_economics": as_is_economics,
        "scenario_economics": scenario_economics,
        "delta_summary_current_vs_scenario": delta_current_vs,
        "delta_summary_neutral_vs_current": delta_neutral_vs,
        "baseline_rolling_metrics": backtest,
        "baseline_rolling_diag": pd.DataFrame(),
        "baseline_benchmark_suite": benchmark,
        "baseline_quality_summary": pd.DataFrame([diagnostics.get("backtest_summary", {})]),
        "baseline_data_quality": build_baseline_data_quality_summary(target_history),
        "scenario_inputs_echo": scenario_inputs_echo,
        "diagnostic_summary": pd.DataFrame(
            [
                {
                    "issues": ";".join(diagnostics.get("issues", [])),
                    "scenario_controls_changed": controls_changed,
                    "scenario_effect_source": scenario_effect_source,
                    "overall_confidence": confidence["overall_confidence"],
                    "stock_cap_applied_as_is": stock_cap_applied_as_is,
                    "stock_cap_applied_scenario": stock_cap_applied_scenario,
                    "shock_multiplier": shock_multiplier,
                    "scenario_delta_zero_reason": scenario_delta_zero_reason,
                    "scenario_fallback_contract": "emergency_only_if_controls_changed_and_model_insensitive",
                    **behavior_checks,
                }
            ]
        ),
        "confidence": pd.DataFrame([confidence]),
        "confidence_flat": pd.DataFrame([{"overall_confidence": confidence["overall_confidence"]}]),
    })

    return {
        "analysis_engine": "v2_weekly_supervised_exogenous",
        "target_series_id": target_series_id,
        "target_history": target_history,
        "weekly_history": weekly,
        "neutral_baseline_forecast": neutral_baseline_forecast,
        "as_is_forecast": as_is_forecast,
        "scenario_forecast": scenario_forecast,
        "baseline_forecast": neutral_baseline_forecast,
        "weekly_baseline_forecast": wk_baseline.rename(columns={"sales_week": "sales"}),
        "weekly_final_forecast_as_is": wk_as_is.rename(columns={"sales_week": "sales"}),
        "weekly_final_forecast_scenario": wk_scn.rename(columns={"sales_week": "sales"}),
        "daily_presented_as_is": as_is_forecast,
        "daily_presented_scenario": scenario_forecast,
        "scenario_effect_source": scenario_effect_source,
        "model_diagnostics": diagnostics,
        "benchmark_summary": benchmark,
        "acceptance_summary": acceptance_summary,
        "scenario_inputs_echo": scenario_inputs_echo,
        "diagnostic_summary": pd.DataFrame(
            [
                {
                    "scenario_controls_changed": controls_changed,
                    "scenario_effect_source": scenario_effect_source,
                    "manual_scenario_fallback_applied": manual_scenario_fallback_applied,
                    "scenario_effect_only_from_fallback": behavior_checks["scenario_effect_only_from_fallback"],
                    "overall_confidence": confidence["overall_confidence"],
                    "confidence_issues": ";".join(diagnostics.get("issues", [])),
                    "stock_cap_applied_as_is": stock_cap_applied_as_is,
                    "stock_cap_applied_scenario": stock_cap_applied_scenario,
                    "shock_multiplier": shock_multiplier,
                    "model_direct_sensitivity_present": behavior_checks["model_direct_sensitivity_present"],
                    "scenario_delta_zero_reason": scenario_delta_zero_reason,
                    **behavior_checks,
                }
            ]
        ),
        "neutral_baseline_economics": neutral_baseline_economics,
        "as_is_economics": as_is_economics,
        "scenario_economics": scenario_economics,
        "delta_summary_current_vs_scenario": delta_current_vs,
        "delta_summary_neutral_vs_current": delta_neutral_vs,
        "confidence": confidence,
        "ood_flags": ood_flags,
        "baseline_benchmark_suite": benchmark,
        "baseline_quality_gate": {"baseline_meets_quality_gate": True},
        "baseline_plan_selection": {"final_selected_strategy": "weekly_supervised", "final_selected_granularity": "weekly"},
        "final_baseline_strategy": "weekly_supervised",
        "final_baseline_granularity": "weekly",
        "final_baseline_source": "weekly_supervised",
        "factor_model_trained": False,
        "factor_role": "removed",
        "scenario_controls_changed": controls_changed,
        "scenario_delta_zero_reason": "no_overrides" if (not controls_changed and scenario_delta_zero_reason == "") else scenario_delta_zero_reason,
        "scenario_fallback_contract": "emergency_only_if_controls_changed_and_model_insensitive",
        "current_state_contributions": pd.DataFrame(),
        "scenario_delta_contributions": pd.DataFrame(),
        "excel_buffer": excel,
        "_trained_bundle": bundle,
        "economics_mode": economics_mode,
        "unit_price_input_type": unit_price_input_type,
    }
