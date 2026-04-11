from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pricing_core.weekly_forecast_model import recursive_weekly_forecast, train_weekly_forecast_model


def _metrics(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(actual, errors="coerce").fillna(0.0)
    p = pd.to_numeric(pred, errors="coerce").fillna(0.0)
    err = p - a
    denom = max(float(a.abs().sum()), 1e-9)
    return {
        "wape": float(err.abs().sum() / denom),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias_pct": float(err.sum() / denom),
        "sum_ratio": float(p.sum() / max(float(a.sum()), 1e-9)),
        "std_ratio": float(p.std(ddof=0) / max(float(a.std(ddof=0)), 1e-9)),
    }


def _benchmark(history: pd.Series, horizon: int, method: str) -> np.ndarray:
    h = pd.to_numeric(history, errors="coerce").fillna(0.0)
    if method == "seasonal_naive_4":
        vals = h.tail(4).tolist() or [0.0]
        return np.array((vals * ((horizon // len(vals)) + 1))[:horizon], dtype=float)
    if method == "moving_avg_8":
        return np.full(horizon, float(h.tail(8).mean() if len(h) else 0.0), dtype=float)
    return np.full(horizon, float(h.tail(4).mean() if len(h) else 0.0), dtype=float)


def run_weekly_rolling_backtest(weekly_df: pd.DataFrame, initial_window: int = 26, horizon: int = 4, step: int = 4) -> pd.DataFrame:
    df = weekly_df.sort_values("week_start").reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    idx = 0
    for split_end in range(initial_window, len(df) - horizon + 1, step):
        train = df.iloc[:split_end].copy()
        test = df.iloc[split_end : split_end + horizon].copy()
        if len(train) < initial_window or len(test) < horizon:
            continue
        model = train_weekly_forecast_model(train)
        exog = test.drop(columns=["sales_week"], errors="ignore")
        pred = recursive_weekly_forecast(model, train, exog)["sales_week"]
        m = _metrics(test["sales_week"], pred)
        b4 = _metrics(test["sales_week"], pd.Series(_benchmark(train["sales_week"], horizon, "moving_avg_4")))
        b8 = _metrics(test["sales_week"], pd.Series(_benchmark(train["sales_week"], horizon, "moving_avg_8")))
        bs = _metrics(test["sales_week"], pd.Series(_benchmark(train["sales_week"], horizon, "seasonal_naive_4")))
        rows.append(
            {
                "window_id": idx,
                **m,
                "benchmark_wape_ma4": b4["wape"],
                "benchmark_wape_ma8": b8["wape"],
                "benchmark_wape_seasonal4": bs["wape"],
            }
        )
        idx += 1
    return pd.DataFrame(rows)


def evaluate_vs_benchmarks(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "model_median_wape",
                "benchmark_median_wape",
                "model_better_by_pct",
            ]
        )
    model_median = float(pd.to_numeric(backtest_df["wape"], errors="coerce").median())
    mapping = {
        "moving_avg_4": "benchmark_wape_ma4",
        "moving_avg_8": "benchmark_wape_ma8",
        "seasonal_naive_4": "benchmark_wape_seasonal4",
    }
    rows = []
    for name, col in mapping.items():
        bench = float(pd.to_numeric(backtest_df.get(col, np.nan), errors="coerce").median())
        rows.append(
            {
                "benchmark": name,
                "model_median_wape": model_median,
                "benchmark_median_wape": bench,
                "model_better_by_pct": float((bench - model_median) / max(bench, 1e-9)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best_bench = float(pd.to_numeric(out["benchmark_median_wape"], errors="coerce").min())
    model_wape = float(pd.to_numeric(out["model_median_wape"], errors="coerce").median())
    out["model_beats_best_benchmark_median_wape"] = model_wape < best_bench
    out["model_not_worse_than_best_benchmark_by_more_than_10pct"] = model_wape <= best_bench * 1.10
    out["model_bias_ok"] = float(pd.to_numeric(backtest_df.get("bias_pct", np.nan), errors="coerce").abs().median()) <= 0.10
    out["model_std_ratio_ok"] = float(pd.to_numeric(backtest_df.get("std_ratio", np.nan), errors="coerce").median()) >= 0.60
    return out


def build_acceptance_summary(backtest_df: pd.DataFrame, benchmark_summary: pd.DataFrame, manual_fallback_used: bool, issues: list[str]) -> dict:
    n_windows = int(len(backtest_df))
    med_wape = float(pd.to_numeric(backtest_df.get("wape", np.nan), errors="coerce").median()) if n_windows else np.nan
    severe_instability = bool(pd.to_numeric(backtest_df.get("std_ratio", np.nan), errors="coerce").median() < 0.40) if n_windows else True
    scenario_exploration = (n_windows >= 3) and np.isfinite(med_wape) and (med_wape <= 0.65) and (not severe_instability)
    auto_ok = False
    if not benchmark_summary.empty:
        r = benchmark_summary.iloc[0]
        auto_ok = bool(r.get("model_not_worse_than_best_benchmark_by_more_than_10pct", False)) and bool(r.get("model_bias_ok", False)) and bool(r.get("model_std_ratio_ok", False))
    auto_ok = auto_ok and (not manual_fallback_used) and ("scenario_effect_from_manual_fallback" not in set(issues))
    return {
        "accepted_for_scenario_exploration": bool(scenario_exploration),
        "accepted_for_automated_recommendation": bool(auto_ok),
        "n_backtest_windows": n_windows,
        "median_wape": med_wape,
        "manual_fallback_used": bool(manual_fallback_used),
    }
