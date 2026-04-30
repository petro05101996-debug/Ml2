from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import catboost

    CatBoostRegressor = catboost.CatBoostRegressor
    USE_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    USE_CATBOOST = False


CATBOOST_FULL_FACTOR_MODE = "catboost_full_factors"


CORE_FACTOR_COLUMNS = [
    "price",
    "discount",
    "net_unit_price",
    "freight_value",
    "cost",
    "promotion",
    "review_score",
    "reviews_count",
    "stock",
]


CALENDAR_FEATURES = [
    "dow",
    "month",
    "weekofyear",
    "dayofmonth",
    "is_weekend",
]


LAG_FEATURES = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_7",
    "sales_roll_14",
    "sales_roll_28",
    "sales_ewm_14",
]


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["date"], errors="coerce")
    out["dow"] = dt.dt.dayofweek.astype(float)
    out["month"] = dt.dt.month.astype(float)
    out["weekofyear"] = dt.dt.isocalendar().week.astype(float)
    out["dayofmonth"] = dt.dt.day.astype(float)
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
    return out


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    sales = _safe_numeric(out["sales"], 0.0).clip(lower=0.0)

    out["sales_lag_1"] = sales.shift(1)
    out["sales_lag_7"] = sales.shift(7)
    out["sales_lag_14"] = sales.shift(14)
    out["sales_lag_28"] = sales.shift(28)

    out["sales_roll_7"] = sales.shift(1).rolling(7, min_periods=2).mean()
    out["sales_roll_14"] = sales.shift(1).rolling(14, min_periods=3).mean()
    out["sales_roll_28"] = sales.shift(1).rolling(28, min_periods=5).mean()
    out["sales_ewm_14"] = sales.shift(1).ewm(span=14, adjust=False, min_periods=2).mean()

    for col in LAG_FEATURES:
        out[col] = _safe_numeric(out[col], float(sales.median() if len(sales) else 0.0)).clip(lower=0.0)

    return out


def _prepare_base_columns(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in CORE_FACTOR_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["price"] = _safe_numeric(out["price"], 1.0).ffill().bfill().clip(lower=0.01)
    out["discount"] = _safe_numeric(out["discount"], 0.0).ffill().bfill().clip(0.0, 0.95)
    out["net_unit_price"] = _safe_numeric(out["net_unit_price"], np.nan)
    out["net_unit_price"] = out["net_unit_price"].fillna(out["price"] * (1.0 - out["discount"])).clip(lower=0.01)
    out["freight_value"] = _safe_numeric(out["freight_value"], 0.0).ffill().bfill().clip(lower=0.0)
    out["cost"] = _safe_numeric(out["cost"], np.nan).ffill().bfill()
    out["cost"] = out["cost"].fillna(out["price"] * 0.65).clip(lower=0.0)
    out["promotion"] = _safe_numeric(out["promotion"], 0.0).ffill().bfill().clip(0.0, 1.0)
    out["review_score"] = _safe_numeric(out["review_score"], 4.5).ffill().bfill()
    out["reviews_count"] = _safe_numeric(out["reviews_count"], 0.0).ffill().bfill().clip(lower=0.0)
    out["stock"] = _safe_numeric(out["stock"], np.inf).ffill().bfill()

    factor_cols = [c for c in out.columns if str(c).startswith("factor__")]
    for col in factor_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = _safe_numeric(out[col], np.nan).ffill().bfill()
        else:
            out[col] = out[col].astype("object").ffill().bfill().fillna("unknown").astype(str)

    return out


def _build_model_frame(daily: pd.DataFrame) -> pd.DataFrame:
    out = _prepare_base_columns(daily)
    out = _add_calendar_features(out)
    out = _add_lag_features(out)
    out["target_log_sales"] = np.log1p(_safe_numeric(out["sales"], 0.0).clip(lower=0.0))
    return out


def _infer_feature_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
    protected = {
        "date",
        "sales",
        "revenue",
        "target_log_sales",
        "sku_id",
        "product_id",
        "category",
        "region",
        "channel",
        "segment",
    }

    candidates = []
    cat_features = []
    report_rows = []

    for col in frame.columns:
        if col in protected:
            continue

        if col in CORE_FACTOR_COLUMNS or col in CALENDAR_FEATURES or col in LAG_FEATURES or str(col).startswith("factor__"):
            s = frame[col]
            missing_share = float(s.isna().mean()) if len(s) else 1.0
            nunique = int(s.nunique(dropna=True)) if len(s) else 0
            std_val = float(pd.to_numeric(s, errors="coerce").std(ddof=0)) if pd.api.types.is_numeric_dtype(s) and len(s) else float("nan")
            source = "extra_factor" if str(col).startswith("factor__") else ("calendar" if col in CALENDAR_FEATURES else ("lag" if col in LAG_FEATURES else "core"))
            role = "driver"
            dtype_label = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
            raw_column = str(col).replace("factor__", "", 1) if str(col).startswith("factor__") else str(col)

            if missing_share > 0.8 or nunique < 2:
                report_rows.append(
                    {
                        "feature": col,
                        "raw_column": raw_column,
                        "source": source,
                        "dtype": dtype_label,
                        "role": role,
                        "used_in_active_model": False,
                        "reason": "missing_or_constant",
                        "missing_share": missing_share,
                        "unique_count": nunique,
                        "std": std_val,
                    }
                )
                continue

            candidates.append(col)
            is_cat = not pd.api.types.is_numeric_dtype(s)
            if is_cat:
                cat_features.append(col)

            report_rows.append(
                {
                    "feature": col,
                    "raw_column": raw_column,
                    "source": source,
                    "dtype": dtype_label,
                    "role": role,
                    "used_in_active_model": True,
                    "reason": "",
                    "missing_share": missing_share,
                    "unique_count": nunique,
                    "std": std_val,
                }
            )

    out = pd.DataFrame(report_rows)
    if len(out):
        out["importance"] = np.nan
    return candidates, cat_features, out


def train_catboost_full_factor_bundle(
    daily_base: pd.DataFrame,
    future_dates: pd.DataFrame,
    min_train_days: int = 60,
) -> Dict[str, Any]:
    frame = _build_model_frame(daily_base)
    feature_cols, cat_feature_names, feature_report = _infer_feature_columns(frame)

    trainable = frame.dropna(subset=["target_log_sales"]).copy()
    trainable = trainable[trainable["sales"].notna()].copy()

    warnings = []
    factor_catalog = []
    for col in feature_cols:
        s = frame[col] if col in frame.columns else pd.Series(dtype="float64")
        numeric = pd.api.types.is_numeric_dtype(s)
        non_na = s.dropna()
        factor_catalog.append(
            {
                "feature": str(col),
                "raw_column": str(col).replace("factor__", "", 1),
                "dtype": "numeric" if numeric else "categorical",
                "editable": bool(str(col).startswith("factor__") or col in CORE_FACTOR_COLUMNS),
                "current_value": non_na.iloc[-1] if len(non_na) else ("unknown" if not numeric else np.nan),
                "fill_value": np.nan,
                "train_min": float(pd.to_numeric(non_na, errors="coerce").min()) if numeric and len(non_na) else np.nan,
                "train_p10": float(pd.to_numeric(non_na, errors="coerce").quantile(0.10)) if numeric and len(non_na) else np.nan,
                "train_median": float(pd.to_numeric(non_na, errors="coerce").median()) if numeric and len(non_na) else np.nan,
                "train_p90": float(pd.to_numeric(non_na, errors="coerce").quantile(0.90)) if numeric and len(non_na) else np.nan,
                "train_max": float(pd.to_numeric(non_na, errors="coerce").max()) if numeric and len(non_na) else np.nan,
                "missing_share": float(s.isna().mean()) if len(s) else 1.0,
                "nunique": int(s.nunique(dropna=True)) if len(s) else 0,
                "source": "extra_factor" if str(col).startswith("factor__") else ("calendar" if col in CALENDAR_FEATURES else ("lag" if col in LAG_FEATURES else "core")),
            }
        )

    if len(trainable) < min_train_days:
        warnings.append(f"Недостаточно дневной истории для CatBoost full factors: {len(trainable)} < {min_train_days}.")
        return {
            "enabled": False,
            "reason": "not_enough_daily_history",
            "warnings": warnings,
            "feature_cols": feature_cols,
            "cat_feature_names": cat_feature_names,
            "feature_report": feature_report,
            "model_backend": "disabled",
            "factor_catalog": pd.DataFrame(factor_catalog),
        }

    if not USE_CATBOOST or CatBoostRegressor is None:
        warnings.append("CatBoost недоступен в окружении.")
        return {
            "enabled": False,
            "reason": "catboost_unavailable",
            "warnings": warnings,
            "feature_cols": feature_cols,
            "cat_feature_names": cat_feature_names,
            "feature_report": feature_report,
            "model_backend": "disabled",
            "factor_catalog": pd.DataFrame(factor_catalog),
        }

    if len(feature_cols) == 0:
        warnings.append("Нет пригодных факторов для CatBoost full factors.")
        return {
            "enabled": False,
            "reason": "no_usable_features",
            "warnings": warnings,
            "feature_cols": [],
            "cat_feature_names": [],
            "feature_report": feature_report,
            "model_backend": "disabled",
            "factor_catalog": pd.DataFrame(factor_catalog),
        }

    holdout_size = int(max(14, min(56, round(len(trainable) * 0.2))))
    train_part = trainable.iloc[:-holdout_size].copy()
    holdout_part = trainable.iloc[-holdout_size:].copy()

    X_train = train_part[feature_cols].copy()
    y_train = train_part["target_log_sales"].astype(float)
    X_holdout = holdout_part[feature_cols].copy()
    y_holdout_sales = _safe_numeric(holdout_part["sales"], 0.0).clip(lower=0.0)

    for col in feature_cols:
        if col in cat_feature_names:
            X_train[col] = X_train[col].astype(str).fillna("unknown")
            X_holdout[col] = X_holdout[col].astype(str).fillna("unknown")
        else:
            median_val = pd.to_numeric(X_train[col], errors="coerce").median()
            if not np.isfinite(median_val):
                median_val = 0.0
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(median_val)
            X_holdout[col] = pd.to_numeric(X_holdout[col], errors="coerce").fillna(median_val)

    cat_indices = [feature_cols.index(c) for c in cat_feature_names if c in feature_cols]

    model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=8.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        thread_count=1,
    )

    model.fit(X_train, y_train, cat_features=cat_indices)

    train_fill_values = {}
    for col in feature_cols:
        if col in cat_feature_names:
            train_fill_values[col] = "unknown"
        else:
            median_val = pd.to_numeric(X_train[col], errors="coerce").median()
            train_fill_values[col] = float(median_val) if np.isfinite(median_val) else 0.0

    base_cols_for_recursive = ["date", "sales"] + [c for c in frame.columns if c in CORE_FACTOR_COLUMNS or str(c).startswith("factor__")]
    recursive_history = frame.iloc[: len(train_part)][base_cols_for_recursive].copy()
    recursive_preds: List[float] = []
    holdout_rows = holdout_part[base_cols_for_recursive].copy().reset_index(drop=True)
    for _, hr in holdout_rows.iterrows():
        rec = {"date": pd.Timestamp(hr["date"]), "sales": 0.0}
        for c in base_cols_for_recursive:
            if c in {"date", "sales"}:
                continue
            rec[c] = hr.get(c, np.nan)
        tmp = pd.concat([recursive_history, pd.DataFrame([rec])], ignore_index=True)
        tmp_model = _build_model_frame(tmp)
        x = tmp_model.tail(1)[feature_cols].copy()
        for col in feature_cols:
            if col in cat_feature_names:
                x[col] = x[col].astype(str).fillna(str(train_fill_values.get(col, "unknown")))
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce").fillna(float(train_fill_values.get(col, 0.0)))
        pred = float(np.expm1(model.predict(x)[0]))
        pred = max(0.0, pred)
        recursive_preds.append(pred)
        rec["sales"] = pred
        recursive_history = pd.concat([recursive_history, pd.DataFrame([rec])], ignore_index=True)

    holdout_pred = np.asarray(recursive_preds, dtype=float)
    actual = y_holdout_sales.to_numpy(dtype=float)
    denom = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(actual - holdout_pred)) / denom * 100.0) if denom > 0 else float("nan")
    mae = float(np.mean(np.abs(actual - holdout_pred))) if len(actual) else float("nan")
    rmse = float(np.sqrt(np.mean((actual - holdout_pred) ** 2))) if len(actual) else float("nan")
    mape = float(np.mean(np.abs((actual - holdout_pred) / np.clip(np.abs(actual), 1e-9, None))) * 100.0) if len(actual) else float("nan")
    smape = float(np.mean(2.0 * np.abs(actual - holdout_pred) / np.clip(np.abs(actual) + np.abs(holdout_pred), 1e-9, None)) * 100.0) if len(actual) else float("nan")
    bias = float(np.mean(holdout_pred - actual)) if len(actual) else float("nan")

    X_full = trainable[feature_cols].copy()
    y_full = trainable["target_log_sales"].astype(float)

    fill_values = {}
    for col in feature_cols:
        if col in cat_feature_names:
            X_full[col] = X_full[col].astype(str).fillna("unknown")
            fill_values[col] = "unknown"
        else:
            median_val = pd.to_numeric(X_full[col], errors="coerce").median()
            if not np.isfinite(median_val):
                median_val = 0.0
            X_full[col] = pd.to_numeric(X_full[col], errors="coerce").fillna(median_val)
            fill_values[col] = float(median_val)
    for row in factor_catalog:
        if row["feature"] in fill_values:
            row["fill_value"] = fill_values[row["feature"]]
    if len(feature_report):
        rep_map = feature_report.set_index("feature").to_dict("index")
        for row in factor_catalog:
            meta = rep_map.get(row["feature"], {})
            row["used_in_active_model"] = bool(meta.get("used_in_active_model", False))
            row["exclusion_reason"] = str(meta.get("reason", ""))
            row["importance"] = meta.get("importance", np.nan)

    model.fit(X_full, y_full, cat_features=cat_indices)

    importances = []
    try:
        raw_importances = model.get_feature_importance()
        importances = [{"feature": feature_cols[i], "importance": float(raw_importances[i])} for i in range(len(feature_cols))]
        importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
        if len(feature_report):
            imp_map = {str(x["feature"]): float(x["importance"]) for x in importances}
            feature_report["importance"] = feature_report["feature"].astype(str).map(imp_map)
    except Exception:
        importances = []

    return {
        "enabled": True,
        "reason": "ok",
        "warnings": warnings,
        "model": model,
        "model_backend": "catboost",
        "feature_cols": feature_cols,
        "cat_feature_names": cat_feature_names,
        "cat_indices": cat_indices,
        "fill_values": fill_values,
        "feature_report": feature_report,
        "feature_importances": importances,
        "holdout_metrics": {
            "wape": wape,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "smape": smape,
            "bias": bias,
            "mode": "recursive_daily_holdout",
            "holdout_days": int(holdout_size),
        },
        "holdout_predictions": pd.DataFrame(
            {
                "date": pd.to_datetime(holdout_part["date"], errors="coerce"),
                "actual_sales": actual,
                "predicted_sales": holdout_pred,
                "abs_error": np.abs(actual - holdout_pred),
                "ape": np.abs(actual - holdout_pred) / np.clip(np.abs(actual), 1e-9, None),
            }
        ),
        "history_tail": _prepare_base_columns(daily_base).tail(120).copy(),
        "future_dates": future_dates.copy(),
        "factor_catalog": pd.DataFrame(factor_catalog),
        "guardrails": {
            "numeric_feature_ranges": {
                row["feature"]: {
                    "train_min": row["train_min"],
                    "train_p10": row["train_p10"],
                    "train_median": row["train_median"],
                    "train_p90": row["train_p90"],
                    "train_max": row["train_max"],
                }
                for row in factor_catalog
                if row["dtype"] == "numeric"
            }
        },
    }


def predict_catboost_full_factor_projection(
    trained_bundle: Dict[str, Any],
    manual_price: float,
    freight_multiplier: float = 1.0,
    demand_multiplier: float = 1.0,
    horizon_days: Optional[int] = None,
    discount_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    stock_cap: float = 0.0,
    overrides: Optional[Dict[str, Any]] = None,
    factor_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scenario_overrides = dict(overrides or {})
    scenario_overrides.update(dict(factor_overrides or {}))
    full_bundle = trained_bundle.get("catboost_full_factor_bundle", {})

    if not full_bundle or not full_bundle.get("enabled", False):
        raise RuntimeError(f"CatBoost full factor mode is unavailable: {full_bundle.get('reason', 'unknown')}")

    model = full_bundle["model"]
    feature_cols = list(full_bundle["feature_cols"])
    cat_feature_names = set(full_bundle.get("cat_feature_names", []))
    fill_values = dict(full_bundle.get("fill_values", {}))

    history = trained_bundle["daily_base"].copy()
    history = _prepare_base_columns(history)
    future_dates = trained_bundle["future_dates"].copy()
    if horizon_days is not None:
        future_dates = future_dates.head(int(horizon_days)).copy()

    current_ctx = dict(trained_bundle.get("base_ctx", {}))
    base_price = float(current_ctx.get("price", history["price"].dropna().iloc[-1] if len(history) else manual_price))
    requested_price = float(manual_price)

    train_prices = pd.to_numeric(history["price"], errors="coerce").dropna()
    train_min = float(train_prices.min()) if len(train_prices) else requested_price
    train_max = float(train_prices.max()) if len(train_prices) else requested_price
    price_lo = max(0.01, train_min * 0.90)
    price_hi = max(price_lo, train_max * 1.10)
    model_price = float(np.clip(requested_price, price_lo, price_hi))
    price_clipped = bool(abs(model_price - requested_price) > 1e-9)

    base_discount = float(current_ctx.get("discount", history["discount"].dropna().iloc[-1] if len(history) else 0.0))
    scenario_discount = float(scenario_overrides.get("discount", base_discount * float(discount_multiplier)))
    scenario_discount = float(np.clip(scenario_discount, 0.0, 0.95))

    base_promo = float(current_ctx.get("promotion", history["promotion"].dropna().iloc[-1] if len(history) else 0.0))
    scenario_promo = float(scenario_overrides.get("promotion", base_promo))
    scenario_promo = float(np.clip(scenario_promo, 0.0, 1.0))

    base_freight = float(current_ctx.get("freight_value", history["freight_value"].dropna().iloc[-1] if len(history) else 0.0))
    scenario_freight = float(scenario_overrides.get("freight_value", base_freight * float(freight_multiplier)))
    scenario_freight = max(0.0, scenario_freight)

    base_cost = float(current_ctx.get("cost", history["cost"].dropna().iloc[-1] if len(history) else model_price * 0.65))
    scenario_cost = max(0.0, base_cost * float(cost_multiplier))

    factor_cols = [c for c in history.columns if str(c).startswith("factor__")]
    factor_last_values = {}
    for col in factor_cols:
        vals = history[col].dropna()
        factor_last_values[col] = vals.iloc[-1] if len(vals) else ("unknown" if col in cat_feature_names else 0.0)
    for k, v in scenario_overrides.items():
        if str(k).startswith("factor__"):
            factor_last_values[str(k)] = v
    factor_catalog_df = full_bundle.get("factor_catalog", pd.DataFrame())
    factor_guardrails: Dict[str, Dict[str, Any]] = {}
    ood_count = 0
    guardrail_warnings: List[str] = []
    if isinstance(factor_catalog_df, pd.DataFrame) and len(factor_catalog_df):
        for _, row in factor_catalog_df.iterrows():
            f = str(row.get("feature", ""))
            if not f.startswith("factor__"):
                continue
            val = factor_last_values.get(f, np.nan)
            if str(row.get("dtype", "")) == "numeric":
                lo = float(row.get("train_min", np.nan))
                hi = float(row.get("train_max", np.nan))
                status = "ok"
                if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(pd.to_numeric(val, errors="coerce")):
                    num_val = float(pd.to_numeric(val, errors="coerce"))
                    if num_val < lo or num_val > hi:
                        status = "out_of_range"
                        ood_count += 1
                        guardrail_warnings.append(f"Фактор {f} вне исторического диапазона CatBoost.")
                factor_guardrails[f] = {"value": val, "train_min": lo, "train_max": hi, "status": status}
            else:
                factor_guardrails[f] = {"value": val, "status": "categorical"}

    work_history = history.copy()
    rows = []

    for _, fd_row in future_dates.iterrows():
        dt = pd.Timestamp(fd_row["date"])

        rec = {
            "date": dt,
            "sales": 0.0,
            "price": model_price,
            "discount": scenario_discount,
            "net_unit_price": max(0.01, model_price * (1.0 - scenario_discount)),
            "freight_value": scenario_freight,
            "cost": scenario_cost,
            "promotion": scenario_promo,
            "review_score": float(history["review_score"].dropna().iloc[-1]) if len(history["review_score"].dropna()) else 4.5,
            "reviews_count": float(history["reviews_count"].dropna().iloc[-1]) if len(history["reviews_count"].dropna()) else 0.0,
            "stock": float(stock_cap)
            if stock_cap and stock_cap > 0
            else (float(history["stock"].replace(np.inf, np.nan).dropna().iloc[-1]) if len(history["stock"].replace(np.inf, np.nan).dropna()) else np.inf),
        }

        for col, val in factor_last_values.items():
            rec[col] = val

        tmp = pd.concat([work_history, pd.DataFrame([rec])], ignore_index=True)
        tmp_model = _build_model_frame(tmp)
        x = tmp_model.tail(1)[feature_cols].copy()

        for col in feature_cols:
            if col in cat_feature_names:
                x[col] = x[col].astype(str).fillna(str(fill_values.get(col, "unknown")))
            else:
                x[col] = pd.to_numeric(x[col], errors="coerce").fillna(float(fill_values.get(col, 0.0)))

        pred_sales = float(np.expm1(model.predict(x)[0]))
        pred_sales = max(0.0, pred_sales)
        pred_sales *= float(demand_multiplier)

        available_stock = rec["stock"]
        lost_sales = 0.0
        if np.isfinite(available_stock):
            lost_sales = max(0.0, pred_sales - float(available_stock))
            pred_sales = min(pred_sales, float(available_stock))

        rec["sales"] = pred_sales
        rec["actual_sales"] = pred_sales
        rec["lost_sales"] = lost_sales
        rec["revenue"] = pred_sales * rec["net_unit_price"]
        rec["gross_revenue"] = pred_sales * rec["price"]
        rec["profit"] = rec["revenue"] - pred_sales * rec["cost"] - pred_sales * rec["freight_value"]
        rec["margin"] = rec["profit"] / rec["revenue"] if rec["revenue"] > 0 else 0.0

        rows.append(rec)
        work_history = pd.concat([work_history, pd.DataFrame([rec])], ignore_index=True)

    daily = pd.DataFrame(rows)

    demand_total = float(pd.to_numeric(daily["actual_sales"], errors="coerce").fillna(0.0).sum())
    revenue_total = float(pd.to_numeric(daily["revenue"], errors="coerce").fillna(0.0).sum())
    profit_total = float(pd.to_numeric(daily["profit"], errors="coerce").fillna(0.0).sum())
    lost_sales_total = float(pd.to_numeric(daily["lost_sales"], errors="coerce").fillna(0.0).sum())

    support_label = "medium"
    warnings = list(full_bundle.get("warnings", []))
    warnings.extend(guardrail_warnings)
    if price_clipped:
        warnings.append("Цена была ограничена guardrails CatBoost full factor mode.")
    metrics = full_bundle.get("holdout_metrics", {})
    wape = metrics.get("wape", np.nan)
    if np.isfinite(wape) and float(wape) > 40.0:
        warnings.append(f"Высокий holdout WAPE CatBoost full factors: {float(wape):.1f}%.")

    confidence = float(1.0 / (1.0 + max(0.0, float(wape) / 100.0))) if np.isfinite(wape) else 0.5
    if ood_count >= 3:
        support_label = "low"
        confidence *= 0.7

    return {
        "daily": daily,
        "demand_total": demand_total,
        "profit_total": profit_total,
        "profit_total_raw": profit_total,
        "profit_total_adjusted": profit_total,
        "uncertainty_penalty": 0.0,
        "disagreement_penalty": 0.0,
        "revenue_total": revenue_total,
        "lost_sales_total": lost_sales_total,
        "confidence": confidence,
        "confidence_base": confidence,
        "confidence_scenario": confidence,
        "confidence_label": "medium" if confidence >= 0.5 else "low",
        "uncertainty": 1.0 - confidence,
        "ood_flag": bool(price_clipped),
        "requested_price": requested_price,
        "model_price": model_price,
        "price_for_model": model_price,
        "current_price_raw": base_price,
        "price_clipped": price_clipped,
        "clip_applied": price_clipped,
        "clip_reason": "catboost_full_factor_price_guardrail" if price_clipped else "",
        "net_price_supported": True,
        "net_price_support": {},
        "scenario_price_effect_source": "catboost_full_factor_reprediction",
        "fallback_multiplier_used": False,
        "fallback_reason": "",
        "learned_uplift_active": False,
        "uplift_mode": "not_used_in_catboost_full_factor_mode",
        "uplift_used_in_production": False,
        "scenario_status": "computed",
        "scenario_driver_mode": "catboost_full_factor_reprediction",
        "weekly_driver_mode": "not_used_daily_catboost",
        "baseline_has_exogenous_driver": True,
        "legacy_simulation_used": False,
        "legacy_baseline_meta": {},
        "active_path_contract": "daily_catboost_full_factors+model_reprediction",
        "baseline_forecast_path": "daily_catboost_full_factor_baseline",
        "scenario_calculation_path": "catboost_full_factor_reprediction",
        "learned_uplift_path": "inactive_not_used_in_this_mode",
        "final_user_visible_path": "daily_catboost_full_factor_baseline + catboost_full_factor_reprediction",
        "model_backend": "catboost",
        "backend_reason": "catboost_full_factor_mode",
        "scenario_engine_meta": {
            "mode": CATBOOST_FULL_FACTOR_MODE,
            "feature_count": len(feature_cols),
            "cat_feature_count": len(cat_feature_names),
            "top_features": full_bundle.get("feature_importances", [])[:15],
            "holdout_metrics": metrics,
        },
        "applied_overrides": scenario_overrides,
        "effects": {
            "model_reprediction": True,
            "manual_shock_multiplier": float(demand_multiplier),
            "price_clipped": price_clipped,
        },
        "scenario_inputs_contract": {
            "base_price": base_price,
            "scenario_price": model_price,
            "requested_price": requested_price,
            "model_price": model_price,
            "base_promo": base_promo,
            "scenario_promo": scenario_promo,
            "base_freight": base_freight,
            "scenario_freight": scenario_freight,
            "base_discount": base_discount,
            "scenario_discount": scenario_discount,
            "shock_multiplier": float(demand_multiplier),
        },
        "scenario_calc_mode": CATBOOST_FULL_FACTOR_MODE,
        "scenario_calc_mode_label": "CatBoost full factors: модельный what-if по факторам",
        "active_path_contract_label": "Daily CatBoost full factors + model reprediction",
        "effect_source": "catboost_full_factor_reprediction",
        "effect_source_label": "CatBoost повторно прогнозирует спрос по изменённым факторам",
        "support_label": support_label,
        "scenario_support_info": {
            "support_label": support_label,
            "warnings": warnings,
        },
        "legacy_or_enhanced_label": "catboost_full_factors",
        "applied_path_summary": {
            "mode": "CatBoost full factors: модельный what-if по факторам",
            "segment_count": 0,
            "price_net_min": float(daily["net_unit_price"].min()) if len(daily) else np.nan,
            "price_net_avg": float(daily["net_unit_price"].mean()) if len(daily) else np.nan,
            "price_net_max": float(daily["net_unit_price"].max()) if len(daily) else np.nan,
            "promo_days": int((daily["promotion"] > 0).sum()) if "promotion" in daily else 0,
            "promo_share": float((daily["promotion"] > 0).mean()) if "promotion" in daily and len(daily) else 0.0,
            "avg_freight": float(daily["freight_value"].mean()) if len(daily) else 0.0,
            "avg_demand_multiplier": float(demand_multiplier),
            "segment_shocks": [],
            "support_label": support_label,
            "warnings": warnings,
        },
        "effective_scenario": {
            "requested_price_gross": requested_price,
            "applied_price_gross": model_price,
            "applied_discount": scenario_discount,
            "applied_price_net": max(0.01, model_price * (1.0 - scenario_discount)),
            "promotion": scenario_promo,
            "freight_value": scenario_freight,
            "clip_reason": "catboost_full_factor_price_guardrail" if price_clipped else "",
        },
        "confidence_factors": {},
        "warnings": warnings,
        "factor_guardrails": factor_guardrails,
    }
