from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_schema import canonical_alias_map, canonical_required_fields
from df_utils import get_numeric_series, get_text_series

USER_FACTOR_NUM_PREFIX = "user_factor_num__"
USER_FACTOR_CAT_PREFIX = "user_factor_cat__"


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def suggest_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    norm_to_orig = {_norm_col(c): c for c in columns}
    for alias in aliases:
        if alias in norm_to_orig:
            return norm_to_orig[alias]
    for col in columns:
        nc = _norm_col(col)
        if any(alias in nc for alias in aliases):
            return col
    return None


def build_auto_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    alias_map = canonical_alias_map()
    return {field: suggest_column(columns, aliases) for field, aliases in alias_map.items()}


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {src: dst for dst, src in mapping.items() if src is not None and src in renamed.columns}
    return renamed.rename(columns=rename_map)


def _mode_or_unknown(s: pd.Series) -> str:
    v = s.dropna().astype(str)
    if v.empty:
        return "unknown"
    m = v.mode(dropna=True)
    return str(m.iloc[0]) if not m.empty else "unknown"


def _weighted_mean_by_quantity(values: pd.Series, quantity: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    q = pd.to_numeric(quantity, errors="coerce").fillna(0.0)
    mask = v.notna() & q.notna()
    if not mask.any():
        return float("nan")
    v2 = v[mask]
    q2 = q[mask].clip(lower=0.0)
    total_q = float(q2.sum())
    if total_q > 0:
        return float((v2 * q2).sum() / total_q)
    return float(v2.mean())


def normalize_transactions(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = apply_mapping(df, mapping)
    quality: Dict[str, Any] = {"warnings": [], "errors": [], "raw_stats": {}, "can_recommend": True, "data_quality": "ok"}

    required = canonical_required_fields()
    missing_required = [c for c in required if c not in out.columns]
    if missing_required:
        quality["errors"].append(f"Отсутствуют обязательные поля: {missing_required}")
        out.attrs["normalization_quality"] = quality
        return out, quality

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["price", "quantity", "revenue", "cost", "discount_rate", "discount_amount", "freight_value", "stock", "promotion", "rating", "review_score", "reviews_count"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    discount_amount_present = "discount_amount" in out.columns and get_numeric_series(out, "discount_amount", np.nan).notna().any()
    discount_rate_raw = get_numeric_series(out, "discount_rate", 0.0)
    dr_non_na = discount_rate_raw.dropna()
    if dr_non_na.empty:
        discount_rate = discount_rate_raw.fillna(0.0)
    elif ((dr_non_na >= 0) & (dr_non_na <= 1)).all():
        discount_rate = discount_rate_raw.fillna(0.0)
    elif ((dr_non_na >= 0) & (dr_non_na <= 100)).all() and not discount_amount_present:
        discount_rate = discount_rate_raw.fillna(0.0) / 100.0
        quality["warnings"].append("discount_rate interpreted as percent 0..100 and converted to 0..1")
    else:
        quality["errors"].append("discount_rate has invalid or mixed scale; expected 0..1 (or 0..100 only without discount_amount)")
        discount_rate = discount_rate_raw.clip(lower=0.0, upper=1.0).fillna(0.0)
    out["discount_rate"] = discount_rate
    out["discount"] = out["discount_rate"]

    out["quantity"] = get_numeric_series(out, "quantity", 1.0).fillna(1.0).clip(lower=0.0)

    out["price"] = get_numeric_series(out, "price", np.nan)
    out = out[out["price"].notna() & (out["price"] > 0)].copy()

    if "revenue" not in out.columns:
        out["revenue"] = out["price"] * out["quantity"]
    out["revenue"] = get_numeric_series(out, "revenue", np.nan).fillna(out["price"] * out["quantity"])

    if "cost" not in out.columns or get_numeric_series(out, "cost", np.nan).notna().sum() == 0:
        out["cost"] = out["price"] * 0.65
        quality["warnings"].append("cost missing: unit economics approximate, fallback cost=65% of price used")
    else:
        out["cost"] = get_numeric_series(out, "cost", np.nan).fillna(out["price"] * 0.65)

    if "review_score" not in out.columns and "rating" in out.columns:
        out["review_score"] = get_numeric_series(out, "rating", np.nan)
    out["review_score"] = get_numeric_series(out, "review_score", np.nan)

    out["reviews_count"] = get_numeric_series(out, "reviews_count", 0.0).fillna(0.0)
    out["freight_value"] = get_numeric_series(out, "freight_value", 0.0).fillna(0.0).clip(lower=0.0)
    out["stock"] = get_numeric_series(out, "stock", 0.0).fillna(0.0).clip(lower=0.0)
    out["promotion"] = get_numeric_series(out, "promotion", 0.0).fillna(0.0).clip(lower=0.0)

    for c in ["category", "region", "channel", "segment"]:
        out[c] = get_text_series(out, c, "unknown")
    out["product_id"] = get_text_series(out, "product_id", "unknown")

    out["quantity"] = get_numeric_series(out, "quantity", 0.0).fillna(0.0).clip(lower=0.0)
    out["revenue"] = get_numeric_series(out, "revenue", 0.0).fillna(0.0).clip(lower=0.0)
    out["cost"] = get_numeric_series(out, "cost", 0.0).fillna(0.0).clip(lower=0.0)

    base_known_cols = {
        "date", "product_id", "category", "quantity", "revenue", "price", "cost", "discount_rate", "discount", "discount_amount",
        "freight_value", "stock", "promotion", "rating", "review_score", "reviews_count", "region", "channel", "segment",
    }

    for c in list(out.columns):
        if c in base_known_cols:
            continue
        nc = _norm_col(c)
        if nc == "review_score":
            continue
        if c.startswith(USER_FACTOR_NUM_PREFIX) or c.startswith(USER_FACTOR_CAT_PREFIX):
            continue
        num = pd.to_numeric(out[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(out[c]) or num.notna().mean() > 0.9:
            out[f"{USER_FACTOR_NUM_PREFIX}{nc}"] = num
        else:
            out[f"{USER_FACTOR_CAT_PREFIX}{nc}"] = out[c].astype(str)

    for c in list(out.columns):
        if c in {f"{USER_FACTOR_NUM_PREFIX}review_score", f"{USER_FACTOR_CAT_PREFIX}review_score", f"{USER_FACTOR_NUM_PREFIX}reviews_count", f"{USER_FACTOR_CAT_PREFIX}reviews_count"}:
            out = out.drop(columns=[c])

    checks = run_data_quality_checks(out)
    quality["warnings"].extend(checks.get("warnings", []))
    quality["stats"] = checks.get("stats", {})
    out.attrs["normalization_quality"] = quality
    return out.reset_index(drop=True), quality


def run_data_quality_checks(df: pd.DataFrame, raw_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    issues: Dict[str, Any] = {"warnings": [], "stats": {}}
    if len(df) == 0:
        issues["warnings"].append("Данные пустые после нормализации")
        return issues
    issues["stats"]["rows"] = int(len(df))
    issues["stats"]["missing_share"] = float(df.isna().mean().mean())
    issues["stats"]["history_days"] = int((df["date"].max() - df["date"].min()).days + 1)
    if issues["stats"]["history_days"] < 60:
        issues["warnings"].append("Короткая история (<60 дней): рекомендации менее устойчивы")
    if raw_stats:
        issues["stats"].update(raw_stats)
    return issues


def build_daily_panel_from_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    df = txn.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["quantity"] = get_numeric_series(df, "quantity", 0.0).fillna(0.0).clip(lower=0.0)
    df["revenue"] = get_numeric_series(df, "revenue", 0.0).fillna(0.0).clip(lower=0.0)

    numeric_covariates = ["price", "cost", "discount_rate", "freight_value", "stock", "promotion", "review_score", "reviews_count"] + [c for c in df.columns if c.startswith(USER_FACTOR_NUM_PREFIX)]
    cat_mode_cols = ["region", "channel", "segment"] + [c for c in df.columns if c.startswith(USER_FACTOR_CAT_PREFIX)]
    series_keys = ["product_id", "category", "region", "channel", "segment"]

    for c in ["product_id", "category"] + cat_mode_cols:
        df[c] = get_text_series(df, c, "unknown")
    for c in numeric_covariates:
        df[c] = get_numeric_series(df, c, np.nan)

    grouped = []
    for keys, g in df.groupby([pd.Grouper(key="date", freq="D")] + series_keys, dropna=False):
        dt, pid, cat, region, channel, segment = keys
        row: Dict[str, Any] = {
            "date": pd.Timestamp(dt),
            "product_id": str(pid),
            "category": str(cat),
            "region": str(region),
            "channel": str(channel),
            "segment": str(segment),
            "sales": float(pd.to_numeric(g["quantity"], errors="coerce").fillna(0.0).sum()),
            "revenue": float(pd.to_numeric(g["revenue"], errors="coerce").fillna(0.0).sum()),
            "price": _weighted_mean_by_quantity(g["price"], g["quantity"]),
            "cost": _weighted_mean_by_quantity(g["cost"], g["quantity"]),
            "discount_rate": _weighted_mean_by_quantity(g["discount_rate"], g["quantity"]),
            "freight_value": _weighted_mean_by_quantity(g["freight_value"], g["quantity"]),
            "promotion": float(pd.to_numeric(g["promotion"], errors="coerce").max(skipna=True)) if len(g) else np.nan,
            "stock": float(pd.to_numeric(g["stock"], errors="coerce").max(skipna=True)) if len(g) else np.nan,
            "review_score": float(pd.to_numeric(g["review_score"], errors="coerce").mean(skipna=True)) if len(g) else np.nan,
            "reviews_count": float(pd.to_numeric(g["reviews_count"], errors="coerce").max(skipna=True)) if len(g) else np.nan,
        }
        for c in [c for c in numeric_covariates if c.startswith(USER_FACTOR_NUM_PREFIX)]:
            row[c] = _weighted_mean_by_quantity(g[c], g["quantity"])
        for c in cat_mode_cols:
            row[c] = _mode_or_unknown(g[c])
        grouped.append(row)

    panel = pd.DataFrame(grouped)
    if panel.empty:
        return panel

    pieces = []
    num_fill_cols = ["price", "cost", "discount_rate", "freight_value", "stock", "promotion", "review_score", "reviews_count"] + [c for c in panel.columns if c.startswith(USER_FACTOR_NUM_PREFIX)]
    for _, g in panel.groupby(series_keys, dropna=False):
        g = g.sort_values("date").reset_index(drop=True)
        full = pd.DataFrame({"date": pd.date_range(g["date"].min(), g["date"].max(), freq="D")})
        for key in series_keys:
            full[key] = g[key].iloc[0] if key in g.columns else "unknown"
        m = full.merge(g, on=["date"] + series_keys, how="left")
        m["has_observation_flag"] = get_numeric_series(m, "sales", np.nan).notna().astype(int)

        for c in ["sales", "revenue"]:
            m[c] = get_numeric_series(m, c, 0.0).fillna(0.0)
        for c in ["price", "cost", "freight_value"]:
            if c in m.columns:
                m[c] = get_numeric_series(m, c, np.nan).ffill(limit=7)
        for c in ["discount_rate", "promotion"]:
            if c in m.columns:
                m[c] = get_numeric_series(m, c, 0.0).fillna(0.0).clip(lower=0.0)
        if "stock" in m.columns:
            m["stock"] = get_numeric_series(m, "stock", 0.0).fillna(0.0).clip(lower=0.0)
        for c in [c for c in num_fill_cols if c.startswith(USER_FACTOR_NUM_PREFIX)]:
            m[c] = get_numeric_series(m, c, np.nan)
        for c in cat_mode_cols:
            m[c] = get_text_series(m, c, "unknown").ffill().fillna("unknown")
        m["stockout_flag"] = (get_numeric_series(m, "stock", 0.0).fillna(0.0) <= 0.0).astype(int)
        pieces.append(m)

    out = pd.concat(pieces, ignore_index=True)
    out["discount"] = get_numeric_series(out, "discount_rate", 0.0).fillna(0.0)
    out["series_id"] = (
        out["product_id"].astype(str)
        + "|"
        + out["region"].astype(str)
        + "|"
        + out["channel"].astype(str)
        + "|"
        + out["segment"].astype(str)
    )
    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def build_daily_from_transactions(txn: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    panel = build_daily_panel_from_transactions(txn)
    daily = panel[panel["product_id"].astype(str) == str(sku_id)].copy()
    if daily.empty:
        raise ValueError("Нет данных по выбранному SKU.")
    daily["sku_id"] = str(sku_id)
    daily["price_median"] = daily["price"]
    return daily.reset_index(drop=True)


def build_daily_from_transactions_scoped(txn: pd.DataFrame, sku_id: str, category: Optional[str] = None) -> pd.DataFrame:
    mask = txn["product_id"].astype(str) == str(sku_id)
    if category is not None and "category" in txn.columns:
        mask &= txn["category"].astype(str) == str(category)
    return build_daily_from_transactions(txn[mask].copy(), sku_id)


def objective_to_weights(mode: str) -> Dict[str, float]:
    presets = {
        "maximize_profit": {"profit": 1.0, "revenue": 0.25, "volume": 0.15, "margin": 0.35, "risk": 0.25},
        "maximize_revenue": {"profit": 0.35, "revenue": 1.0, "volume": 0.2, "margin": 0.15, "risk": 0.1},
        "protect_volume": {"profit": 0.2, "revenue": 0.35, "volume": 1.0, "margin": 0.15, "risk": 0.8},
        "balanced_mode": {"profit": 0.7, "revenue": 0.7, "volume": 0.7, "margin": 0.7, "risk": 0.7},
        "control_margin": {"profit": 0.7, "revenue": 0.7, "volume": 0.7, "margin": 0.7, "risk": 0.7},
    }
    if mode not in presets:
        raise ValueError(f"Unknown objective mode: {mode}")
    return presets[mode]
