from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_schema import canonical_alias_map, canonical_required_fields


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
    renamed = renamed.rename(columns=rename_map)
    return renamed


def normalize_transactions(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = apply_mapping(df, mapping)
    quality: Dict[str, Any] = {"warnings": [], "errors": []}

    required = canonical_required_fields()
    missing_required = [c for c in required if c not in out.columns]
    if missing_required:
        quality["errors"].append(f"Отсутствуют обязательные поля: {missing_required}")
        return out, quality

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    for c in ["price", "quantity", "revenue", "cost", "discount", "freight_value", "stock", "promotion", "rating", "reviews_count"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "quantity" not in out.columns:
        out["quantity"] = 1.0
        quality["warnings"].append("quantity не найдено — использовано значение 1 для каждой записи")

    if "revenue" not in out.columns:
        out["revenue"] = out["price"].fillna(0.0) * out["quantity"].fillna(0.0)
        quality["warnings"].append("revenue не найдено — рассчитано как price * quantity")

    if "cost" not in out.columns:
        out["cost"] = out["price"].fillna(0.0) * 0.65
        quality["warnings"].append("cost не найдено — использован proxy: 65% от цены")

    if "freight_value" not in out.columns:
        out["freight_value"] = 0.0

    if "discount" not in out.columns:
        out["discount"] = 0.0

    if "category" not in out.columns:
        out["category"] = "unknown"
    out["category"] = out["category"].fillna("unknown").astype(str)

    out["product_id"] = out["product_id"].astype(str)

    checks = run_data_quality_checks(out)
    quality["warnings"].extend(checks.get("warnings", []))
    quality["stats"] = checks.get("stats", {})
    return out, quality


def run_data_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    issues: Dict[str, Any] = {"warnings": [], "stats": {}}
    if len(df) == 0:
        issues["warnings"].append("Данные пустые после нормализации")
        return issues

    issues["stats"]["rows"] = int(len(df))
    issues["stats"]["missing_share"] = float(df.isna().mean().mean())
    issues["stats"]["duplicates"] = int(df.duplicated().sum())
    history_days = int((df["date"].max() - df["date"].min()).days + 1)
    issues["stats"]["history_days"] = history_days

    if issues["stats"]["missing_share"] > 0.2:
        issues["warnings"].append("Высокая доля пропусков (>20%)")
    if issues["stats"]["duplicates"] > 0:
        issues["warnings"].append(f"Обнаружены дубликаты: {issues['stats']['duplicates']}")
    if history_days < 60:
        issues["warnings"].append("Короткая история (<60 дней): рекомендации менее устойчивы")
    if df["product_id"].nunique() < 2:
        issues["warnings"].append("Слишком мало SKU для надежного сравнения")

    if "price" in df.columns and df["price"].notna().sum() > 10:
        q1, q99 = df["price"].quantile([0.01, 0.99]).tolist()
        outliers = int(((df["price"] < q1) | (df["price"] > q99)).sum())
        issues["stats"]["price_outliers"] = outliers
        if outliers / max(len(df), 1) > 0.03:
            issues["warnings"].append("Много ценовых выбросов (>3%)")

    return issues


def build_daily_from_transactions(txn: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    sku = txn[txn["product_id"].astype(str) == str(sku_id)].copy()
    if len(sku) == 0:
        raise ValueError("Нет данных по выбранному SKU.")

    agg_map = {
        "sales": ("quantity", "sum"),
        "revenue": ("revenue", "sum"),
        "price": ("price", "mean"),
        "price_median": ("price", "median"),
    }
    optional_aggs = {
        "freight_value": ("freight_value", "mean"),
        "discount": ("discount", "mean"),
        "cost": ("cost", "mean"),
        "stock": ("stock", "mean"),
        "review_score": ("rating", "mean"),
    }
    for out_col, (in_col, func) in optional_aggs.items():
        if in_col in sku.columns:
            agg_map[out_col] = (in_col, func)

    daily = sku.groupby(pd.Grouper(key="date", freq="D")).agg(**agg_map).reset_index().rename(columns={"date": "date"})
    if len(daily) == 0 or daily["date"].isna().all():
        raise ValueError("После агрегации не осталось валидных дат по SKU.")
    full_dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full_dates.merge(daily, on="date", how="left").sort_values("date").reset_index(drop=True)

    for c in ["sales", "revenue"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
    for c in ["price", "price_median", "freight_value", "discount", "cost", "stock", "review_score"]:
        if c not in daily.columns:
            daily[c] = np.nan
        daily[c] = pd.to_numeric(daily[c], errors="coerce").ffill().bfill()

    daily["price"] = daily["price"].fillna(1.0).clip(lower=0.01)
    daily["price_median"] = daily["price_median"].fillna(daily["price"]).clip(lower=0.01)
    daily["freight_value"] = daily["freight_value"].fillna(0.0).clip(lower=0.0)
    daily["review_score"] = daily["review_score"].fillna(4.5)
    daily["category"] = sku["category"].mode().iloc[0] if "category" in sku.columns and not sku["category"].dropna().empty else "unknown"
    daily["sku_id"] = str(sku_id)
    return daily


def objective_to_weights(mode: str) -> Dict[str, float]:
    presets = {
        "maximize_profit": {"profit": 1.0, "revenue": 0.2, "volume": 0.1, "margin": 0.4},
        "maximize_revenue": {"profit": 0.3, "revenue": 1.0, "volume": 0.4, "margin": 0.1},
        "protect_volume": {"profit": 0.3, "revenue": 0.4, "volume": 1.0, "margin": 0.2},
        "control_margin": {"profit": 0.6, "revenue": 0.2, "volume": 0.1, "margin": 1.0},
    }
    return presets.get(mode, presets["maximize_profit"])
