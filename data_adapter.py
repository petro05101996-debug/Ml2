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
    quality: Dict[str, Any] = {"warnings": [], "errors": [], "raw_stats": {}}

    required = canonical_required_fields()
    missing_required = [c for c in required if c not in out.columns]
    if missing_required:
        quality["errors"].append(f"Отсутствуют обязательные поля: {missing_required}")
        return out, quality

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    bad_dates = int(out["date"].isna().sum())
    out = out.dropna(subset=["date"]).copy()
    if bad_dates > 0:
        quality["warnings"].append(f"Удалены строки с невалидной датой: {bad_dates}")
    out = out.sort_values("date").reset_index(drop=True)

    if "discount_rate" not in out.columns and "discount" in out.columns:
        out["discount_rate"] = out["discount"]

    for c in ["price", "quantity", "revenue", "cost", "discount_rate", "freight_value", "stock", "promotion", "rating", "review_score", "reviews_count"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "price" in out.columns:
        bad_price_count = int(out["price"].isna().sum())
        nonpositive_price_count = int((out["price"].fillna(0.0) <= 0).sum())
        quality["raw_stats"]["bad_price_count"] = bad_price_count
        quality["raw_stats"]["nonpositive_price_count"] = nonpositive_price_count
        out = out.dropna(subset=["price"]).copy()
        out = out[out["price"] > 0].copy()
        invalid_price_total = bad_price_count + nonpositive_price_count
        if invalid_price_total > 0:
            quality["warnings"].append(f"Удалены строки с невалидной ценой (NaN или <=0): {invalid_price_total}")

    if "quantity" not in out.columns:
        out["quantity"] = 1.0
        quality["warnings"].append("quantity не найдено — использовано значение 1 для каждой записи")
        missing_quantity_count = 0
    else:
        missing_quantity_count = int(out["quantity"].isna().sum())
        out["quantity"] = out["quantity"].fillna(1.0)
    quality["raw_stats"]["missing_quantity_count"] = missing_quantity_count
    if missing_quantity_count > 0:
        quality["warnings"].append(f"quantity содержит пропуски: {missing_quantity_count}. Заполнено значением 1.")

    if "revenue" not in out.columns:
        out["revenue"] = out["price"].fillna(0.0) * out["quantity"].fillna(0.0)
        quality["warnings"].append("revenue не найдено — рассчитано как price * quantity")
        missing_revenue_count = 0
    else:
        missing_revenue_count = int(out["revenue"].isna().sum())
        out["revenue"] = out["revenue"].where(out["revenue"].notna(), out["price"].fillna(0.0) * out["quantity"].fillna(0.0))
    quality["raw_stats"]["missing_revenue_count"] = missing_revenue_count
    if missing_revenue_count > 0:
        quality["warnings"].append(f"revenue содержит пропуски: {missing_revenue_count}. Заполнено как price * quantity.")

    if "cost" not in out.columns:
        out["cost"] = out["price"].fillna(0.0) * 0.65
        quality["warnings"].append("cost не найдено — использован proxy: 65% от цены")
        missing_cost_count = 0
    else:
        missing_cost_count = int(out["cost"].isna().sum())
        out["cost"] = out["cost"].where(out["cost"].notna(), out["price"].fillna(0.0) * 0.65)
    quality["raw_stats"]["missing_cost_count"] = missing_cost_count
    if missing_cost_count > 0:
        quality["warnings"].append(f"cost содержит пропуски: {missing_cost_count}. Заполнено proxy 65% от цены.")

    if "discount_amount" in out.columns:
        raw_amount = pd.to_numeric(out["discount_amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
        out["discount_rate"] = (
            raw_amount / out["price"].replace(0, np.nan)
        ).fillna(0.0).clip(lower=0.0, upper=0.95)

    elif "discount_rate" in out.columns:
        raw_rate = pd.to_numeric(out["discount_rate"], errors="coerce")

        mixed_scale = ((raw_rate > 1.0).any()) and ((raw_rate.between(0.0, 1.0, inclusive="both")).any())
        if mixed_scale:
            quality["warnings"].append(
                "Обнаружен смешанный формат discount_rate: часть строк похожа на rate, часть на absolute amount. "
                "Для корректного расчёта нужен отдельный столбец discount_amount."
            )

        if (raw_rate > 1.0).all():
            out["discount_rate"] = (
                raw_rate / out["price"].replace(0, np.nan)
            ).fillna(0.0).clip(lower=0.0, upper=0.95)
            quality["warnings"].append("discount_rate интерпретирован как абсолютная скидка по всем строкам.")
        else:
            out["discount_rate"] = raw_rate.fillna(0.0).clip(lower=0.0, upper=0.95)

    if "freight_value" not in out.columns:
        out["freight_value"] = 0.0

    if "discount_rate" not in out.columns:
        out["discount_rate"] = 0.0
    # Legacy compatibility: core still reads discount in multiple places.
    out["discount"] = pd.to_numeric(out["discount_rate"], errors="coerce").fillna(0.0)
    if "review_score" not in out.columns and "rating" in out.columns:
        out["review_score"] = pd.to_numeric(out["rating"], errors="coerce")
    if "review_score" in out.columns:
        out["review_score"] = pd.to_numeric(out["review_score"], errors="coerce")
    if "rating" not in out.columns and "review_score" in out.columns:
        out["rating"] = out["review_score"]

    if "category" not in out.columns:
        out["category"] = "unknown"
    out["category"] = out["category"].fillna("unknown").astype(str)

    out["product_id"] = out["product_id"].astype(str)
    raw_duplicates = int(out.duplicated().sum())
    quality["raw_stats"]["raw_duplicates"] = raw_duplicates
    out = out.drop_duplicates().reset_index(drop=True)
    quality["raw_stats"]["deduped_rows"] = raw_duplicates

    checks = run_data_quality_checks(out, raw_stats=quality.get("raw_stats", {}))
    quality["warnings"].extend(checks.get("warnings", []))
    quality["stats"] = checks.get("stats", {})
    return out, quality


def run_data_quality_checks(df: pd.DataFrame, raw_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    issues: Dict[str, Any] = {"warnings": [], "stats": {}}
    if len(df) == 0:
        issues["warnings"].append("Данные пустые после нормализации")
        return issues

    issues["stats"]["rows"] = int(len(df))
    issues["stats"]["missing_share"] = float(df.isna().mean().mean())
    issues["stats"]["duplicates_in_clean_data"] = int(df.duplicated().sum())
    if raw_stats:
        issues["stats"]["raw_duplicates"] = int(raw_stats.get("raw_duplicates", 0))
        issues["stats"]["deduped_rows"] = int(raw_stats.get("deduped_rows", 0))
        issues["stats"]["bad_price_count"] = int(raw_stats.get("bad_price_count", 0))
        issues["stats"]["nonpositive_price_count"] = int(raw_stats.get("nonpositive_price_count", 0))
    history_days = int((df["date"].max() - df["date"].min()).days + 1)
    issues["stats"]["history_days"] = history_days

    if issues["stats"]["missing_share"] > 0.2:
        issues["warnings"].append("Высокая доля пропусков (>20%)")
    if issues["stats"].get("raw_duplicates", 0) > 0:
        issues["warnings"].append(f"Обнаружены дубликаты в сыром входе: {issues['stats']['raw_duplicates']}")
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
    sku["date"] = pd.to_datetime(sku["date"], errors="coerce")
    sku = sku.dropna(subset=["date"]).copy()
    if len(sku) == 0:
        raise ValueError("Нет валидных дат по выбранному SKU.")

    agg_map = {
        "sales": ("quantity", "sum"),
        "revenue": ("revenue", "sum"),
        "price": ("price", "mean"),
        "price_median": ("price", "median"),
    }
    optional_aggs = {
        "freight_value": ("freight_value", "mean"),
        "discount_rate": ("discount_rate", "mean"),
        "cost": ("cost", "mean"),
        "stock": ("stock", "mean"),
        "promotion": ("promotion", "mean"),
        "review_score": ("rating", "mean"),
        "reviews_count": ("reviews_count", "mean"),
    }
    base_known_cols = {
        "date", "product_id", "category", "quantity", "revenue", "price",
        "cost", "discount_rate", "discount", "freight_value", "stock", "promotion", "rating",
        "reviews_count",
    }
    extra_factor_cols: Dict[str, str] = {}
    for c in sku.columns:
        if c in base_known_cols:
            continue
        if c.startswith("user_factor__"):
            extra_factor_cols[c] = c
            continue
        if pd.api.types.is_numeric_dtype(sku[c]):
            extra_factor_cols[c] = f"user_factor__{_norm_col(c)}"
            continue
        converted = pd.to_numeric(sku[c], errors="coerce")
        if converted.notna().any():
            sku[c] = converted
            extra_factor_cols[c] = f"user_factor__{_norm_col(c)}"
    for src_col, out_col in extra_factor_cols.items():
        agg_map[out_col] = (src_col, "mean")
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
    for c in ["price", "price_median", "freight_value", "discount_rate", "cost", "stock", "promotion", "review_score", "reviews_count"]:
        if c not in daily.columns:
            daily[c] = np.nan
        daily[c] = pd.to_numeric(daily[c], errors="coerce").ffill()
    for c in extra_factor_cols.values():
        if c not in daily.columns:
            daily[c] = np.nan
        daily[c] = pd.to_numeric(daily[c], errors="coerce").ffill()

    daily["price"] = daily["price"].fillna(1.0).clip(lower=0.01)
    daily["price_median"] = daily["price_median"].fillna(daily["price"]).clip(lower=0.01)
    daily["freight_value"] = daily["freight_value"].fillna(0.0).clip(lower=0.0)
    daily["discount_rate"] = daily["discount_rate"].fillna(0.0).clip(lower=0.0, upper=0.95)
    daily["discount"] = daily["discount_rate"]
    daily["cost"] = daily["cost"].fillna(daily["price"] * 0.65).clip(lower=0.0)
    daily["stock"] = daily["stock"].fillna(0.0).clip(lower=0.0)
    daily["promotion"] = daily["promotion"].fillna(0.0).clip(lower=0.0)
    daily["review_score"] = daily["review_score"].fillna(4.5)
    daily["reviews_count"] = daily["reviews_count"].fillna(0.0).clip(lower=0.0)
    for c in extra_factor_cols.values():
        median_val = float(pd.to_numeric(daily[c], errors="coerce").median()) if pd.to_numeric(daily[c], errors="coerce").notna().any() else 0.0
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(median_val)
    daily["category"] = sku["category"].mode().iloc[0] if "category" in sku.columns and not sku["category"].dropna().empty else "unknown"
    daily["sku_id"] = str(sku_id)
    return daily


def build_daily_from_transactions_scoped(txn: pd.DataFrame, sku_id: str, category: Optional[str] = None) -> pd.DataFrame:
    mask = txn["product_id"].astype(str) == str(sku_id)
    if category is not None and "category" in txn.columns:
        mask &= txn["category"].astype(str) == str(category)
    scoped = txn[mask].copy()
    return build_daily_from_transactions(scoped, sku_id)


def objective_to_weights(mode: str) -> Dict[str, float]:
    presets = {
        "maximize_profit": {"profit": 1.0, "revenue": 0.25, "volume": 0.15, "margin": 0.35, "risk": 0.25},
        "maximize_revenue": {"profit": 0.35, "revenue": 1.0, "volume": 0.2, "margin": 0.15, "risk": 0.1},
        "protect_volume": {"profit": 0.2, "revenue": 0.35, "volume": 1.0, "margin": 0.15, "risk": 0.8},
        "balanced_mode": {"profit": 0.7, "revenue": 0.7, "volume": 0.7, "margin": 0.7, "risk": 0.7},
        # Legacy alias kept for backward compatibility in API/old sessions.
        "control_margin": {"profit": 0.7, "revenue": 0.7, "volume": 0.7, "margin": 0.7, "risk": 0.7},
    }
    if mode not in presets:
        raise ValueError(f"Unknown objective mode: {mode}")
    return presets[mode]
