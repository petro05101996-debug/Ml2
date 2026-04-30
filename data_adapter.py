from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_schema import canonical_alias_map, canonical_field_registry, canonical_required_fields


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
    quality["feature_eligibility"] = build_feature_eligibility_report(out)
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


def _wavg(frame: pd.DataFrame, value_col: str, weight_col: str = "quantity") -> float:
    vals = pd.to_numeric(frame[value_col], errors="coerce")
    w = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    valid = vals.notna() & w.notna() & (w > 0)
    if valid.any():
        return float(np.average(vals[valid], weights=w[valid]))
    return float(vals.dropna().mean()) if vals.notna().any() else float("nan")


def build_daily_from_transactions(
    txn: pd.DataFrame,
    sku_id: str,
    target_category: Optional[str] = None,
    region: Optional[str] = None,
    channel: Optional[str] = None,
    segment: Optional[str] = None,
    include_extra_factors: bool = False,
) -> pd.DataFrame:
    sku = txn[txn["product_id"].astype(str) == str(sku_id)].copy()
    if target_category is not None and "category" in sku.columns:
        sku = sku[sku["category"].astype(str) == str(target_category)].copy()
    for col, val in [("region", region), ("channel", channel), ("segment", segment)]:
        if val is not None and col in sku.columns:
            sku = sku[sku[col].astype(str) == str(val)].copy()
    if len(sku) == 0:
        raise ValueError("Нет данных по выбранному SKU.")

    extra_factor_cols = {"numeric": [], "categorical": []}
    if include_extra_factors:
        extra_factor_cols = infer_extra_factor_columns(sku)

    rows: List[Dict[str, Any]] = []
    for day, g in sku.groupby(pd.Grouper(key="date", freq="D")):
        if pd.isna(day):
            continue
        rec: Dict[str, Any] = {
            "date": pd.Timestamp(day),
            "sales": float(pd.to_numeric(g.get("quantity"), errors="coerce").fillna(0.0).sum()) if "quantity" in g.columns else float(len(g)),
            "revenue": float(pd.to_numeric(g.get("revenue"), errors="coerce").fillna(0.0).sum()) if "revenue" in g.columns else 0.0,
            "price": _wavg(g, "price"),
            "price_median": float(pd.to_numeric(g.get("price"), errors="coerce").median()) if "price" in g.columns else float("nan"),
        }
        if "discount" in g.columns:
            rec["discount"] = _wavg(g, "discount")
        if "cost" in g.columns:
            rec["cost"] = _wavg(g, "cost")
        if "freight_value" in g.columns:
            rec["freight_value"] = _wavg(g, "freight_value")
        if "promotion" in g.columns:
            rec["promotion"] = float(pd.to_numeric(g["promotion"], errors="coerce").fillna(0.0).max())
        if "rating" in g.columns:
            rec["review_score"] = float(pd.to_numeric(g["rating"], errors="coerce").dropna().iloc[-1]) if pd.to_numeric(g["rating"], errors="coerce").notna().any() else float("nan")
        if "reviews_count" in g.columns:
            rc = pd.to_numeric(g["reviews_count"], errors="coerce").dropna()
            rec["reviews_count"] = float(rc.iloc[-1]) if len(rc) else float("nan")
        if "stock" in g.columns:
            stock_vals = pd.to_numeric(g["stock"], errors="coerce").dropna()
            rec["stock"] = float(stock_vals.iloc[-1]) if len(stock_vals) else float("nan")
        for ctx in ["category", "region", "channel", "segment"]:
            if ctx in g.columns and g[ctx].notna().any():
                rec[ctx] = str(g[ctx].dropna().iloc[-1])
        if include_extra_factors:
            for col in extra_factor_cols["numeric"]:
                safe_col = f"factor__{col}"
                vals = pd.to_numeric(g[col], errors="coerce")
                rec[safe_col] = float(vals.mean()) if vals.notna().any() else float("nan")
            for col in extra_factor_cols["categorical"]:
                safe_col = f"factor__{col}"
                vals = g[col].dropna().astype(str)
                rec[safe_col] = str(vals.mode().iloc[0]) if len(vals) else "unknown"
        rows.append(rec)
    daily = pd.DataFrame(rows)
    if len(daily) == 0 or daily["date"].isna().all():
        raise ValueError("После агрегации не осталось валидных дат по SKU.")
    full_dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full_dates.merge(daily, on="date", how="left").sort_values("date").reset_index(drop=True)

    for c in ["sales", "revenue"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
    for c in ["price", "price_median", "freight_value", "discount", "cost", "stock", "review_score", "reviews_count", "promotion"]:
        if c not in daily.columns:
            daily[c] = np.nan
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    daily["promotion"] = daily["promotion"].fillna(0.0).clip(lower=0.0, upper=1.0)
    daily["discount"] = daily["discount"].fillna(0.0).clip(lower=0.0, upper=0.95)
    daily["price"] = daily["price"].ffill()
    daily["price_median"] = daily["price_median"].ffill()
    daily["cost"] = daily["cost"].ffill()
    daily["freight_value"] = daily["freight_value"].ffill()
    daily["stock"] = daily["stock"].ffill()
    daily["review_score"] = daily["review_score"].ffill()
    daily["reviews_count"] = daily["reviews_count"].ffill()

    daily["price"] = daily["price"].fillna(1.0).clip(lower=0.01)
    daily["price_median"] = daily["price_median"].fillna(daily["price"]).clip(lower=0.01)
    daily["freight_value"] = daily["freight_value"].fillna(0.0).clip(lower=0.0)
    daily["review_score"] = daily["review_score"].fillna(4.5)
    daily["reviews_count"] = daily["reviews_count"].fillna(0.0).clip(lower=0.0)
    daily["stock"] = daily["stock"].fillna(np.inf)
    daily["cost"] = daily["cost"].fillna(daily["price"] * 0.65).clip(lower=0.0)
    daily["net_unit_price"] = np.where(
        daily["sales"] > 0,
        daily["revenue"] / daily["sales"].replace(0, np.nan),
        daily["price"] * (1.0 - daily["discount"]),
    )
    daily["net_unit_price"] = pd.to_numeric(daily["net_unit_price"], errors="coerce").fillna(daily["price"] * (1.0 - daily["discount"])).clip(lower=0.01)
    if include_extra_factors:
        factor_cols = [c for c in daily.columns if str(c).startswith("factor__")]
        for c in factor_cols:
            if pd.api.types.is_numeric_dtype(daily[c]):
                daily[c] = pd.to_numeric(daily[c], errors="coerce").ffill().bfill()
            else:
                daily[c] = daily[c].astype("object").ffill().bfill().fillna("unknown")
    daily["category"] = sku["category"].mode().iloc[0] if "category" in sku.columns and not sku["category"].dropna().empty else "unknown"
    daily["sku_id"] = str(sku_id)
    for ctx in ["region", "channel", "segment"]:
        if ctx in sku.columns:
            daily[ctx] = str(sku[ctx].mode().iloc[0]) if not sku[ctx].dropna().empty else "unknown"
    return daily


def infer_extra_factor_columns(txn: pd.DataFrame) -> Dict[str, List[str]]:
    protected = {
        "date",
        "product_id",
        "sku_id",
        "quantity",
        "sales",
        "revenue",
        "profit",
        "margin",
        "price",
        "price_median",
        "net_unit_price",
        "discount",
        "cost",
        "freight_value",
        "promotion",
        "rating",
        "review_score",
        "reviews_count",
        "stock",
        "category",
        "region",
        "channel",
        "segment",
    }
    leak_keywords = [
        "order",
        "orders",
        "order_count",
        "units_sold",
        "sold_units",
        "demand",
        "target",
        "label",
        "actual",
        "forecast",
        "prediction",
        "predicted",
        "gmv",
        "turnover",
        "profit",
        "margin",
        "revenue",
        "sales",
        "quantity",
        "qty",
    ]
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in txn.columns:
        raw_name = str(col)
        name = raw_name.strip().lower()
        if name in protected:
            continue
        if any(k in name for k in leak_keywords):
            continue
        s = txn[col]
        non_null_share = float(s.notna().mean()) if len(s) else 0.0
        if non_null_share < 0.2:
            continue
        if pd.api.types.is_numeric_dtype(s):
            unique_count = int(pd.to_numeric(s, errors="coerce").nunique(dropna=True))
            if unique_count >= 2:
                numeric_cols.append(raw_name)
        else:
            unique_count = int(s.astype(str).nunique(dropna=True))
            if 2 <= unique_count <= 50:
                categorical_cols.append(raw_name)
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def build_feature_eligibility_report(df: pd.DataFrame) -> List[Dict[str, Any]]:
    registry = canonical_field_registry()
    report: List[Dict[str, Any]] = []
    for name, meta in registry.items():
        found = name in df.columns
        series = df[name] if found else pd.Series(dtype="float64")
        missing_share = float(series.isna().mean()) if found and len(df) else 1.0
        unique_count = int(series.nunique(dropna=True)) if found else 0
        usable_in_model = bool(found and meta.model_eligible and missing_share <= 0.4 and unique_count >= meta.min_variability)
        usable_in_scenario = bool(found and meta.scenario_allowed)
        reason = ""
        if not found:
            reason = "not_found"
        elif not usable_in_model:
            reason = "excluded_by_policy"
        report.append(
            {
                "factor": name,
                "dtype": meta.dtype,
                "role": meta.role,
                "found": found,
                "missing_share": missing_share,
                "unique_count": unique_count,
                "usable_in_model": usable_in_model,
                "usable_in_scenario": usable_in_scenario,
                "reason_if_excluded": reason,
            }
        )
    return report
