from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CanonicalField:
    name: str
    aliases: List[str]
    required: bool = False
    dtype: str = "str"
    description: str = ""
    role: str = "context"
    aggregation: str = "last"
    scenario_allowed: bool = False
    min_variability: int = 1
    feature_policy: str = "raw"
    future_policy: str = "known_or_override"
    model_eligible: bool = True


CANONICAL_FIELDS: List[CanonicalField] = [
    CanonicalField("date", ["date", "timestamp", "datetime", "order_date", "created_at", "order_purchase_timestamp"], required=True, dtype="datetime", description="Дата/время операции", role="metadata", aggregation="last", model_eligible=False),
    CanonicalField("product_id", ["product_id", "sku", "item_id", "product", "item", "article"], required=True, dtype="str", description="SKU / товар", role="metadata", aggregation="last", model_eligible=False),
    CanonicalField("category", ["category", "product_category_name", "category_name", "product_category"], required=False, dtype="str", description="Категория", role="context_selector", aggregation="mode", model_eligible=False),
    CanonicalField("price", ["price", "unit_price", "sale_price", "selling_price"], required=True, dtype="float", description="Цена продажи", role="controllable", aggregation="weighted_mean", scenario_allowed=True, feature_policy="raw_lag1_ma7"),
    CanonicalField("quantity", ["quantity", "qty", "sales", "units", "units_sold"], required=False, dtype="float", description="Количество", role="target", aggregation="sum", model_eligible=False),
    CanonicalField("revenue", ["revenue", "gmv", "sales_amount", "turnover"], required=False, dtype="float", description="Выручка", role="economics_only", aggregation="sum", model_eligible=False),
    CanonicalField("cost", ["cost", "cogs", "unit_cost", "purchase_cost"], required=False, dtype="float", description="Себестоимость", role="economics_only", aggregation="weighted_mean", scenario_allowed=True, model_eligible=False),
    CanonicalField("discount", ["discount", "discount_rate", "discount_pct", "promo_rate", "discount_value", "discount_amount", "promo_discount"], required=False, dtype="float", description="Скидка", role="controllable", aggregation="weighted_mean", scenario_allowed=True, feature_policy="raw_lag1_ma7"),
    CanonicalField("freight_value", ["freight", "shipping", "shipping_cost", "freight_value", "delivery_cost"], required=False, dtype="float", description="Логистика", role="controllable", aggregation="weighted_mean", scenario_allowed=True, feature_policy="raw_lag1_ma7"),
    CanonicalField("stock", ["stock", "inventory", "stock_qty", "inventory_level"], required=False, dtype="float", description="Остаток", role="constraint", aggregation="last", scenario_allowed=True, model_eligible=False),
    CanonicalField("promotion", ["promotion", "promo", "promo_flag", "is_promo", "campaign"], required=False, dtype="float", description="Промо-фактор", role="controllable", aggregation="max", scenario_allowed=True, feature_policy="raw_lag1_share7"),
    CanonicalField("rating", ["rating", "review_score", "score", "stars"], required=False, dtype="float", description="Рейтинг", role="context", aggregation="last", feature_policy="raw"),
    CanonicalField("reviews_count", ["reviews", "reviews_count", "review_count", "num_reviews", "n_reviews"], required=False, dtype="float", description="Кол-во отзывов", role="context", aggregation="last", feature_policy="raw"),
    CanonicalField("region", ["region", "geo", "state", "area"], required=False, dtype="str", description="Регион", role="context_selector", aggregation="mode", model_eligible=False),
    CanonicalField("channel", ["channel", "sales_channel", "platform"], required=False, dtype="str", description="Канал", role="context_selector", aggregation="mode", model_eligible=False),
    CanonicalField("segment", ["segment", "customer_segment", "tier"], required=False, dtype="str", description="Сегмент", role="context_selector", aggregation="mode", model_eligible=False),
]


def canonical_required_fields() -> List[str]:
    return [f.name for f in CANONICAL_FIELDS if f.required]


def canonical_alias_map() -> Dict[str, List[str]]:
    return {f.name: f.aliases for f in CANONICAL_FIELDS}


def canonical_field_registry() -> Dict[str, CanonicalField]:
    return {f.name: f for f in CANONICAL_FIELDS}


def scenario_allowed_fields() -> List[str]:
    return [f.name for f in CANONICAL_FIELDS if f.scenario_allowed]
