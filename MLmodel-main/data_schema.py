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


CANONICAL_FIELDS: List[CanonicalField] = [
    CanonicalField("date", ["date", "timestamp", "datetime", "order_date", "created_at", "order_purchase_timestamp"], required=True, dtype="datetime", description="Дата/время операции"),
    CanonicalField("product_id", ["product_id", "sku", "item_id", "product", "item", "article"], required=True, dtype="str", description="SKU / товар"),
    CanonicalField("category", ["category", "product_category_name", "category_name", "product_category"], required=False, dtype="str", description="Категория"),
    CanonicalField("price", ["price", "unit_price", "sale_price", "selling_price"], required=True, dtype="float", description="Цена продажи"),
    CanonicalField("quantity", ["quantity", "qty", "sales", "units", "units_sold"], required=False, dtype="float", description="Количество"),
    CanonicalField("revenue", ["revenue", "gmv", "sales_amount", "turnover"], required=False, dtype="float", description="Выручка"),
    CanonicalField("cost", ["cost", "cogs", "unit_cost", "purchase_cost"], required=False, dtype="float", description="Себестоимость"),
    CanonicalField("discount", ["discount", "discount_value", "discount_amount", "promo_discount"], required=False, dtype="float", description="Скидка"),
    CanonicalField("freight_value", ["freight", "shipping", "shipping_cost", "freight_value", "delivery_cost"], required=False, dtype="float", description="Логистика"),
    CanonicalField("stock", ["stock", "inventory", "stock_qty", "inventory_level"], required=False, dtype="float", description="Остаток"),
    CanonicalField("promotion", ["promotion", "promo", "is_promo", "campaign"], required=False, dtype="float", description="Промо-фактор"),
    CanonicalField("rating", ["rating", "review_score", "score", "stars"], required=False, dtype="float", description="Рейтинг"),
    CanonicalField("reviews_count", ["reviews", "reviews_count", "n_reviews"], required=False, dtype="float", description="Кол-во отзывов"),
    CanonicalField("region", ["region", "geo", "state", "area"], required=False, dtype="str", description="Регион"),
    CanonicalField("channel", ["channel", "sales_channel", "platform"], required=False, dtype="str", description="Канал"),
    CanonicalField("segment", ["segment", "customer_segment", "tier"], required=False, dtype="str", description="Сегмент"),
]


def canonical_required_fields() -> List[str]:
    return [f.name for f in CANONICAL_FIELDS if f.required]


def canonical_alias_map() -> Dict[str, List[str]]:
    return {f.name: f.aliases for f in CANONICAL_FIELDS}
