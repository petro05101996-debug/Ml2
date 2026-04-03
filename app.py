from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_analysis_runner import run_analysis_from_context
from app_domain import (
    PRESET_AGGRESSIVE_INCREASE,
    PRESET_CAUTIOUS_INCREASE,
    PRESET_COST_FREIGHT_STRESS,
    PRESET_KEEP_CURRENT,
    PRESET_LIMITED_STOCK,
    PRESET_LOWER_FOR_VOLUME,
    PRESET_PROMO_PUSH,
    REQUIRED_PRESET_KEYS,
    build_default_scenario_inputs,
    build_seller_scenario_presets,
)
from app_presenters import build_main_decision_text
from data_adapter import build_auto_mapping, normalize_transactions
from data_schema import CANONICAL_FIELDS
from pricing_core import (
    CONFIG,
    OBJECTIVE_HINTS,
    OBJECTIVE_LABEL_TO_MODE,
    generate_explanation,
    run_what_if_projection,
)
from pricing_core.core import *  # noqa: F401,F403
from ui_docs import render_docs
from ui_overview import render_overview
from ui_shell import apply_enterprise_styles, render_navigation
from what_if import build_sensitivity_grid, run_scenario_set

st.set_page_config(page_title="Студия ценовой аналитики", layout="wide", page_icon="📊")

for key, default in {
    "results": None,
    "what_if_result": None,
    "scenario_table": None,
    "sensitivity_df": None,
    "what_if_last_run_at": None,
    "scenario_inputs_state": None,
    "active_page": "Обзор",
    "selected_category_for_results": None,
    "selected_sku_for_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

HORIZON_OPTIONS = [7, 14, 30, 60, 90, 180, 360]
LOAD_MODE_UNIVERSAL = "Универсальный CSV"
LOAD_MODE_OLIST = "Наследуемый Olist (3 CSV)"

UI_TEXT_RU = {
    "setup_title": "Настройка",
    "setup_subtitle": "Пошаговая настройка для надёжного сценарного анализа.",
    "data_mode": "Режим данных",
    "orders_file": "Файл заказов (Orders) *",
    "items_file": "Файл позиций заказа (Items) *",
    "products_file": "Файл каталога товаров (Products) *",
    "reviews_file": "Файл отзывов (Reviews, опционально)",
    "universal_file": "Файл транзакций (универсальный CSV) *",
    "need_universal": "Сначала загрузите и сопоставьте универсальный CSV.",
    "need_olist": "Загрузите обязательные файлы: Orders, Items и Products.",
}


def _horizon_help_text() -> str:
    return "7–30 дней — краткосрочно, 60–90 — среднесрочно, 180+ — долгосрочное планирование."




def _render_excel_download_link(file_name: str, excel_payload: Any) -> None:
    data_bytes = excel_payload.getvalue() if hasattr(excel_payload, "getvalue") else excel_payload
    if not isinstance(data_bytes, (bytes, bytearray)):
        st.warning("Не удалось подготовить Excel для скачивания.")
        return
    st.download_button(
        label="Скачать Excel-отчёт",
        data=bytes(data_bytes),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=f"excel_download_{file_name}",
    )

def _base_plotly_layout(title: str) -> Dict[str, Any]:
    return dict(title=title, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20))


def render_setup_page() -> Dict[str, Any]:
    st.markdown(f'<div class="page-title">{UI_TEXT_RU["setup_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">{UI_TEXT_RU["setup_subtitle"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card"><strong>Шаг 1 · Загрузка данных</strong><br/><span class="muted">Качество данных напрямую влияет на уверенность и качество рекомендации.</span></div>', unsafe_allow_html=True)
    load_mode = st.radio(UI_TEXT_RU["data_mode"], [LOAD_MODE_UNIVERSAL, LOAD_MODE_OLIST], horizontal=True)
    orders_file = items_file = products_file = reviews_file = None
    universal_file = None

    if load_mode == LOAD_MODE_UNIVERSAL:
        universal_file = st.file_uploader(UI_TEXT_RU["universal_file"], type=["csv"], key="universal_file")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            orders_file = st.file_uploader(UI_TEXT_RU["orders_file"], type=["csv"], key="orders_file")
        with c2:
            items_file = st.file_uploader(UI_TEXT_RU["items_file"], type=["csv"], key="items_file")
        with c3:
            products_file = st.file_uploader(UI_TEXT_RU["products_file"], type=["csv"], key="products_file")
        reviews_file = st.file_uploader(UI_TEXT_RU["reviews_file"], type=["csv"], key="reviews_file")

    raw_for_select = None
    universal_txn = None
    universal_mapping: Dict[str, Optional[str]] = {}
    orders_col_map: Dict[str, Optional[str]] = {}
    items_col_map: Dict[str, Optional[str]] = {}
    products_col_map: Dict[str, Optional[str]] = {}
    reviews_col_map: Dict[str, Optional[str]] = {}

    st.markdown('<div class="step-card"><strong>Шаг 2 · Проверка сопоставления</strong><br/><span class="muted">Сначала сопоставьте обязательные поля. Дополнительные параметры — вторично.</span></div>', unsafe_allow_html=True)

    if load_mode == LOAD_MODE_UNIVERSAL and universal_file is not None:
        preview = pd.read_csv(universal_file)
        auto_map = build_auto_mapping(list(preview.columns))
        required, optional = st.columns(2)
        with required:
            st.markdown("**Обязательные поля**")
            for f in [x for x in CANONICAL_FIELDS if x.required]:
                choices = ["<skip>"] + list(preview.columns)
                guessed = auto_map.get(f.name)
                idx = choices.index(guessed) if guessed in choices else 0
                selected = st.selectbox(f"{f.name} *", choices, index=idx, key=f"map_u_req_{f.name}")
                universal_mapping[f.name] = None if selected == "<skip>" else selected
        with optional:
            st.markdown("**Дополнительные поля**")
            for f in [x for x in CANONICAL_FIELDS if not x.required]:
                choices = ["<skip>"] + list(preview.columns)
                guessed = auto_map.get(f.name)
                idx = choices.index(guessed) if guessed in choices else 0
                selected = st.selectbox(f.name, choices, index=idx, key=f"map_u_opt_{f.name}")
                universal_mapping[f.name] = None if selected == "<skip>" else selected

        universal_txn, quality = normalize_transactions(preview, universal_mapping)
        for w in quality.get("warnings", []):
            st.warning(w)
        if quality.get("errors"):
            st.error("; ".join(quality["errors"]))
        else:
            raw_for_select = universal_txn.copy().rename(columns={"category": "product_category_name"})
            st.success(f"Нормализация завершена: {len(raw_for_select):,} строк")

    elif load_mode != LOAD_MODE_UNIVERSAL and all([orders_file, items_file, products_file]):
        orders = pd.read_csv(orders_file)
        items = pd.read_csv(items_file)
        products = pd.read_csv(products_file)
        reviews = pd.read_csv(reviews_file) if reviews_file else pd.DataFrame()

        def pick(label: str, cols: List[str], aliases: List[str], key: str, required: bool = True) -> Optional[str]:
            guessed = _suggest_column(cols, aliases)
            choices = cols if required else ["<skip>"] + cols
            idx = choices.index(guessed) if guessed in choices else 0
            val = st.selectbox(label, choices, index=idx, key=key)
            return None if (not required and val == "<skip>") else val

        req1, req2 = st.columns(2)
        with req1:
            st.markdown("**Обязательные поля**")
            orders_col_map = {
                "order_id": pick("Orders: ID заказа *", list(orders.columns), ["order_id", "orderid"], "m1"),
                "order_purchase_timestamp": pick("Orders: дата *", list(orders.columns), ["order_purchase_timestamp", "order_date", "date"], "m2"),
            }
            items_col_map = {
                "order_id": pick("Items: ID заказа *", list(items.columns), ["order_id", "orderid"], "m3"),
                "product_id": pick("Items: SKU *", list(items.columns), ["product_id", "sku"], "m4"),
                "price": pick("Items: цена *", list(items.columns), ["price", "unit_price"], "m5"),
            }
        with req2:
            st.markdown("**Дополнительные/второстепенные поля**")
            products_col_map = {
                "product_id": pick("Products: SKU *", list(products.columns), ["product_id", "sku"], "m6"),
                "product_category_name": pick("Products: категория *", list(products.columns), ["product_category_name", "category"], "m7"),
            }
            items_col_map.update({
                "freight_value": pick("Items: логистика", list(items.columns), ["freight_value", "shipping_cost"], "m8", False),
                "order_item_id": pick("Items: ID позиции", list(items.columns), ["order_item_id", "line_id"], "m9", False),
            })
            reviews_col_map = {"order_id": pick("Reviews: ID заказа", list(reviews.columns), ["order_id"], "m10", False), "review_score": pick("Reviews: оценка", list(reviews.columns), ["review_score", "rating"], "m11", False)} if len(reviews) else {}

        orders_norm = _rename_with_mapping(orders, orders_col_map)
        items_norm = _rename_with_mapping(items, items_col_map)
        products_norm = _rename_with_mapping(products, products_col_map)
        reviews_norm = _rename_with_mapping(reviews, reviews_col_map) if len(reviews) else pd.DataFrame()
        if "order_item_id" not in items_norm.columns:
            items_norm["order_item_id"] = np.arange(1, len(items_norm) + 1)
        if "freight_value" not in items_norm.columns:
            items_norm["freight_value"] = 0.0
        raw_for_select = build_raw_frame(orders_norm, items_norm, products_norm, reviews_norm)
        st.success(f"Предпросмотр данных готов: {len(raw_for_select):,} строк")

    st.markdown('<div class="step-card"><strong>Шаг 3 · Выбор категории и SKU</strong><br/><span class="muted">Выберите конкретный товар для принятия решения.</span></div>', unsafe_allow_html=True)
    target_category, target_sku = None, None
    if raw_for_select is not None and len(raw_for_select):
        category_col = "product_category_name" if "product_category_name" in raw_for_select.columns else "category"
        categories = sorted(raw_for_select[category_col].astype(str).dropna().unique())
        target_category = st.selectbox("Категория *", categories) if categories else None
        sku_search = st.text_input("Поиск SKU", placeholder="Введите SKU...")
        if target_category is not None:
            skus = raw_for_select[raw_for_select[category_col].astype(str) == str(target_category)]["product_id"].astype(str).dropna().unique().tolist()
            skus = sorted([s for s in skus if sku_search.lower() in s.lower()]) if sku_search else sorted(skus)
            target_sku = st.selectbox("SKU *", skus) if skus else None

    st.markdown('<div class="step-card"><strong>Шаг 4 · Параметры анализа</strong><br/><span class="muted">Задайте цель, горизонт и уровень осторожности.</span></div>', unsafe_allow_html=True)
    objective_labels = list(OBJECTIVE_LABEL_TO_MODE.keys())
    selected_objective_label = st.selectbox("Цель оптимизации", objective_labels)
    st.caption(OBJECTIVE_HINTS[selected_objective_label])
    objective_mode = OBJECTIVE_LABEL_TO_MODE[selected_objective_label]
    forecast_horizon_days = st.select_slider("Горизонт, дней", options=HORIZON_OPTIONS, value=30, help=_horizon_help_text())
    caution_level = st.selectbox("Уровень осторожности", ["Низкий", "Средний", "Высокий"], index=1)

    with st.expander("Расширенные настройки", expanded=False):
        show_risk = st.checkbox("Показывать риск/уверенность", value=True)
        objective_weights_override = None
        if st.checkbox("Ручная настройка весов цели", value=False):
            objective_weights_override = {
                "profit": float(st.slider("Прибыль", 0.0, 2.0, 0.8, 0.1)),
                "revenue": float(st.slider("Выручка", 0.0, 2.0, 0.7, 0.1)),
                "volume": float(st.slider("Объём", 0.0, 2.0, 0.7, 0.1)),
                "margin": float(st.slider("Маржа", 0.0, 2.0, 0.7, 0.1)),
                "risk": float(st.slider("Риск", 0.0, 2.0, 0.7, 0.1)),
            }

    st.markdown('<div class="step-card"><strong>Шаг 5 · Запуск анализа</strong><br/><span class="muted">Запустите расчёт и откройте результаты в формате бизнес-решения.</span></div>', unsafe_allow_html=True)
    run_requested = st.button("Запустить ценовой анализ", type="primary", use_container_width=True)

    _, objective_mode, objective_warning = resolve_objective_weights(objective_mode)

    return {
        "load_mode": load_mode,
        "orders_file": orders_file,
        "items_file": items_file,
        "products_file": products_file,
        "reviews_file": reviews_file,
        "universal_file": universal_file,
        "universal_txn": universal_txn,
        "target_category": target_category,
        "target_sku": target_sku,
        "objective_mode": objective_mode,
        "objective_warning": objective_warning,
        "objective_weights_override": objective_weights_override if "objective_weights_override" in locals() else None,
        "forecast_horizon_days": int(forecast_horizon_days),
        "caution_level": caution_level,
        "show_risk": bool(show_risk if "show_risk" in locals() else True),
        "run_requested": run_requested,
        "orders_col_map": orders_col_map,
        "items_col_map": items_col_map,
        "products_col_map": products_col_map,
        "reviews_col_map": reviews_col_map,
    }


def render_scenario_lab(r: Dict[str, Any]) -> None:
    st.markdown('<div class="page-title">Лаборатория сценариев</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Сценарная рабочая зона: быстрые пресеты и расширенные настройки.</div>', unsafe_allow_html=True)

    presets = build_seller_scenario_presets(float(r["current_price"]), r["_trained_bundle"]["base_ctx"], int(r.get("forecast_horizon_days", 30)))
    missing_presets = [k for k in REQUIRED_PRESET_KEYS if k not in presets]
    if missing_presets:
        st.warning(f"Отсутствуют обязательные пресеты: {', '.join(missing_presets)}. Используются доступные варианты.")

    quick_presets = {
        "Сохранить текущую цену": presets.get(PRESET_KEEP_CURRENT, list(presets.values())[0]),
        "Осторожное повышение": presets.get(PRESET_CAUTIOUS_INCREASE, list(presets.values())[0]),
        "Агрессивное повышение": presets.get(PRESET_AGGRESSIVE_INCREASE, list(presets.values())[0]),
        "Снижение цены ради объёма": presets.get(PRESET_LOWER_FOR_VOLUME, list(presets.values())[0]),
        "Промо-ускорение": presets.get(PRESET_PROMO_PUSH, list(presets.values())[0]),
        "Стресс: рост себестоимости/логистики": presets.get(PRESET_COST_FREIGHT_STRESS, list(presets.values())[0]),
        "Сценарий ограниченного запаса": presets.get(PRESET_LIMITED_STOCK, list(presets.values())[0]),
    }
    selected_preset = st.selectbox("Быстрые бизнес-пресеты", list(quick_presets.keys()))
    base_preset = quick_presets[selected_preset]

    p1, p2, p3, p4 = st.columns(4)
    price = p1.number_input("Цена", min_value=0.01, value=float(base_preset.get("price", r["current_price"])), step=1.0)
    demand = p2.slider("Множитель спроса", 0.7, 1.3, float(base_preset.get("demand_multiplier", 1.0)), 0.05)
    freight = p3.slider("Множитель логистики", 0.5, 1.5, float(base_preset.get("freight_multiplier", 1.0)), 0.05)
    horizon = p4.select_slider("Горизонт", options=HORIZON_OPTIONS, value=int(base_preset.get("horizon_days", 30)))

    with st.expander("Расширенный режим", expanded=False):
        discount = st.slider("Скидка", 0.0, 0.95, float(base_preset.get("discount", 0.0)), 0.01)
        promo = st.slider("Промо", 0.0, 1.0, float(base_preset.get("promotion", 0.0)), 0.05)
        stock = st.number_input("Лимит запаса", min_value=0.0, value=float(base_preset.get("stock_cap", r["_trained_bundle"]["base_ctx"].get("stock", 0.0))), step=1.0)
        rating = st.slider("Рейтинг", 0.0, 5.0, float(r["_trained_bundle"]["base_ctx"].get("review_score", 4.5)), 0.1)
        reviews_count = st.number_input("Количество отзывов", min_value=0.0, value=float(r["_trained_bundle"]["base_ctx"].get("reviews_count", 0.0)), step=1.0)
        cost = st.slider("Множитель себестоимости", 0.7, 1.3, float(base_preset.get("cost_multiplier", 1.0)), 0.05)

    st.caption(f"Предпросмотр сценария: {selected_preset} · Цена {price:,.2f} · Спрос {demand:.2f} · Логистика {freight:.2f} · Горизонт {horizon} дней")

    w = run_what_if_projection(
        r["_trained_bundle"],
        manual_price=float(price),
        freight_multiplier=float(freight),
        demand_multiplier=float(demand),
        cost_multiplier=float(cost if "cost" in locals() else base_preset.get("cost_multiplier", 1.0)),
        horizon_days=int(horizon),
        stock_cap=float(stock if "stock" in locals() else base_preset.get("stock_cap", 0.0)),
        scenario={"name": "scenario_lab", "mode": "manual", "horizon_days": int(horizon), "factors": {"price": float(price), "discount": float(discount if "discount" in locals() else 0.0), "promotion": float(promo if "promo" in locals() else 0.0), "stock": float(stock if "stock" in locals() else 0.0), "rating": float(rating if "rating" in locals() else 4.5), "reviews_count": float(reviews_count if "reviews_count" in locals() else 0.0), "cost_multiplier": float(cost if "cost" in locals() else 1.0)}},
    )
    st.session_state.what_if_result = w

    base_revenue = float(r.get("current_revenue", 0.0)) or float((r["forecast_current"]["price"] * r["forecast_current"]["pred_sales"]).sum())
    base_volume = float(r.get("current_volume", 0.0)) or float(r["forecast_current"]["pred_sales"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Изменение прибыли", f"{float(w['profit_total'] - r['current_profit']):+,.0f} ₽")
    c2.metric("Изменение выручки", f"{float(w['revenue_total'] - base_revenue):+,.0f} ₽")
    c3.metric("Изменение объёма", f"{float(w['demand_total'] - base_volume):+,.1f}")


def render_results_page(r: Dict[str, Any]) -> None:
    st.markdown('<div class="page-title">Результаты</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Слой результатов, ориентированный на решение.</div>', unsafe_allow_html=True)
    engine = r.get("analysis_engine", "unknown")
    engine_ver = r.get("analysis_engine_version", "unknown")
    route = r.get("analysis_route", "unknown")
    load_mode = r.get("ui_load_mode", "unknown")
    engine_line = f"Engine: {engine} | Version: {engine_ver} | Route: {route} | Load mode: {load_mode}"
    if hasattr(st, "caption"):
        st.caption(f"Технический режим анализа: {engine_line}")
    else:
        st.markdown(f"Технический режим анализа: {engine_line}")
    decision = build_main_decision_text(r)
    biz = r.get("business_recommendation", {})
    structured = biz.get("structured", {}) if isinstance(biz, dict) else {}
    confidence_score = int(round(float(r.get("_trained_bundle", {}).get("confidence", 0.0)) * 100))

    base_revenue = float(r.get("current_revenue", 0.0)) or float((r["forecast_current"]["price"] * r["forecast_current"]["pred_sales"]).sum())
    recommended_revenue = float(r.get("best_revenue", base_revenue))
    revenue_delta = float(structured.get("expected_revenue_change", recommended_revenue - base_revenue) or 0.0)
    base_volume = float(r.get("current_volume", 0.0)) or float(r["forecast_current"]["pred_sales"].sum())
    volume_delta = float(structured.get("expected_volume_change", float(r.get("best_volume", base_volume)) - base_volume) or 0.0)

    st.markdown('<div class="enterprise-card primary-card">', unsafe_allow_html=True)
    st.markdown(f"### Исполнительное решение: {decision.get('action', '—')}")
    st.markdown('<div class="kpi-strip">', unsafe_allow_html=True)
    for label, value in [
        ("Рекомендованная цена", decision["recommended_price"]),
        ("Изменение цены", decision["price_delta"]),
        ("Ожидаемая прибыль", decision["profit_delta"]),
        ("Ожидаемая выручка", f"{revenue_delta:+,.0f} ₽"),
        ("Ожидаемый объём", f"{volume_delta:+,.1f}"),
        ("Уровень риска", str(structured.get("risk_level", biz.get("risk_text", "Оценка риска")))),
        ("Уверенность", f"{confidence_score}/100"),
        ("Режим внедрения", decision.get("implementation_mode", "pilot")),
    ]:
        st.markdown(f'<div class="kpi"><label>{label}</label><strong>{value}</strong></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    why1, why2 = st.columns(2)
    why1.markdown(f'<div class="enterprise-card secondary-card"><div class="card-title">Почему это решение</div><div class="muted">{biz.get("seller_friendly_reason", biz.get("plain_reason", "Решение балансирует ожидаемую прибыль и риск для текущих условий."))}</div></div>', unsafe_allow_html=True)
    why2.markdown(f'<div class="enterprise-card secondary-card"><div class="card-title">Главное ограничение / негативный сценарий</div><div class="muted">{biz.get("why_not_more", "Эффект может ослабнуть, если спрос отреагирует на цену сильнее ожидаемого.")}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="enterprise-card secondary-card"><div class="card-title">Сравнение сценариев</div></div>', unsafe_allow_html=True)
    default_inputs = build_default_scenario_inputs(float(r["current_price"]), int(r.get("forecast_horizon_days", 30)), r["_trained_bundle"]["base_ctx"])
    if st.button("Запустить: текущий vs рекомендованный vs консервативный", use_container_width=True):
        st.session_state.scenario_table = run_scenario_set(r["_trained_bundle"], default_inputs[:3], run_what_if_projection)
    if st.session_state.get("scenario_table") is not None:
        st.dataframe(st.session_state.scenario_table, use_container_width=True)

    rollout = biz.get("rollout_plan", structured.get("rollout", {}))
    st.markdown(
        f'<div class="enterprise-card secondary-card"><div class="card-title">Мониторинг и план внедрения</div><ul><li><b>Что мониторить:</b> {", ".join(rollout.get("monitor_metrics", biz.get("what_to_monitor", ["profit", "volume", "conversion"])))}</li><li><b>Длительность теста:</b> {rollout.get("test_plan", biz.get("seller_friendly_next_step", "Проведите пилот на одном сегменте/канале."))}</li><li><b>Условие успеха:</b> {rollout.get("success_rule", biz.get("success_condition", "Рост прибыли при контролируемом изменении объёма."))}</li><li><b>Условие отката:</b> {rollout.get("rollback_rule", biz.get("rollback_condition", "Откат при выходе KPI за допустимые границы."))}</li><li><b>Когда пересмотреть:</b> {int(rollout.get("review_after_days", biz.get("review_after_days", 7)))} дней</li></ul></div>',
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2)
    r1.markdown(
        f'<div class="enterprise-card secondary-card"><div class="card-title">Когда не применять рекомендацию</div><div class="muted">{biz.get("when_not_to_apply", "Не применять без пилота, если меняются канал, сезонность или структура ассортимента одновременно.")}</div></div>',
        unsafe_allow_html=True,
    )
    r2.markdown(
        f'<div class="enterprise-card secondary-card"><div class="card-title">Важные допущения</div><div class="muted">{biz.get("key_assumptions", "Оценка основана на исторических данных, текущей эластичности и стабильности внешних условий.")}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="enterprise-card flat-card"><div class="card-title">Детальная аналитика</div></div>', unsafe_allow_html=True)
    fig_profit = px.line(r["profit_curve"], x="price", y="adjusted_profit", template="plotly_dark")
    fig_profit.add_vline(x=r["current_price"], line_dash="dash", line_color="#ffffff")
    fig_profit.add_vline(x=r["best_price"], line_color="#5da0ff")
    fig_profit.update_layout(**_base_plotly_layout("Risk-adjusted profit vs Цена"))
    st.plotly_chart(fig_profit, use_container_width=True)

    explanation = generate_explanation(r, data_quality=r.get("data_quality", {}))
    with st.expander("Расширенная аналитика", expanded=False):
        st.write(explanation.get("summary", ""))
        st.dataframe(r.get("holdout_metrics", pd.DataFrame()), use_container_width=True)
        if st.button("Запустить карту чувствительности", use_container_width=True):
            st.session_state.sensitivity_df = build_sensitivity_grid(r["_trained_bundle"], base_price=float(r["current_price"]), runner=run_what_if_projection)
        if st.session_state.get("sensitivity_df") is not None:
            heat = px.density_heatmap(st.session_state.sensitivity_df, x="price", y="demand_multiplier", z="profit", template="plotly_dark")
            heat.update_layout(**_base_plotly_layout("Чувствительность: цена × спрос"))
            st.plotly_chart(heat, use_container_width=True)


def maybe_run_analysis(ctx: Dict[str, Any]) -> None:
    if not ctx.get("run_requested"):
        return
    if ctx.get("objective_warning"):
        st.warning(ctx["objective_warning"])
    if ctx.get("load_mode") == LOAD_MODE_UNIVERSAL and ctx.get("universal_txn") is None:
        st.error(UI_TEXT_RU["need_universal"])
        return
    if ctx.get("load_mode") != LOAD_MODE_UNIVERSAL and not (ctx.get("orders_file") and ctx.get("items_file") and ctx.get("products_file")):
        st.error(UI_TEXT_RU["need_olist"])
        return
    if ctx.get("target_category") is None or ctx.get("target_sku") is None:
        st.error("Выберите категорию и SKU.")
        return
    with st.spinner("Выполняется ценовой анализ..."):
        results = run_analysis_from_context(ctx)
        results["show_risk"] = bool(ctx.get("show_risk", True))
        st.session_state.results = results
        st.session_state.selected_category_for_results = ctx.get("target_category")
        st.session_state.selected_sku_for_results = ctx.get("target_sku")
        st.session_state.active_page = "Результаты"
        st.rerun()


apply_enterprise_styles()
active_page = render_navigation()

if st.session_state.results is not None:
    top1, top2 = st.columns([1, 1])
    with top1:
        if st.button("Новая загрузка", use_container_width=True):
            for k in ["results", "what_if_result", "scenario_table", "sensitivity_df", "scenario_inputs_state"]:
                st.session_state[k] = None
            st.session_state.active_page = "Настройка"
            st.rerun()
    with top2:
        engine = st.session_state.results.get("analysis_engine", "unknown")
        sku = st.session_state.results.get("sku", st.session_state.get("selected_sku_for_results", "report"))
        file_name = f"pricing_report_{sku}_{engine}.xlsx"
        excel_payload = st.session_state.results.get("excel_buffer")
        if excel_payload is None:
            st.warning("Excel-отчёт недоступен для этого запуска. Повторите анализ.")
        else:
            _render_excel_download_link(file_name=file_name, excel_payload=excel_payload)

if active_page == "Обзор":
    render_overview()
elif active_page == "Настройка":
    context = render_setup_page()
    maybe_run_analysis(context)
elif active_page == "Лаборатория сценариев":
    if st.session_state.results is None:
        st.info("Сначала пройдите настройку, чтобы открыть Лабораторию сценариев.")
    else:
        render_scenario_lab(st.session_state.results)
elif active_page == "Результаты":
    if st.session_state.results is None:
        st.info("Сначала пройдите настройку, чтобы увидеть Результаты.")
    else:
        render_results_page(st.session_state.results)
else:
    render_docs()

st.caption("Студия ценовой аналитики • Streamlit")
