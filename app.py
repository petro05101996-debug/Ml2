from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
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
from data_adapter import build_auto_mapping, normalize_transactions, objective_to_weights
from data_schema import CANONICAL_FIELDS
from pricing_core import CONFIG, OBJECTIVE_HINTS, OBJECTIVE_LABEL_TO_MODE, generate_explanation, run_v1_what_if_projection
from ui_docs import render_docs
from ui_overview import render_overview
from ui_shell import apply_enterprise_styles, render_navigation
from what_if import build_sensitivity_grid, run_scenario_set

st.set_page_config(page_title="Студия ценовой аналитики", layout="wide", page_icon="📊")
for key, default in {"results": None, "what_if_result": None, "scenario_table": None, "sensitivity_df": None, "active_page": "Обзор"}.items():
    if key not in st.session_state:
        st.session_state[key] = default

HORIZON_OPTIONS = [7, 14, 30, 60, 90, 180, 360]
LOAD_MODE_UNIVERSAL = "Универсальный CSV"


def resolve_objective_weights(mode: str):
    try:
        return objective_to_weights(mode), mode, None
    except ValueError:
        fallback = "balanced_mode"
        return objective_to_weights(fallback), fallback, "Неизвестный режим оптимизации. Применён безопасный режим: «Сбалансированный режим»."


def _base_plotly_layout(title: str) -> Dict[str, Any]:
    return dict(title=title, template="plotly_dark")


def _build_target_selection(universal_txn: Optional[pd.DataFrame]) -> Dict[str, Optional[str]]:
    if universal_txn is None or universal_txn.empty:
        return {"target_category": None, "target_sku": None, "category_options": [], "sku_options": []}

    category_series = universal_txn.get("category", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    category_options = sorted([c for c in category_series.unique().tolist() if c])
    if not category_options:
        category_options = ["unknown"]

    target_category = st.selectbox("Категория *", category_options)
    scoped = universal_txn[universal_txn["category"].astype(str) == str(target_category)] if "category" in universal_txn.columns else universal_txn
    sku_series = scoped.get("product_id", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    sku_options = sorted([sku for sku in sku_series.unique().tolist() if sku])
    if not sku_options:
        sku_options = sorted([sku for sku in universal_txn.get("product_id", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique().tolist() if sku])
    target_sku = st.selectbox("SKU *", sku_options) if sku_options else None
    return {"target_category": target_category or None, "target_sku": target_sku or None, "category_options": category_options, "sku_options": sku_options}


def render_setup_page() -> Dict[str, Any]:
    st.markdown("### Настройка")
    universal_file = st.file_uploader("Файл транзакций (универсальный CSV) *", type=["csv"], key="universal_file")
    universal_txn = None
    if universal_file is not None:
        preview = pd.read_csv(universal_file)
        auto_map = build_auto_mapping(list(preview.columns))
        mapping: Dict[str, Optional[str]] = {}
        for f in CANONICAL_FIELDS:
            choices = ["<skip>"] + list(preview.columns)
            guessed = auto_map.get(f.name)
            idx = choices.index(guessed) if guessed in choices else 0
            selected = st.selectbox(f"{f.name}{' *' if f.required else ''}", choices, index=idx)
            mapping[f.name] = None if selected == "<skip>" else selected
        universal_txn, quality = normalize_transactions(preview, mapping)
        for w in quality.get("warnings", []):
            st.warning(w)

    target_selection = _build_target_selection(universal_txn)
    if target_selection["target_category"] is None and target_selection["target_sku"] is None:
        target_category = st.text_input("Категория *", value="")
        target_sku = st.text_input("SKU *", value="")
    else:
        target_category = target_selection["target_category"]
        target_sku = target_selection["target_sku"]
    selected_objective_label = st.selectbox("Цель оптимизации", list(OBJECTIVE_LABEL_TO_MODE.keys()))
    st.caption(OBJECTIVE_HINTS[selected_objective_label])
    objective_mode = OBJECTIVE_LABEL_TO_MODE[selected_objective_label]
    forecast_horizon_days = st.select_slider("Горизонт, дней", options=HORIZON_OPTIONS, value=30)
    caution_level = st.selectbox("Уровень осторожности", ["Низкий", "Средний", "Высокий"], index=1)
    run_requested = st.button("Запустить ценовой анализ", type="primary", use_container_width=True)
    _, objective_mode, objective_warning = resolve_objective_weights(objective_mode)
    return {"load_mode": LOAD_MODE_UNIVERSAL, "universal_txn": universal_txn, "target_category": target_category or None, "target_sku": target_sku or None, "objective_mode": objective_mode, "objective_warning": objective_warning, "forecast_horizon_days": int(forecast_horizon_days), "caution_level": caution_level, "show_risk": True, "run_requested": run_requested}


def render_scenario_lab(r: Dict[str, Any]) -> None:
    presets = build_seller_scenario_presets(float(r["current_price"]), r["_trained_bundle"]["base_ctx"], int(r.get("forecast_horizon_days", 30)))
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
        review_score = st.slider("Рейтинг", 0.0, 5.0, float(r["_trained_bundle"]["base_ctx"].get("review_score", 4.5)), 0.1)
        reviews_count = st.number_input("Количество отзывов", min_value=0.0, value=float(r["_trained_bundle"]["base_ctx"].get("reviews_count", 0.0)), step=1.0)
        cost = st.slider("Множитель себестоимости", 0.7, 1.3, float(base_preset.get("cost_multiplier", 1.0)), 0.05)
        st.markdown("**Дополнительные пользовательские факторы**")
        user_factor_overrides = {}
        for factor in r["_trained_bundle"]["feature_spec"].get("user_factor_features", []):
            label = factor.replace("user_factor__", "", 1)
            user_factor_overrides[factor] = st.number_input(label, value=float(r["_trained_bundle"]["base_ctx"].get(factor, 0.0)), step=0.1)

    w = run_v1_what_if_projection(
        r["_trained_bundle"], manual_price=float(price), freight_multiplier=float(freight), demand_multiplier=float(demand), cost_multiplier=float(cost), horizon_days=int(horizon), stock_cap=float(stock),
        scenario={"name": "scenario_lab", "mode": "manual", "horizon_days": int(horizon), "factors": {"price": float(price), "discount": float(discount), "promotion": float(promo), "stock": float(stock), "review_score": float(review_score), "reviews_count": float(reviews_count), **user_factor_overrides}},
    )
    st.session_state.what_if_result = w


def render_results_page(r: Dict[str, Any]) -> None:
    decision = build_main_decision_text(r)
    st.markdown(f"### {decision.get('action', '—')}")
    if st.button("Запустить: текущий vs рекомендованный vs консервативный", use_container_width=True):
        st.session_state.scenario_table = run_scenario_set(r["_trained_bundle"], build_default_scenario_inputs(float(r["current_price"]), int(r.get("forecast_horizon_days", 30)), r["_trained_bundle"]["base_ctx"])[:3], run_v1_what_if_projection)
    if st.session_state.get("scenario_table") is not None:
        st.dataframe(st.session_state.scenario_table, use_container_width=True)
    explanation = generate_explanation(r, data_quality=r.get("data_quality", {}))
    with st.expander("Расширенная аналитика", expanded=False):
        st.write(explanation.get("summary", ""))
        if st.button("Запустить карту чувствительности", use_container_width=True):
            st.session_state.sensitivity_df = build_sensitivity_grid(r["_trained_bundle"], base_price=float(r["current_price"]), runner=run_v1_what_if_projection)
        if st.session_state.get("sensitivity_df") is not None:
            heat = px.density_heatmap(st.session_state.sensitivity_df, x="price", y="demand_multiplier", z="profit", template="plotly_dark")
            heat.update_layout(**_base_plotly_layout("Чувствительность: цена × спрос"))
            st.plotly_chart(heat, use_container_width=True)


def maybe_run_analysis(ctx: Dict[str, Any]) -> None:
    if not ctx.get("run_requested"):
        return
    if ctx.get("universal_txn") is None:
        st.error("Сначала загрузите и сопоставьте универсальный CSV.")
        return
    if ctx.get("target_category") is None or ctx.get("target_sku") is None:
        st.error("Выберите категорию и SKU.")
        return

    with st.status("Запускаем ценовой анализ…", expanded=True) as status:
        st.write("Подготовка данных и обучение модели")
        try:
            result = run_analysis_from_context(ctx)
        except Exception as exc:
            status.update(label="Ошибка при запуске анализа", state="error", expanded=True)
            st.error("Не удалось завершить анализ. Проверьте входные данные и попробуйте снова.")
            st.exception(exc)
            return

        if not result or "_trained_bundle" not in result:
            status.update(label="Анализ не вернул обученную модель", state="error", expanded=True)
            st.error("Анализ завершился без обученной модели. Проверьте полноту исторических данных.")
            return

        st.session_state.results = result
        reco_price = result.get("recommended_price")
        if reco_price is not None:
            st.write(f"Модель обучена. Рекомендованная цена: {float(reco_price):,.2f}")
        status.update(label="Анализ завершён", state="complete", expanded=False)

    st.session_state.active_page = "Результаты"
    st.session_state.top_nav = "Результаты"
    st.rerun()


apply_enterprise_styles()
active_page = render_navigation()
if active_page == "Обзор":
    render_overview()
elif active_page == "Настройка":
    maybe_run_analysis(render_setup_page())
elif active_page == "Лаборатория сценариев":
    if st.session_state.results is not None:
        render_scenario_lab(st.session_state.results)
elif active_page == "Результаты":
    if st.session_state.results is not None:
        render_results_page(st.session_state.results)
else:
    render_docs()
