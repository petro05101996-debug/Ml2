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
from pricing_core import CONFIG, OBJECTIVE_HINTS, OBJECTIVE_LABEL_TO_MODE, generate_explanation, run_v2_what_if_projection
from ui_docs import render_docs
from ui_overview import render_overview
from ui_shell import apply_enterprise_styles, render_navigation
from what_if import build_sensitivity_grid, run_scenario_set

st.set_page_config(page_title="Demand What-If Studio", layout="wide", page_icon="📊")
for key, default in {"results": None, "what_if_result": None, "scenario_table": None, "sensitivity_df": None, "active_page": "Обзор"}.items():
    if key not in st.session_state:
        st.session_state[key] = default

HORIZON_OPTIONS = [7, 14, 30, 60, 90, 180, 360]
LOAD_MODE_UNIVERSAL = "Универсальный файл транзакций"


def resolve_objective_weights(mode: str):
    try:
        return objective_to_weights(mode), mode, None
    except ValueError:
        fallback = "balanced_mode"
        return objective_to_weights(fallback), fallback, "Неизвестный режим оптимизации. Применён безопасный режим: «Сбалансированный режим»."


def _base_plotly_layout(title: str) -> Dict[str, Any]:
    return dict(title=title, template="plotly_dark")


def _format_report(r: Dict[str, Any], explanation: Dict[str, Any]) -> str:
    if r.get("analysis_engine") == "v2_decomposed_baseline_factor_shock":
        c = r.get("v2_result_contract", {})
        lines = [
            "# Отчёт v2 сценарного анализа",
            "",
            "## Краткий итог",
            f"- Действие: {c.get('headline_action', '—')}",
            f"- Обоснование: {c.get('headline_reason', '—')}",
            f"- Режим: {c.get('mode', 'baseline_only')}",
            f"- Overall confidence: {c.get('overall_confidence', 'low')}",
            f"- Baseline granularity: {c.get('baseline_granularity', 'daily')}",
            f"- Baseline strategy: {c.get('baseline_strategy', '—')}",
            f"- Baseline selector reason: {c.get('baseline_selector_reason', '—')}",
            f"- Daily best strategy: {c.get('best_daily_strategy', '—')}",
            f"- Weekly best strategy: {c.get('best_weekly_strategy', '—')}",
            f"- Baseline demand: {float(c.get('baseline_total_demand', 0.0)):,.2f}",
            f"- Scenario demand: {float(c.get('scenario_total_demand', 0.0)):,.2f}",
            f"- Demand delta: {float(c.get('demand_delta_pct', 0.0)):+.2%}",
            f"- Revenue delta: {float(c.get('revenue_delta_pct', 0.0)):+.2%}",
            f"- Profit delta: {float(c.get('profit_delta_pct', 0.0)):+.2%}",
            "",
            "## OOD flags",
            ", ".join([str(x) for x in c.get("ood_flags", [])]) if c.get("ood_flags") else "нет",
            "",
            "## Пояснение модели",
            str(explanation.get("summary", "Нет дополнительного пояснения.")),
        ]
        return "\n".join(lines)

    decision = build_main_decision_text(r)
    objective = r.get("objective_mode", "—")
    current_price = float(r.get("current_price", 0.0))
    recommended_price = r.get("recommended_price")
    confidence = r.get("confidence")

    lines = [
        "# Отчёт ценового анализа",
        "",
        "## Краткий итог",
        f"- Решение: {decision.get('action', '—')}",
        f"- Цель оптимизации: {objective}",
        f"- Текущая цена: {current_price:,.2f}",
        f"- Рекомендованная цена: {float(recommended_price):,.2f}" if recommended_price is not None else "- Рекомендованная цена: —",
        f"- Уверенность (эвристика): {float(confidence):.2f}" if confidence is not None else "- Уверенность (эвристика): —",
        "",
        "## Пояснение модели",
        str(explanation.get("summary", "Нет дополнительного пояснения.")),
    ]

    scenario_table = st.session_state.get("scenario_table")
    if scenario_table is not None and not scenario_table.empty:
        lines.extend(["", "## Сценарии", scenario_table.to_csv(index=False)])

    sensitivity_df = st.session_state.get("sensitivity_df")
    if sensitivity_df is not None and not sensitivity_df.empty:
        lines.extend(["", "## Карта чувствительности", sensitivity_df.to_csv(index=False)])

    return "\n".join(lines)


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
    universal_file = st.file_uploader("Файл транзакций (.csv/.xlsx) *", type=["csv", "xlsx"], key="universal_file")
    universal_txn = None
    if universal_file is not None:
        if str(universal_file.name).lower().endswith(".xlsx"):
            preview = pd.read_excel(universal_file)
        else:
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
    st.file_uploader(
        "Upload document events (pdf/docx) — beta placeholder",
        type=["pdf", "docx"],
        key="document_events_file",
        help="Документы пока не влияют напрямую на модель. Следующий шаг — извлечение фактов в таблицу событий.",
    )

    target_selection = _build_target_selection(universal_txn)
    if target_selection["target_category"] is None and target_selection["target_sku"] is None:
        target_category = st.text_input("Категория *", value="")
        target_sku = st.text_input("SKU *", value="")
    else:
        target_category = target_selection["target_category"]
        target_sku = target_selection["target_sku"]
    selected_objective_label = st.selectbox("Цель оптимизации", list(OBJECTIVE_LABEL_TO_MODE.keys()))
    st.caption(OBJECTIVE_HINTS[selected_objective_label])
    st.caption("Цель влияет на ранжирование и выбор рекомендуемого сценария, а не на сам механизм прогноза спроса.")
    objective_mode = OBJECTIVE_LABEL_TO_MODE[selected_objective_label]
    forecast_horizon_days = st.select_slider("Горизонт, дней", options=HORIZON_OPTIONS, value=30)
    run_requested = st.button("Запустить Demand What-If анализ", type="primary", use_container_width=True)
    _, objective_mode, objective_warning = resolve_objective_weights(objective_mode)
    return {"load_mode": LOAD_MODE_UNIVERSAL, "universal_txn": universal_txn, "target_category": target_category or None, "target_sku": target_sku or None, "objective_mode": objective_mode, "objective_warning": objective_warning, "forecast_horizon_days": int(forecast_horizon_days), "show_risk": True, "run_requested": run_requested}


def render_scenario_lab(r: Dict[str, Any]) -> None:
    bundle = r.get("_trained_bundle", {})
    base_ctx = bundle.get("base_ctx", {})
    scenario_spec = bundle.get("scenario_feature_spec", bundle.get("feature_spec", {}))
    current_price = float(r.get("current_price", bundle.get("current_price", base_ctx.get("price", 1.0))))
    presets = build_seller_scenario_presets(current_price, base_ctx, int(r.get("forecast_horizon_days", bundle.get("forecast_horizon_days", 30))))
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
    price = p1.number_input("Цена", min_value=0.01, value=float(base_preset.get("price", current_price)), step=1.0)
    demand = p2.slider("Множитель спроса", 0.7, 1.3, float(base_preset.get("demand_multiplier", 1.0)), 0.05)
    freight = p3.slider("Множитель логистики", 0.5, 1.5, float(base_preset.get("freight_multiplier", 1.0)), 0.05)
    horizon = p4.select_slider("Горизонт", options=HORIZON_OPTIONS, value=int(base_preset.get("horizon_days", 30)))
    with st.expander("Расширенный режим", expanded=False):
        discount = st.slider("Скидка", 0.0, 0.95, float(base_preset.get("discount", 0.0)), 0.01)
        promo = st.slider("Промо", 0.0, 1.0, float(base_preset.get("promotion", 0.0)), 0.05)
        use_stock_cap = st.checkbox("Использовать ограничение запаса", value=False)
        stock_default = float(base_preset.get("stock_cap", 0.0))
        stock = st.number_input("Лимит запаса", min_value=0.0, value=stock_default, step=1.0, disabled=not use_stock_cap)
        cost = st.slider("Множитель себестоимости", 0.7, 1.3, float(base_preset.get("cost_multiplier", 1.0)), 0.05)
        st.markdown("**Дополнительные пользовательские факторы**")
        user_factor_overrides = {}
        for factor in scenario_spec.get("user_numeric_features", []):
            label = factor.replace("user_factor_num__", "", 1)
            user_factor_overrides[factor] = st.number_input(label, value=float(base_ctx.get(factor, 0.0)), step=0.1)

    w = run_v2_what_if_projection(
        r["_trained_bundle"], manual_price=float(price), freight_multiplier=float(freight), demand_multiplier=float(demand), cost_multiplier=float(cost), horizon_days=int(horizon), stock_cap=(float(stock) if use_stock_cap else None),
        scenario={"name": "scenario_lab", "mode": "manual", "horizon_days": int(horizon), "factors": {"price": float(price), "discount": float(discount), "promotion": float(promo), **user_factor_overrides}},
    )
    st.session_state.what_if_result = w


def render_results_page(r: Dict[str, Any]) -> None:
    if r.get("analysis_engine") == "v2_decomposed_baseline_factor_shock":
        contract = r.get("v2_result_contract", {})
        st.markdown(f"### {contract.get('headline_action', 'Сценарный анализ v2')}")
        st.caption(contract.get("headline_reason", ""))
        st.caption(
            f"baseline={contract.get('baseline_granularity', 'daily')}/{contract.get('baseline_strategy', 'xgb_recursive')}; "
            f"daily_best={contract.get('best_daily_strategy', '—')}; "
            f"weekly_best={contract.get('best_weekly_strategy', '—')}"
        )
        if contract.get("baseline_selector_reason"):
            st.caption(f"selector: {contract.get('baseline_selector_reason')}")
        k1, k2, k3 = st.columns(3)
        k1.metric("Спрос Δ", f"{float(contract.get('demand_delta_pct', 0.0)):+.2%}")
        k2.metric("Выручка Δ", f"{float(contract.get('revenue_delta_pct', 0.0)):+.2%}")
        k3.metric("Прибыль Δ", f"{float(contract.get('profit_delta_pct', 0.0)):+.2%}")
    else:
        decision = build_main_decision_text(r)
        st.markdown(f"### {decision.get('action', '—')}")
    explanation = {} if r.get("analysis_engine") == "v2_decomposed_baseline_factor_shock" else generate_explanation(r, data_quality=r.get("data_quality", {}))
    col_excel, col_md = st.columns(2)
    excel_buffer = r.get("excel_buffer")
    if excel_buffer is not None:
        try:
            excel_payload = excel_buffer.getvalue()
        except Exception:
            excel_payload = None
        if excel_payload:
            col_excel.download_button(
                "Скачать отчёт (.xlsx)",
                data=excel_payload,
                file_name="pricing_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    report_text = _format_report(r, explanation)
    col_md.download_button(
        "Скачать отчёт (.md)",
        data=report_text.encode("utf-8"),
        file_name="pricing_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dq = r.get("data_quality", {}) or {}
    if r.get("analysis_engine") == "v2_decomposed_baseline_factor_shock":
        conf = r.get("confidence", {})
        st.caption(
            f"v2 confidence: overall={conf.get('overall_confidence', 'low')}; "
            f"intervals_available={bool(r.get('intervals_available', False))}."
        )
        if conf.get("issues"):
            st.warning("Ограничения confidence: " + "; ".join([str(x) for x in conf.get("issues", [])]))
    else:
        st.caption(
            f"Качество анализа: {dq.get('label', '—')}. "
            f"Уровень: {dq.get('level', '—')}. "
            "Метрика качества прогноза рассчитывается через WAPE на holdout-периоде."
        )
        if dq.get("issues"):
            st.warning("Ограничения качества данных: " + "; ".join([str(x) for x in dq.get("issues", [])]))
    holdout_metrics = r.get("holdout_metrics")
    if r.get("analysis_engine") != "v2_decomposed_baseline_factor_shock" and isinstance(holdout_metrics, pd.DataFrame) and not holdout_metrics.empty:
        st.dataframe(holdout_metrics, use_container_width=True)
        diag_row = holdout_metrics.iloc[0].to_dict()
        st.markdown("#### Диагностика модели")
        st.write(
            {
                "forecast_wape": diag_row.get("forecast_wape"),
                "mae": diag_row.get("mae"),
                "rmse": diag_row.get("rmse"),
                "sum_ratio": diag_row.get("sum_ratio"),
                "bias_pct": diag_row.get("bias_pct"),
                "forecast_mode": diag_row.get("forecast_mode"),
                "price_signal_ok": diag_row.get("price_signal_ok"),
                "weak_factors": diag_row.get("weak_factors"),
                "ood_flags": diag_row.get("ood_flags"),
                "can_recommend_price": diag_row.get("can_recommend_price"),
            }
        )
        if not bool(diag_row.get("can_recommend_price", False)):
            st.warning("Ценовой сигнал недостаточен для надёжной рекомендации.")
    if r.get("analysis_engine") != "v2_decomposed_baseline_factor_shock" and st.button("Запустить: текущий vs рекомендованный vs консервативный", use_container_width=True):
        st.session_state.scenario_table = run_scenario_set(r["_trained_bundle"], build_default_scenario_inputs(float(r["current_price"]), int(r.get("forecast_horizon_days", 30)), r["_trained_bundle"]["base_ctx"])[:3], run_v2_what_if_projection)
    if r.get("analysis_engine") != "v2_decomposed_baseline_factor_shock" and st.session_state.get("scenario_table") is not None:
        st.dataframe(st.session_state.scenario_table, use_container_width=True)
    with st.expander("Расширенная аналитика", expanded=False):
        st.write(explanation.get("summary", ""))
        if r.get("analysis_engine") != "v2_decomposed_baseline_factor_shock" and st.button("Запустить карту чувствительности", use_container_width=True):
            st.session_state.sensitivity_df = build_sensitivity_grid(r["_trained_bundle"], base_price=float(r["current_price"]), runner=run_v2_what_if_projection)
        if r.get("analysis_engine") != "v2_decomposed_baseline_factor_shock" and st.session_state.get("sensitivity_df") is not None:
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
        if result.get("analysis_engine") == "v2_decomposed_baseline_factor_shock":
            action_line = (result.get("v2_result_contract") or {}).get("headline_action")
            if action_line:
                st.write(action_line)
        else:
            reco_price = result.get("recommended_price")
            if reco_price is not None:
                st.write(f"Модель обучена. Рекомендованная цена: {float(reco_price):,.2f}")
        status.update(label="Анализ завершён", state="complete", expanded=False)

    st.session_state.active_page = "Результаты"
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
