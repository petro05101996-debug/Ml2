from __future__ import annotations

from typing import Any, Iterable, Sequence
from html import escape
import streamlit as st


def _safe(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def open_surface(title: str | None = None, subtitle: str | None = None) -> None:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mini">{subtitle}</div>', unsafe_allow_html=True)


def close_surface() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def render_top_header() -> None:
    st.markdown(
        """
<div class="top-header">
  <div class="top-header-left">
    <div class="top-header-title">What-if <span class="accent">Cloud</span></div>
  </div>
  <div class="top-header-right"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="page-header">{_safe(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{_safe(subtitle)}</div>', unsafe_allow_html=True)


def render_stepper(steps: list[dict], active_index: int = 0) -> None:
    items = []
    for i, s in enumerate(steps):
        status = str(s.get("status") or ("active" if i == active_index else "pending"))
        items.append(
            f'<div class="stepper-item {status}"><div class="stepper-index">{i+1}</div><div><div class="stepper-title">{_safe(s.get("title",""))}</div><div class="stepper-caption">{_safe(s.get("caption",""))}</div></div></div>'
        )
    st.markdown(f'<div class="stepper">{"".join(items)}</div>', unsafe_allow_html=True)


def render_help_callout(title: str, text: str, tone: str = "info") -> None:
    st.markdown(f'<div class="help-callout {tone}"><b>{_safe(title)}</b><br/>{_safe(text)}</div>', unsafe_allow_html=True)


def render_kpi_strip(items: list[dict]) -> None:
    blocks = []
    for item in items:
        blocks.append(
            f'<div class="kpi-card"><div class="kpi-label">{_safe(item.get("label",""))}</div><div class="kpi-value">{_safe(item.get("value","—"))}</div><div class="kpi-delta">{_safe(item.get("delta",""))}</div><div class="kpi-base">{_safe(item.get("base",""))}</div></div>'
        )
    st.markdown(f'<div class="kpi-row">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_decision_summary_card(decision_label: str, tone: str, reason: str, metrics: list[dict], economy_label: str | None = None, reliability_label: str | None = None) -> None:
    metric_html = "".join(
        [f'<div class="decision-metric"><div class="decision-metric-label">{_safe(m.get("label",""))}</div><div class="decision-metric-value">{_safe(m.get("value",""))}</div><div class="decision-metric-delta">{_safe(m.get("delta",""))}</div></div>' for m in metrics]
    )
    st.markdown(
        f'<div class="decision-card {tone}"><div class="decision-title">Решение: {_safe(decision_label)}</div><div class="decision-reason">{_safe(reason)}</div><div class="decision-grid">{metric_html}</div><div class="technical-muted">Экономика: {_safe(economy_label or "—")} · Надёжность: {_safe(reliability_label or "—")}</div></div>',
        unsafe_allow_html=True,
    )


def humanize_feature_name(name: str) -> str:
    n = str(name)
    mapping = {"price":"Цена","manual_price":"Цена","discount":"Скидка","promotion":"Промо","promo":"Промо","freight":"Логистика за единицу","freight_value":"Логистика за единицу","cost":"Себестоимость за единицу","quantity":"Продажи","sales":"Продажи","revenue":"Выручка","profit":"Прибыль","margin":"Маржа","sales_lag_7":"Продажи 7 дней назад","sales_lag_14":"Продажи 14 дней назад","sales_roll_7":"Средние продажи за 7 дней","sales_roll_14":"Средние продажи за 14 дней","day_of_week":"День недели","month":"Месяц","holiday":"Праздники","weather":"Погода"}
    if n.startswith("factor__"):
        return f"Внешний фактор: {n.replace('factor__','')}"
    return mapping.get(n, n.replace("_", " ").strip().capitalize())


def render_object_header(
    object_title: str,
    status_text: str,
    horizon_text: str,
    last_update: str,
    status_color: str = "#7AD0A9",
) -> bool:
    back_to_landing = st.button("← Назад", key="back_to_landing", use_container_width=False)
    st.markdown(
        f"""
<div class="object-header">
  <div class="object-row" style="margin-top:10px;">
    <div class="obj-badge">◉</div>
    <div>
      <div class="obj-title">{_safe(object_title)}</div>
      <div class="obj-meta"><span style="color:{_safe(status_color)}">●</span> {_safe(status_text)} · {_safe(horizon_text)} · {_safe(last_update)}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return back_to_landing


def render_workspace_guide(active_tab: str, has_applied_scenario: bool, has_saved_scenarios: bool, has_decision_analysis: bool = False) -> None:
    steps = [
        {"title": "Итог", "caption": "Базовый прогноз", "status": "active" if active_tab == "Итог" else "done"},
        {"title": "Сценарий", "caption": "Проверка гипотезы", "status": "active" if active_tab == "Сценарий" else ("done" if has_applied_scenario else "pending")},
        {"title": "Анализ решений", "caption": "Риски и пилот", "status": "active" if active_tab == "Анализ решений" else ("done" if has_decision_analysis else "pending")},
        {"title": "Сравнение", "caption": "Выбор варианта", "status": "active" if active_tab == "Сравнение" else ("done" if has_saved_scenarios else "pending")},
        {"title": "Отчёт", "caption": "Выгрузка", "status": "active" if active_tab == "Отчёт" else "pending"},
    ]
    render_stepper(steps)


def render_action_row(has_applied_scenario: bool = False, has_saved_scenarios: bool = False) -> str | None:
    items = [
        ("new", "Начать новый сценарий"),
        ("reset_form", "Сбросить параметры"),
        ("cancel_active", "Вернуться к базовому прогнозу"),
        ("compare", "Сравнить варианты"),
        ("export", "Открыть отчёт"),
    ]
    disabled_map = {
        "cancel_active": not has_applied_scenario,
        "compare": not (has_applied_scenario or has_saved_scenarios),
        "export": False,
        "new": False,
        "reset_form": False,
    }
    clicked: str | None = None
    cols = st.columns(5)
    for i, (action_id, label) in enumerate(items):
        with cols[i]:
            if st.button(label, key=f"act_{action_id}", use_container_width=True, disabled=disabled_map.get(action_id, False)):
                clicked = action_id
    return clicked



def render_product_empty_state(title: str, text: str, action_label: str | None = None) -> None:
    open_surface(title)
    st.markdown(f'<div class="muted">{_safe(text)}</div>', unsafe_allow_html=True)
    if action_label:
        st.caption(action_label)
    close_surface()

def render_tabs(active_tab: str, tabs: Sequence[str], key: str = "workspace_tab_radio") -> str:
    if key not in st.session_state or st.session_state.get(key) not in tabs:
        st.session_state[key] = active_tab if active_tab in tabs else tabs[0]
    selected = st.radio(
        "Раздел",
        tabs,
        horizontal=True,
        label_visibility="collapsed",
        key=key,
    )
    return str(selected)


def render_chart_card(title: str, subtitle: str, fig: Any, footer_values: Sequence[tuple[str, str]]) -> None:
    open_surface(title, subtitle)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    cols = st.columns(3)
    for i, (label, value) in enumerate(footer_values[:3]):
        cols[i].metric(label, value)
    close_surface()


def render_metric_summary_card(title: str, big_value: str, caption: str, values: Sequence[tuple[str, str]]) -> None:
    open_surface(title)
    st.markdown(f'<div class="big-metric">{big_value}</div><div class="muted">{caption}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (label, val) in zip([c1, c2, c3], values[:3]):
        with col:
            st.metric(label, val)
    close_surface()


def render_insight_card(lines: Sequence[str]) -> None:
    open_surface("Ключевой вывод")
    for line in lines[:4]:
        st.markdown(f"- {line}")
    close_surface()


def render_compare_card(df) -> None:
    open_surface("Сравнение")
    st.dataframe(df, use_container_width=True, height=220)
    close_surface()


def render_report_card(title: str, body_lines: Sequence[str]) -> None:
    open_surface(title)
    for line in body_lines:
        st.markdown(f"- {line}")
    close_surface()


def render_warning_card(warnings_list: Iterable[str]) -> None:
    open_surface("Предупреждения")
    has = False
    for msg in warnings_list:
        has = True
        st.warning(msg)
    if not has:
        st.success("Критичных предупреждений нет.")
    close_surface()


def render_empty_state(title: str, text: str) -> None:
    open_surface(title)
    st.markdown(text)
    close_surface()


def render_debug_expander(payload: Any) -> None:
    with st.expander("Служебная информация", expanded=False):
        st.json(payload)


# Landing components

def render_landing_nav() -> None:
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.markdown('<div class="top-header-title">What-if <span class="accent">Cloud</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Как работает · Что получите · Какие данные нужны · Ограничения</div>', unsafe_allow_html=True)
    with c2:
        st.button("Начать анализ", key="landing_nav_start", type="primary", use_container_width=True)


def render_landing_hero() -> str | None:
    st.markdown(
        """
<div class="hero-grid" style="margin-top:18px;">
  <div class="surface-card">
    <div class="eyebrow">Сценарный анализ</div>
    <div class="hero-headline">Проверьте цену до запуска акции</div>
    <div class="muted">Загрузите историю продаж и посмотрите, как изменение цены, скидки или промо может повлиять на спрос, выручку, прибыль и маржу.</div>
  </div>
  <div class="surface-card hero-preview-card">
    <div class="card-title">Пример результата</div>
    <div class="mini">Пример результата, не расчёт по вашим данным</div>
    <div class="preview-row"><div class="preview-label">Цена</div><div class="preview-value">1 290 ₽ → 1 390 ₽</div></div>
    <div class="preview-row"><div class="preview-label">Спрос</div><div class="preview-value">-12%</div></div>
    <div class="preview-row"><div class="preview-label">Выручка</div><div class="preview-value">+4%</div></div>
    <div class="preview-row"><div class="preview-label">Прибыль</div><div class="preview-value">+18%</div></div>
    <div class="preview-row"><div class="preview-label">Надёжность</div><div class="preview-value">Средняя</div></div>
    <div class="scenario-help">Вывод: можно проверить как пилот.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Загрузить данные", type="primary", use_container_width=True, key="landing_try"):
        return "app"
    return None


def render_landing_decisions() -> None:
    st.markdown(
        """
<div class="section-title">Какие решения можно проверить</div>
<div class="grid-3">
  <div class="surface-card"><div class="card-title">Цена</div><div class="muted">Что будет, если повысить или снизить цену.</div></div>
  <div class="surface-card"><div class="card-title">Скидка и промо</div><div class="muted">Окупится ли акция, скидка или промо-поддержка.</div></div>
  <div class="surface-card"><div class="card-title">Затраты и внешний спрос</div><div class="muted">Как изменится прибыль при росте логистики или ручной поправке спроса.</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_pipeline() -> None:
    st.markdown(
        """
<div class="section-title">Как это работает</div>
<div class="pipeline-row">
  <div class="pipeline-step"><span>🧾</span><b>Загрузите файл продаж</b></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><span>📦</span><b>Выберите товар</b></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><span>🎛️</span><b>Измените цену, скидку или промо</b></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><span>✅</span><b>Сравните сценарий с текущим планом</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_outputs() -> None:
    st.markdown('<div class="section-title">Что вы получите после расчёта</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="surface-card"><div class="card-title">Текущий план</div><div class="muted">Оценка спроса, выручки, прибыли и маржи на выбранный период.</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="surface-card"><div class="card-title">Сценарий</div><div class="muted">Сравнение текущего плана с вашим сценарием.</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="surface-card"><div class="card-title">Отчёт</div><div class="muted">Excel/CSV/JSON с результатами для проверки и обсуждения.</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="surface-card mini-report-preview">
  <div class="card-title">Краткий вывод</div>
  <div class="muted">Сценарий повышает прибыль, но снижает спрос. Надёжность средняя: цену стоит проверять пилотом.</div>
</div>
""",
        unsafe_allow_html=True,
    )

def render_landing_data_requirements() -> None:
    st.markdown('<div class="section-title">Какие данные нужны</div>', unsafe_allow_html=True)
    st.markdown("""<div class="grid-3"><div class="surface-card"><div class="card-title">Минимально</div><ul class="landing-list"><li>Дата</li><li>Товар / SKU</li><li>Продажи в штуках</li><li>Цена</li></ul></div>
<div class="surface-card"><div class="card-title">Желательно</div><ul class="landing-list"><li>Себестоимость</li><li>Скидка</li><li>Промо</li><li>Логистика</li><li>Остатки</li></ul></div>
<div class="surface-card"><div class="card-title">Дополнительно</div><ul class="landing-list"><li>Трафик</li><li>Спрос</li><li>Сезонность</li><li>Конкуренты</li><li>Погода и праздники</li></ul></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="muted">Чем больше в истории реальных изменений цены, скидок и промо, тем надёжнее сценарный анализ.</div>', unsafe_allow_html=True)

def render_landing_limits() -> None:
    st.markdown('<div class="surface-card"><div class="section-title">Что важно понимать</div><div class="muted">Инструмент не обещает идеальную цену автоматически. Он сравнивает сценарии на основе вашей истории продаж. Если данных мало или новая цена сильно отличается от прошлых значений, система покажет предупреждение и снизит уровень надёжности.</div></div>', unsafe_allow_html=True)

def render_landing_cta() -> str | None:
    st.markdown(
        """
<div class="surface-card" style="text-align:center;">
<div class="section-title">Проверьте сценарий на своих данных</div>
  <div class="muted">Загрузите файл продаж и получите понятное сравнение текущего плана и вашего сценария.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Начать анализ", type="primary", use_container_width=True, key="landing_cta_start"):
        return "app"
    return None


def render_landing_footer() -> None:
    st.markdown('<div class="mini" style="text-align:center;padding:20px 0;">What-if Cloud · Сценарный анализ цены, спроса и прибыли</div>', unsafe_allow_html=True)
