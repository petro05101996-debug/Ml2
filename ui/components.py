from __future__ import annotations

from typing import Any, Iterable, Sequence
from html import escape
import streamlit as st

from ui.copy import PAGE_WHAT_IF


def _safe(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def open_surface(title: str | None = None, subtitle: str | None = None) -> None:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{_safe(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="mini">{_safe(subtitle)}</div>', unsafe_allow_html=True)


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
            f'<div class="stepper-item {_safe(status)}"><div class="stepper-index">{i+1}</div><div><div class="stepper-title">{_safe(s.get("title",""))}</div><div class="stepper-caption">{_safe(s.get("caption",""))}</div></div></div>'
        )
    st.markdown(f'<div class="stepper">{"".join(items)}</div>', unsafe_allow_html=True)


def render_help_callout(title: str, text: str, tone: str = "info") -> None:
    st.markdown(f'<div class="help-callout {_safe(tone)}"><b>{_safe(title)}</b><br/>{_safe(text)}</div>', unsafe_allow_html=True)


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
    next_step = "Запустить пилот" if str(tone) == "success" else ("Не запускать" if str(tone) == "danger" else "Проверить другой сценарий")
    st.markdown(
        f'<div class="decision-hero {tone}"><div class="decision-hero-title">Вердикт: {_safe(decision_label)}</div><div class="decision-hero-text"><b>Почему:</b> {_safe(reason)}</div><div class="decision-grid">{metric_html}</div><div class="decision-section-grid"><div class="decision-section-card"><div class="decision-section-label">Надёжность</div><div class="decision-section-value">{_safe(reliability_label or "—")}</div></div><div class="decision-section-card"><div class="decision-section-label">Следующий шаг</div><div class="decision-section-value">{_safe(next_step)}</div></div></div><div class="technical-muted">Экономика: {_safe(economy_label or "—")}</div></div>',
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
    status_color: str = "success",
) -> bool:
    back_to_landing = st.button("← Назад", key="back_to_landing", use_container_width=False)
    st.markdown(
        f"""
<div class="object-header">
  <div class="object-row" style="margin-top:10px;">
    <div class="obj-badge">◉</div>
    <div>
      <div class="obj-title">{_safe(object_title)}</div>
      <div class="obj-meta"><span class="status-dot status-{_safe(status_color)}">●</span> {_safe(status_text)} · {_safe(horizon_text)} · {_safe(last_update)}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return back_to_landing


def render_workspace_guide(active_tab: str, has_applied_scenario: bool, has_saved_scenarios: bool, has_decision_analysis: bool = False) -> None:
    steps = [
        {"title": "Что у меня сейчас", "caption": "Текущий план", "status": "done"},
        {"title": "Что хочу проверить", "caption": "Бизнес-гипотеза", "status": "active" if active_tab == PAGE_WHAT_IF else ("done" if has_applied_scenario else "pending")},
        {"title": "Что получилось", "caption": "Итог", "status": "done" if has_applied_scenario else "pending"},
        {"title": "Можно ли доверять", "caption": "Риски", "status": "done" if has_decision_analysis else "pending"},
        {"title": "Что дальше", "caption": "Сравнение и отчёт", "status": "done" if has_saved_scenarios else "pending"},
    ]
    render_stepper(steps)


def render_action_row(has_applied_scenario: bool = False, has_saved_scenarios: bool = False) -> str | None:
    clicked: str | None = None
    primary_label = "Проверить решение"
    cta_col, menu_col = st.columns([0.68, 0.32])
    with cta_col:
        if st.button(primary_label, key="act_primary_decision", type="primary", use_container_width=True):
            clicked = "decision"
    with menu_col:
        popover = getattr(st, "popover", None)
        if callable(popover):
            with st.popover("Другие действия", use_container_width=True):
                if st.button("Сбросить параметры", key="act_reset_form", use_container_width=True):
                    clicked = "reset_form"
                if st.button("Быстрый what-if", key="act_quick_what_if", use_container_width=True):
                    clicked = "scenario"
                if st.button("Новый сценарий", key="act_new", use_container_width=True):
                    clicked = "new"
                if st.button("Вернуться к текущему плану", key="act_cancel_active", use_container_width=True, disabled=not has_applied_scenario):
                    clicked = "cancel_active"
                if not has_applied_scenario:
                    st.caption("Недоступно: сценарий ещё не рассчитан.")
                if st.button("Сравнить", key="act_compare", use_container_width=True, disabled=not (has_applied_scenario or has_saved_scenarios)):
                    clicked = "compare"
                if not (has_applied_scenario or has_saved_scenarios):
                    st.caption("Недоступно: нет рассчитанных или сохранённых вариантов.")
                if st.button("Отчёт", key="act_export", use_container_width=True):
                    clicked = "export"
        else:
            with st.expander("Другие действия", expanded=False):
                if st.button("Сбросить параметры", key="act_reset_form_fallback", use_container_width=True):
                    clicked = "reset_form"
                if st.button("Быстрый what-if", key="act_quick_what_if_fallback", use_container_width=True):
                    clicked = "scenario"
                if st.button("Новый сценарий", key="act_new_fallback", use_container_width=True):
                    clicked = "new"
                if st.button("Вернуться к текущему плану", key="act_cancel_active_fallback", use_container_width=True, disabled=not has_applied_scenario):
                    clicked = "cancel_active"
                if st.button("Сравнить", key="act_compare_fallback", use_container_width=True, disabled=not (has_applied_scenario or has_saved_scenarios)):
                    clicked = "compare"
                if st.button("Отчёт", key="act_export_fallback", use_container_width=True):
                    clicked = "export"
    return clicked




def render_metric_card(label: str, value: str, delta: str | None = None, tone: str = "neutral") -> None:
    st.markdown(
        f'<div class="metric-card {_safe(tone)}"><div class="metric-card-label">{_safe(label)}</div><div class="metric-card-value">{_safe(value)}</div><div class="metric-card-delta">{_safe(delta or "")}</div></div>',
        unsafe_allow_html=True,
    )


def render_next_action_card(title: str, text: str, tone: str = "info") -> None:
    st.markdown(
        f'<div class="next-action-card {_safe(tone)}"><div class="card-title">{_safe(title)}</div><div class="muted">{_safe(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_scenario_preview_card(current: Sequence[tuple[str, str]], future: Sequence[tuple[str, str]], status: str) -> None:
    def rows(items: Sequence[tuple[str, str]]) -> str:
        return "".join(f'<div class="preview-row"><div class="preview-label">{_safe(k)}</div><div class="preview-value">{_safe(v)}</div></div>' for k, v in items)
    st.markdown(
        f'<div class="scenario-preview-card"><div class="card-title">Текущий план</div>{rows(current)}<div class="preview-spacer"></div><div class="card-title">Будущий сценарий</div>{rows(future)}<div class="preview-spacer"></div><div class="preview-row"><div class="preview-label">Статус</div><div class="preview-value">{_safe(status)}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_risk_card(title: str, items: Sequence[str], tone: str = "warning") -> None:
    if not items:
        items = ["Критичных ограничений не найдено."]
    body = "".join(f"<li>{_safe(item)}</li>" for item in items)
    st.markdown(f'<div class="risk-card {_safe(tone)}"><div class="card-title">{_safe(title)}</div><ul>{body}</ul></div>', unsafe_allow_html=True)


def render_product_empty_state(title: str, text: str, action_label: str | None = None) -> None:
    open_surface(title)
    st.markdown(f'<div class="muted">{_safe(text)}</div>', unsafe_allow_html=True)
    if action_label:
        st.caption(action_label)
    close_surface()

def render_tabs(active_tab: str, tabs: Sequence[str], key: str = "workspace_tab_radio") -> str:
    """Render product-section navigation as explicit buttons, not filter-like radio pills."""
    if key not in st.session_state or st.session_state.get(key) not in tabs:
        st.session_state[key] = active_tab if active_tab in tabs else tabs[0]
    st.markdown('<div class="app-nav" role="navigation" aria-label="Разделы продукта">', unsafe_allow_html=True)
    cols = st.columns(len(tabs))
    for idx, tab in enumerate(tabs):
        is_active = st.session_state.get(key) == tab
        with cols[idx]:
            btn_type = "primary" if is_active else "secondary"
            if st.button(tab, key=f"{key}_nav_{idx}", type=btn_type, use_container_width=True):
                st.session_state[key] = tab
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    return str(st.session_state[key])


def render_chart_card(title: str, subtitle: str, fig: Any, footer_values: Sequence[tuple[str, str]]) -> None:
    open_surface(title, subtitle)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    cols = st.columns(3)
    for i, (label, value) in enumerate(footer_values[:3]):
        cols[i].metric(label, value)
    close_surface()


def render_metric_summary_card(title: str, big_value: str, caption: str, values: Sequence[tuple[str, str]]) -> None:
    open_surface(title)
    st.markdown(f'<div class="big-metric">{_safe(big_value)}</div><div class="muted">{_safe(caption)}</div>', unsafe_allow_html=True)
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

def render_landing_nav() -> str | None:
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.markdown('<div class="top-header-title">What-if <span class="accent">Cloud</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Как работает · Что получите · Какие данные нужны · Ограничения</div>', unsafe_allow_html=True)
    with c2:
        if st.button("Начать анализ", key="landing_nav_start", type="primary", use_container_width=True):
            return "app"
    return None


def render_landing_hero() -> str | None:
    st.markdown(
        """
<div class="overview-hero" style="margin-top:18px;">
  <div class="hero-title">Проверьте бизнес-решение до запуска</div>
  <div class="hero-text">Загрузите историю продаж, измените цену, скидку, промо или внешний спрос — система покажет влияние на спрос, выручку, прибыль и риск.</div>
</div>
<div class="action-card-grid">
  <div class="action-card"><div class="card-title">Пример вердикта</div><div class="muted">Запускать через пилот</div><div class="preview-row"><div class="preview-label">Решение</div><div class="preview-value">Поднять цену до 1 640 ₽</div></div><div class="preview-row"><div class="preview-label">Эффект</div><div class="preview-value">Прибыль +8.2% · Спрос −3.1% · Выручка +4.4%</div></div><div class="preview-row"><div class="preview-label">Риск</div><div class="preview-value">Средний — нужна проверка 14 дней</div></div></div>
  <div class="action-card"><div class="card-title">Главная фишка</div><div class="muted">Проверьте изменение цены, скидки, промо или внешнего спроса до запуска и получите план безопасного пилота.</div></div>
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
<div class="grid-3">
  <div class="surface-card"><div class="card-title">1. Прогноз</div><div class="muted">Система строит базовый план по истории продаж.</div></div>
  <div class="surface-card"><div class="card-title">2. Решение</div><div class="muted">Вы задаёте изменение: цена, скидка, промо или внешний спрос.</div></div>
  <div class="surface-card"><div class="card-title">3. Вердикт</div><div class="muted">Система показывает эффект, риск и план безопасного пилота.</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

def render_landing_outputs() -> None:
    st.markdown('<div class="section-title">Что вы получите после расчёта</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="surface-card"><div class="card-title">Вердикт</div><div class="muted">Запускать, пилотировать или пересмотреть условия.</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="surface-card"><div class="card-title">Эффект</div><div class="muted">Спрос, выручка, прибыль и надёжность — максимум 4 главные метрики.</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="surface-card"><div class="card-title">План пилота</div><div class="muted">Как безопасно проверить решение перед масштабированием.</div></div>', unsafe_allow_html=True)

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
<div class="section-title">Проверьте бизнес-решение на своих данных</div>
  <div class="muted">Загрузите файл продаж и получите понятный вердикт, KPI, риски и план пилота.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Начать анализ", type="primary", use_container_width=True, key="landing_cta_start"):
        return "app"
    return None


def render_landing_footer() -> None:
    st.markdown('<div class="mini" style="text-align:center;padding:20px 0;">What-if Cloud · Проверка бизнес-решений до запуска</div>', unsafe_allow_html=True)


def render_status_badge(label: str, tone: str = "neutral") -> None:
    st.markdown(f'<span class="status-badge {_safe(tone)}">{_safe(label)}</span>', unsafe_allow_html=True)


def render_overview_hero(title: str, text: str) -> None:
    st.markdown(
        f'<div class="overview-hero"><div class="hero-title">{_safe(title)}</div><div class="hero-text">{_safe(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_decision_mode_cards(active_mode: str) -> str:
    st.markdown('<div class="action-card-grid">', unsafe_allow_html=True)
    cols = st.columns(3)
    selected = active_mode
    with cols[0]:
        st.markdown(f'<div class="action-card {"active" if active_mode == "find_best" else ""}"><div class="card-title">Найти лучшее решение</div><div class="muted">Система сама переберёт допустимые варианты цены, скидки, промо и логистики.</div></div>', unsafe_allow_html=True)
        if st.button("Найти лучшее решение", key="decision_mode_find_best", use_container_width=True):
            selected = "find_best"
    with cols[1]:
        st.markdown(f'<div class="action-card {"active" if active_mode == "audit_idea" else ""}"><div class="card-title">Проверить мою идею</div><div class="muted">Вы уже знаете, что хотите сделать — система проверит эффект, риск и план пилота.</div></div>', unsafe_allow_html=True)
        if st.button("Проверить мою идею", key="decision_mode_audit_idea", use_container_width=True):
            selected = "audit_idea"
    with cols[2]:
        st.markdown(f'<div class="action-card {"active" if active_mode == "quick_what_if" else ""}"><div class="card-title">Быстрый what-if</div><div class="muted">Сразу перейти к ручной настройке цены, скидки, промо, логистики и внешнего спроса.</div></div>', unsafe_allow_html=True)
        if st.button("Быстрый what-if", key="decision_mode_quick_what_if", use_container_width=True):
            selected = "quick_what_if"
    st.markdown('</div>', unsafe_allow_html=True)
    return selected


def render_verdict_panel(
    verdict_label: str,
    action_title: str,
    reason: str,
    metrics: list[dict],
    reliability_label: str,
    next_step: str,
    tone: str = "warning",
) -> None:
    metric_html = "".join(
        f'<div class="decision-metric"><div class="decision-metric-label">{_safe(m.get("label", ""))}</div><div class="decision-metric-value">{_safe(m.get("value", "—"))}</div><div class="decision-metric-delta">{_safe(m.get("delta", ""))}</div></div>'
        for m in list(metrics or [])[:4]
    )
    st.markdown(
        f'<div class="verdict-panel {_safe(tone)}"><div class="verdict-title">Вердикт: {_safe(verdict_label)}</div><div class="verdict-reason"><b>Решение:</b> {_safe(action_title)}<br/><br/><b>Почему:</b> {_safe(reason)}</div><div class="decision-grid">{metric_html}<div class="decision-metric"><div class="decision-metric-label">Надёжность</div><div class="decision-metric-value">{_safe(reliability_label)}</div></div></div><div class="decision-section-card"><div class="decision-section-label">Следующий шаг</div><div class="decision-section-value">{_safe(next_step)}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_wizard_steps(active_step: int, steps: list[str]) -> None:
    render_stepper(
        [{"title": step, "caption": f"Шаг {i + 1} из {len(steps)}", "status": "active" if i == active_step else ("done" if i < active_step else "pending")} for i, step in enumerate(steps)],
        active_step,
    )


def render_compact_context_card(context: dict) -> None:
    rows = "".join(f'<div class="preview-row"><div class="preview-label">{_safe(k)}</div><div class="preview-value">{_safe(v)}</div></div>' for k, v in (context or {}).items())
    st.markdown(f'<div class="scenario-preview-card">{rows}</div>', unsafe_allow_html=True)
