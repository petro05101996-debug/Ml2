from __future__ import annotations

from typing import Any, Iterable, Sequence

import streamlit as st


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
    <div class="icon-btn">☰</div>
    <div class="top-header-title">What-if <span class="accent">Cloud</span></div>
  </div>
  <div class="top-header-right">
    <div class="icon-btn">⇩</div>
    <div class="icon-btn">?</div>
    <div class="icon-btn">⚙</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_object_header(object_title: str, status_text: str, scenario_id: str, horizon_text: str, last_update: str, status_color: str = "#7AD0A9") -> None:
    st.markdown(
        f"""
<div class="object-header">
  <div class="mini">← Назад</div>
  <div class="object-row" style="margin-top:10px;">
    <div class="obj-badge">◉</div>
    <div>
      <div class="obj-title">{object_title}</div>
      <div class="obj-meta"><span style="color:{status_color}">●</span> {status_text} · {scenario_id} · {horizon_text} · {last_update}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_action_row() -> str | None:
    items = [
        ("new", "✚", "Новый сценарий"),
        ("reset", "⟲", "Сбросить"),
        ("compare", "⇄", "Сравнить"),
        ("export", "⇩", "Экспорт"),
        ("save", "✓", "Сохранить"),
    ]
    clicked: str | None = None
    cols = st.columns(5)
    for i, (action_id, icon, label) in enumerate(items):
        with cols[i]:
            if st.button(f"{icon}\n{label}", key=f"act_{action_id}", use_container_width=True):
                clicked = action_id
    return clicked


def render_tabs(active_tab: str, tabs: Sequence[str]) -> str:
    idx = tabs.index(active_tab) if active_tab in tabs else 0
    selected = st.radio("Раздел", tabs, horizontal=True, label_visibility="collapsed", index=idx)
    return selected


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
    st.markdown(
        """
<div class="landing-nav">
  <div class="top-header-title">What-if <span class="accent">Cloud</span></div>
  <div class="muted">Возможности · Как работает · Интерфейс · Кейсы · Войти · <b>Начать</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hero_section() -> None:
    st.markdown('<div class="hero-grid" style="margin-top:18px;">', unsafe_allow_html=True)
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">What-if прогнозирование продаж</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-headline">Понимайте, как цена, промо и внешние факторы меняют спрос</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Система быстро считает baseline и показывает эффект сценария на спрос, выручку и прибыль в понятном интерфейсе.</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.button("Попробовать", type="primary", use_container_width=True, key="landing_try")
    b2.button("Смотреть демо", use_container_width=True, key="landing_demo")
    st.markdown('<div class="mini divider-top">Понятный прогноз. Прозрачные сценарии. Без магии и хаоса.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="surface-card">
  <div class="mini">Превью продукта</div>
  <div class="object-header" style="margin-top:8px;">
    <div class="mini">Кофе зерновой 1 кг · SCN-014</div>
    <div class="metric-grid" style="margin-top:10px;">
      <div class="metric-item"><div class="mini">Спрос</div><div><b>+8.4%</b></div></div>
      <div class="metric-item"><div class="mini">Выручка</div><div><b>+₽ 124k</b></div></div>
      <div class="metric-item"><div class="mini">Прибыль</div><div><b>+₽ 48k</b></div></div>
      <div class="metric-item"><div class="mini">Маржа</div><div><b>+1.2 п.п.</b></div></div>
    </div>
    <div class="surface-card" style="padding:14px;margin-top:12px;">Линейный график: база и сценарий</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_trust_metrics() -> None:
    st.markdown('<div class="section-title">Метрики доверия</div>', unsafe_allow_html=True)
    st.markdown('<div class="grid-4">', unsafe_allow_html=True)
    cards = [
        ("< 5 минут", "До первого what-if сценария"),
        ("10+ факторов", "Управляемые и внешние драйверы"),
        ("1 сценарий = 1 результат", "Без перегруженных BI-экранов"),
        ("Для команд", "Аналитики, коммерция, продукт"),
    ]
    for title, sub in cards:
        st.markdown(f'<div class="surface-card"><div class="card-title">{title}</div><div class="muted">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_product_features() -> None:
    st.markdown('<div class="section-title">Что делает продукт</div>', unsafe_allow_html=True)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    cards = [
        ("Прогнозирует спрос", "Использует исторические продажи и факторы для построения базового прогноза."),
        ("Позволяет менять сценарии", "Цена, промо, внешние шоки и другие драйверы меняются вручную."),
        ("Показывает бизнес-эффект", "Сразу видно, как меняются спрос, выручка, прибыль и риск."),
    ]
    for t, d in cards:
        st.markdown(f'<div class="surface-card"><div class="card-title">{t}</div><div class="muted">{d}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_how_it_works() -> None:
    st.markdown('<div class="section-title">Как это работает</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-grid">', unsafe_allow_html=True)
    steps = [
        "1. Загрузите данные: история продаж, признаки и внешние факторы",
        "2. Получите базовый прогноз",
        "3. Измените сценарий: цена, промо, шоки, управляемые факторы",
        "4. Сравните результат: спрос, выручка, прибыль, надежность",
    ]
    for s in steps:
        st.markdown(f'<div class="surface-card">{s}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_interface_preview() -> None:
    st.markdown('<div class="section-title">Интерфейс, в котором видно главное</div><div class="muted">Никакого BI-хлама. Только сценарий, результат и понятный отчет.</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface-card" style="margin-top:12px;">Дашборд · Сценарий · Факторы · Диагностика · Отчет</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface-card">Верхний header, header объекта, action row, ключевые карточки, график и summary-блок отчета.</div>', unsafe_allow_html=True)


def render_reliability_section() -> None:
    st.markdown('<div class="section-title">Почему это надежно</div>', unsafe_allow_html=True)
    st.markdown('<div class="grid-4">', unsafe_allow_html=True)
    cards = ["Прозрачный расчет", "Сценарный подход", "Контроль качества", "Удобный отчет"]
    desc = [
        "Пользователь понимает, что менялось и почему изменился результат.",
        "Сравнение базы и сценария, а не одна цифра вне контекста.",
        "Есть отдельная диагностика и предупреждения по качеству.",
        "Итог виден без чтения технической кухни.",
    ]
    for t, d in zip(cards, desc):
        st.markdown(f'<div class="surface-card"><div class="card-title">{t}</div><div class="muted">{d}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_use_cases() -> None:
    st.markdown('<div class="section-title">Кому подходит</div>', unsafe_allow_html=True)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    cases = [
        ("Коммерческие команды", "Проверка эффекта цены и промо."),
        ("Категорийные менеджеры", "Планирование спроса и оценка изменений факторов."),
        ("Продуктовые и аналитические команды", "Быстрая проверка сценариев без перегруженных инструментов."),
    ]
    for t, d in cases:
        st.markdown(f'<div class="surface-card"><div class="card-title">{t}</div><div class="muted">{d}</div></div>', unsafe_allow_html=True)
    st.markdown('</div><div class="mini" style="margin-top:8px;">Подходит для B2B, e-commerce, retail, FMCG и сценарного планирования.</div>', unsafe_allow_html=True)


def render_final_cta() -> None:
    st.markdown(
        """
<div class="surface-card" style="text-align:center;">
  <div class="section-title">Запустите первый what-if сценарий без лишнего шума</div>
  <div class="muted">От загрузки данных до первого сценария — в одном интерфейсе.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.button("Начать", type="primary", use_container_width=True, key="landing_cta_start")
    c2.button("Открыть демо", use_container_width=True, key="landing_cta_demo")


def render_landing_footer() -> None:
    st.markdown('<div class="mini" style="text-align:center;padding:20px 0;">What-if Cloud</div>', unsafe_allow_html=True)
