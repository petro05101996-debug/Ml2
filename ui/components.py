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
      <div class="obj-title">{object_title}</div>
      <div class="obj-meta"><span style="color:{status_color}">●</span> {status_text} · {horizon_text} · {last_update}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return back_to_landing


def render_action_row() -> str | None:
    items = [
        ("new", "✚", "Новый сценарий"),
        ("reset_form", "⟲", "Сбросить форму"),
        ("cancel_active", "⊘", "Отменить активный"),
        ("compare", "⇄", "К сравнению"),
        ("export", "⇩", "К отчёту и экспорту"),
    ]
    clicked: str | None = None
    cols = st.columns(5)
    for i, (action_id, icon, label) in enumerate(items):
        with cols[i]:
            if st.button(f"{icon}\n{label}", key=f"act_{action_id}", use_container_width=True):
                clicked = action_id
    return clicked


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
    st.markdown(
        """
<div class="landing-nav">
  <div class="top-header-title">What-if <span class="accent">Cloud</span></div>
  <div class="muted">Возможности · Как работает · Интерфейс · Кейсы · Войти · <b>Начать</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_hero_v2() -> str | None:
    st.markdown(
        """
<div class="hero-grid" style="margin-top:18px;">
  <div class="surface-card">
    <div class="eyebrow">What-if прогнозирование спроса</div>
    <div class="hero-headline">Проверяйте, как цена, промо и внешние факторы меняют продажи</div>
    <div class="muted">Меняйте сценарий и сразу видите влияние на спрос, выручку и прибыль — в одном понятном отчёте.</div>
    <div class="chip-row">
      <span class="floating-chip">Цена</span>
      <span class="floating-chip">Промо</span>
      <span class="floating-chip">Логистика</span>
      <span class="floating-chip">Внешние факторы</span>
    </div>
  </div>
  <div class="surface-card hero-visual">
    <div class="hero-chart-title">База vs Сценарий</div>
    <svg viewBox="0 0 560 250" class="hero-chart-svg">
      <polyline points="20,188 120,178 220,190 320,170 430,160 540,148" fill="none" stroke="#9DCC84" stroke-width="4"/>
      <polyline points="20,188 120,172 220,168 320,146 430,128 540,112" fill="none" stroke="#6F70FF" stroke-width="5"/>
    </svg>
    <div class="floating-chip chip-a">Цена -5%</div>
    <div class="floating-chip chip-b">Промо +20%</div>
    <div class="floating-chip chip-c">Логистика 1.1x</div>
    <div class="metric-float metric-a"><span>Спрос</span><b>+8.4%</b></div>
    <div class="metric-float metric-b"><span>Выручка</span><b>+124k</b></div>
    <div class="metric-float metric-c"><span>Прибыль</span><b>+48k</b></div>
    <div class="report-badge">Рекомендуется к пилоту</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns(2)
    if b1.button("Перейти в приложение", type="primary", use_container_width=True, key="landing_try_v2"):
        return "app"
    if b2.button("Посмотреть интерфейс", use_container_width=True, key="landing_demo_v2"):
        return "app"
    return None


def render_landing_proof_strip() -> None:
    st.markdown(
        """
<div class="proof-strip">
  <div class="proof-item">⚡ Сценарий за минуты</div>
  <div class="proof-item">🏷️ Цена / промо / факторы</div>
  <div class="proof-item">📈 Спрос / выручка / прибыль</div>
  <div class="proof-item">📄 Понятный итоговый отчёт</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_controls_and_outputs() -> None:
    st.markdown(
        """
<div class="split-showcase">
  <div class="surface-card">
    <div class="card-title">Что можно менять</div>
    <ul class="landing-list">
      <li>Цена</li>
      <li>Скидка и промо</li>
      <li>Логистика</li>
      <li>Внешние факторы</li>
      <li>Дополнительные сценарные шоки</li>
    </ul>
    <div class="mini-control-panel">
      <div>Цена: <b>1290 ₽</b></div>
      <div class="mini-slider"><span>Промо</span><div></div></div>
      <div class="mini-slider"><span>Логистика</span><div></div></div>
      <div class="floating-chip">Шок +5%</div>
      <button>Применить сценарий</button>
    </div>
  </div>
  <div class="surface-card">
    <div class="card-title">Что вы получаете</div>
    <ul class="landing-list">
      <li>Новый прогноз спроса</li>
      <li>Изменение выручки</li>
      <li>Изменение прибыли</li>
      <li>Уровень надёжности</li>
      <li>Готовый управленческий отчёт</li>
    </ul>
    <div class="mini-dashboard">
      <div class="mini-dashboard-grid">
        <div><span>Спрос</span><b>+8.4%</b></div>
        <div><span>Выручка</span><b>+124k</b></div>
        <div><span>Прибыль</span><b>+48k</b></div>
      </div>
      <div class="report-badge">Надёжность: Средняя</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_pipeline() -> None:
    st.markdown(
        """
<div class="section-title">Как это работает</div>
<div class="pipeline-row">
  <div class="pipeline-step"><span>🧾</span><b>Загрузите данные</b></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><span>📈</span><b>Получите базовый прогноз</b></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><span>🎛️</span><b>Измените сценарий</b></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step"><span>✅</span><b>Сравните и примите решение</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_trust_v2() -> None:
    st.markdown('<div class="section-title">Почему можно доверять</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="surface-card"><div class="card-title">Прозрачный сценарий</div><div class="muted">Видно, какие параметры попали в расчёт.</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="surface-card"><div class="card-title">Понятный результат</div><div class="muted">Сразу видно, как меняются спрос, выручка и прибыль.</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="surface-card"><div class="card-title">Надёжность оценки</div><div class="muted">Отчёт показывает уровень уверенности и предупреждения.</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="surface-card mini-report-preview">
  <div class="mini">Итог отчёта</div>
  <div class="report-badge">Рекомендуется к пилоту</div>
  <div class="mini-dashboard-grid">
    <div><span>Δ спроса</span><b>+8.4%</b></div>
    <div><span>Δ выручки</span><b>+124k</b></div>
    <div><span>Δ прибыли</span><b>+48k</b></div>
  </div>
  <div class="mini">Почему результат такой: цена и промо усиливают прогноз, логистика сдерживает рост.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_cta_v2() -> str | None:
    st.markdown(
        """
<div class="surface-card" style="text-align:center; position:relative;">
  <div class="section-title">Запустите первый сценарий за несколько минут</div>
  <div class="muted">Загрузите данные, измените параметры и сразу получите понятный результат.</div>
  <div class="cta-float cta-a">Scenario applied</div>
  <div class="cta-float cta-b">Profit +48k</div>
  <div class="cta-float cta-c">High confidence</div>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    if c1.button("Перейти в приложение", type="primary", use_container_width=True, key="landing_cta_start_v2"):
        return "app"
    if c2.button("Посмотреть интерфейс", use_container_width=True, key="landing_cta_demo_v2"):
        return "app"
    return None


def render_landing_footer() -> None:
    st.markdown('<div class="mini" style="text-align:center;padding:20px 0;">What-if Cloud</div>', unsafe_allow_html=True)
