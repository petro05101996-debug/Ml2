from __future__ import annotations

import streamlit as st
from ui_illustrations import (
    render_factors_diagram,
    render_model_pipeline_diagram,
    render_workflow_diagram,
)


def render_overview() -> None:
    st.markdown('<div class="enterprise-card primary-card">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Студия ценовой аналитики v1.0.0</div>', unsafe_allow_html=True)
    st.markdown("**Инструмент поддержки решений по цене для продавцов и аналитиков.**")
    st.markdown('<div class="muted">Оцените влияние на спрос, выручку и прибыль до изменения цены на рынке.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Перейти к настройке", type="primary", width="stretch"):
            st.session_state.active_page = "Настройка"
            st.rerun()
    with c2:
        if st.button("Как это работает", width="stretch"):
            st.session_state.active_page = "Модель и документация"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="enterprise-card secondary-card"><div class="card-title">Что делает модель</div><div class="muted">Использует исторические продажи и бизнес-факторы, оценивает изменения спроса, выручки и прибыли, сравнивает сценарии и формирует рекомендацию для решения.</div></div>', unsafe_allow_html=True)

    caps = [
        "Сценарный анализ цен",
        "Оценка эффекта на прибыль",
        "Чувствительность спроса",
        "Риск-ориентированные рекомендации",
        "Прогноз с учётом факторов",
        "Поддержка решения для продавца",
    ]
    cols = st.columns(3)
    for i, cap in enumerate(caps):
        cols[i % 3].markdown(f'<div class="enterprise-card flat-card"><div class="card-title">{cap}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="enterprise-card secondary-card"><div class="card-title">Как работает процесс</div><div class="muted">1) Загрузка данных → 2) Сопоставление полей → 3) Настройка анализа → 4) Запуск сценариев → 5) Просмотр рекомендации и эффекта.</div></div>', unsafe_allow_html=True)

    st.markdown('### Поясняющие схемы продукта')
    render_model_pipeline_diagram()
    render_factors_diagram()
    render_workflow_diagram()

    uc1, uc2 = st.columns(2)
    with uc1:
        st.markdown('<div class="enterprise-card flat-card"><div class="card-title">Практические кейсы</div><ul><li>Проверить безопасное повышение цены</li><li>Оценить влияние скидки на прибыль</li><li>Сравнить осторожный и агрессивный шаг</li><li>Проверить стресс при росте логистики/себестоимости</li><li>Решить: держать цену или тестировать новую</li></ul></div>', unsafe_allow_html=True)
    with uc2:
        st.markdown('<div class="enterprise-card flat-card"><div class="card-title">Чем отличается от простых инструментов</div><ul><li>Не только статический отчёт (Excel/BI)</li><li>Учитывает взаимодействие факторов</li><li>Сценарное рассуждение, а не фиксированные правила</li><li>Даёт контекст риска</li><li>Формирует рекомендацию, а не только цифры</li></ul></div>', unsafe_allow_html=True)

    st.markdown('<div class="enterprise-card secondary-card"><div class="card-title">Почему инструменту можно доверять</div><div class="muted">Рекомендация строится на ваших исторических данных, бизнес-факторах и сравнении нескольких сценариев. Это поддержка решения, а не автоматический автопилот.</div></div>', unsafe_allow_html=True)
