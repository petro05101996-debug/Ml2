from __future__ import annotations

import streamlit as st
from ui_illustrations import (
    render_result_interpretation_visual,
    render_scenario_comparison_visual,
)


def render_docs() -> None:
    st.markdown('<div class="page-title">Модель и документация</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Структурированное описание модели для клиентов, аналитиков и коммерческих команд.</div>', unsafe_allow_html=True)

    sections = [
        ("Что это за модель", "Сценарная модель поддержки решений по цене, а не полностью автономный ценовой автопилот."),
        ("Что она рассчитывает", "Ожидаемые изменения спроса, выручки и прибыли, сравнение сценариев и риск-ориентированную рекомендацию."),
        ("Какие факторы использует", "Цена, скидка, себестоимость, логистика, запас, промо, рейтинг/отзывы и пользовательские факторы из данных."),
        ("Как работает сценарный анализ", "Формируется baseline, применяются изменения сценария, сравниваются результаты, выбирается лучший вариант."),
        ("Как формируется рекомендация", "Ожидаемый эффект корректируется с учётом риска и уверенности, после чего выдаётся бизнес-рекомендация."),
        ("Технологический стек", "CatBoost/предиктивное моделирование, сценарный движок, sensitivity-анализ, unit economics и бизнес-ограничения."),
        ("Практические применения", "Проверка роста цены, тест скидочной политики, стресс-тест издержек, подготовка пилота внедрения."),
        ("Ограничения и допущения", "Качество результата зависит от входных данных; это поддержка решений, а не гарантия будущего; рыночные сдвиги могут изменить эффект."),
    ]
    for title, body in sections:
        st.markdown(f'<div class="enterprise-card secondary-card"><div class="card-title">{title}</div><div class="muted">{body}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="enterprise-card flat-card"><div class="card-title">Пошаговая инструкция</div><ol><li>Загрузите данные</li><li>Проверьте сопоставление полей</li><li>Выберите категорию и SKU</li><li>Настройте сценарий</li><li>Запустите анализ</li><li>Сравните результаты</li><li>Используйте рекомендацию для пилота/решения</li></ol></div>', unsafe_allow_html=True)

    render_scenario_comparison_visual()
    render_result_interpretation_visual()
