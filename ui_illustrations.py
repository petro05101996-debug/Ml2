from __future__ import annotations

import streamlit as st


def _svg_card(title: str, body_svg: str) -> None:
    st.markdown(
        f"""
<div class="enterprise-card secondary-card diagram-card">
  <div class="card-title">{title}</div>
  {body_svg}
</div>
""",
        unsafe_allow_html=True,
    )


def render_model_pipeline_diagram() -> None:
    _svg_card(
        "Пайплайн модели",
        """
<svg viewBox="0 0 900 140" class="diagram-svg">
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#7da8ff"/></marker></defs>
  <g fill="#0f1a2e" stroke="#365280" stroke-width="1.5">
    <rect x="10" y="25" width="150" height="70" rx="10"/><rect x="190" y="25" width="150" height="70" rx="10"/>
    <rect x="370" y="25" width="150" height="70" rx="10"/><rect x="550" y="25" width="170" height="70" rx="10"/><rect x="750" y="25" width="140" height="70" rx="10"/>
  </g>
  <g fill="#dfe9ff" font-size="14" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="85" y="66">Данные</text><text x="265" y="66">Подготовка</text><text x="445" y="66">Обучение</text>
    <text x="635" y="66">Сценарный движок</text><text x="820" y="66">Рекомендация</text>
  </g>
  <g stroke="#7da8ff" stroke-width="2" marker-end="url(#arrow)">
    <line x1="160" y1="60" x2="188" y2="60"/><line x1="340" y1="60" x2="368" y2="60"/><line x1="520" y1="60" x2="548" y2="60"/><line x1="720" y1="60" x2="748" y2="60"/>
  </g>
</svg>
""",
    )


def render_factors_diagram() -> None:
    _svg_card(
        "Факторы входа",
        """
<div class="factor-grid">
  <span>Цена</span><span>Скидка</span><span>Себестоимость</span><span>Логистика</span>
  <span>Запас</span><span>Промо</span><span>Рейтинг/Отзывы</span><span>Пользовательские факторы</span>
</div>
""",
    )


def render_workflow_diagram() -> None:
    _svg_card(
        "Пользовательский процесс",
        """
<div class="flow-row"><div>Загрузка</div><div>Сопоставление</div><div>Настройка</div><div>Запуск</div><div>Проверка</div></div>
""",
    )


def render_scenario_comparison_visual() -> None:
    _svg_card(
        "Сравнение сценариев",
        """
<div class="compare-bars">
  <div><label>Текущий</label><div class="bar current" style="width:52%"></div></div>
  <div><label>Рекомендованный</label><div class="bar reco" style="width:78%"></div></div>
  <div><label>Консервативный</label><div class="bar cons" style="width:61%"></div></div>
</div>
""",
    )


def render_result_interpretation_visual() -> None:
    _svg_card(
        "Как читать результат",
        """
<div class="interp-grid">
  <div><strong>Действие</strong><p>Что делать сейчас</p></div>
  <div><strong>Эффект</strong><p>Прибыль, выручка, объём</p></div>
  <div><strong>Риск</strong><p>Контроль негативного сценария</p></div>
  <div><strong>Уверенность</strong><p>Индикатор надёжности</p></div>
  <div><strong>Режим</strong><p>Пилот / поэтапное внедрение</p></div>
</div>
""",
    )
