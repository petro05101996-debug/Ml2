# Production-ready v1 status

## Сделано

- Backend API aliases присутствуют: `/api/health`, `/api/scenarios/templates`, `/api/dialog/start`, `/api/dialog/answer`, `/api/analyze/proposal`, `/api/analyze/scenario`, `/api/analyze/compare`, `/api/analyze/portfolio`, `/api/report/generate`, `/api/compliance/check-text`, `/api/knowledge/instrument/{type}`.
- Recommendation firewall реализован в `investment_lab/engine/safety_text_guard.py`.
- Deterministic offer parser реализован в `investment_lab/engine/offer_text_parser.py`.
- Risk flags и unknown field payload реализованы в `investment_lab/engine/risk_flag_engine.py`.
- Добавлен расчётный модуль `investment_lab/engine/scenario_calculator.py` с вычислениями:
  - `base_result`, `stress_result`, `fees_impact`, `tax_impact`, `inflation_impact`, `liquidity`, `risk`, `complexity`, `sensitivity_summary`;
  - portfolio `composition`, `largest_position_share`, `concentration`, `liquidity`, `stress_result`.
- Legal-safe report payload реализован в `report_builder.py`.
- Добавлен frontend runtime shell `frontend/src/App.tsx` + `frontend/src/main.tsx` + API helper `frontend/src/api.ts`.
- Реализованы API-вызовы frontend -> backend для proposal/scenario с обработкой ошибок и сохранением результатов между шагами.
- Добавлен dark enterprise базовый стиль в `frontend/src/styles.css` (graphite/navy, soft borders, card surface, cyan accent).
- Добавлены страницы и UI-компоненты по требуемому дереву `frontend/src/...`.
- Добавлены тесты: safety wording API/guard, static wording, report legal safety.

## Что осталось

1. Полная production UX-валидация flow:
   - связать все шаги с обязательным подтверждением допущений перед расчётом и строгой валидацией пользовательских полей.
2. Полная интеграция всех вспомогательных страниц:
   - часть страниц/компонентов существует как scaffold и требует полного runtime подключения к API (portfolio/explain/report пути).
3. Полный техпрогон в целевом окружении:
   - `pytest -q` по всему репо,
   - `cd frontend && npm ci && npm run build`,
   - `docker build -t investment-scenario-lab .` и запуск smoke-check.

## Критерий фактической готовности

Legal-safe API, расчётная база scenario/portfolio и frontend↔backend интеграция для ключевых шагов реализованы; для полного production-ready v1 остаются финальная UX-интеграция всех flow-веток и полный техпрогон в окружении с доступными зависимостями.
