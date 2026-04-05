# Demand What-If Studio

Приложение для сценарного прогнозирования спроса (а не price recommender):
- отдельный baseline-слой,
- отдельный factor effect,
- сценарный what-if запуск,
- экспорт отчёта.

## Запуск
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Поддерживаемые входы
- Транзакции: `.csv`, `.xlsx`
- Документы событий: `.pdf`, `.docx` (пока как UI-заглушка для будущего document-to-facts потока)

Минимально обязательные поля: `date`, `product_id`, `price`.

## Каноническая логика v2
1. Нормализация входа (`data_adapter.py`)
2. Построение панелей по series scope
3. Baseline прогноз
4. Factor response
5. Scenario forecast
6. Economics (revenue/profit/margin)
7. Export

## Grain и агрегация
В панели используется series scope:
- `product_id`
- `category`
- `region`
- `channel`
- `segment`

Добавляется `series_id = product_id|region|channel|segment`.

## Price/discount semantics
По умолчанию:
- `price` = net selling price
- `discount` = explanatory signal

Для list-price режима доступен economics mode `list_less_discount`.

## Fallback сценарный режим
Если factor model недоступна, используется deterministic fallback (`fallback_elasticity`) вместо молчаливого `baseline_only`.

## Ограничения и доверие
- confidence остаётся advisory,
- при короткой истории рекомендации нестабильны,
- `cost` при отсутствии может проксироваться как `0.65 * price`.

## Тесты
```bash
pytest -q
```
