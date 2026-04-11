# AI Dynamic Pricing Dashboard (Streamlit MVP)

MVP-приложение для динамического ценообразования: загрузка универсальных данных, оценка эластичности, бизнес-рекомендации и сценарный what-if анализ.

## Что умеет MVP
- **Универсальный ввод данных** через 1 CSV с автосопоставлением и ручным маппингом колонок.
- Поддержка legacy-потока Olist (Orders/Items/Products/Reviews) без поломки старой логики.
- Нормализация в каноническую схему и **проверки качества данных** (пропуски, дубликаты, короткая история, выбросы).
- ML-прогноз спроса/прибыли с оценкой confidence.
- Режимы оптимизации: `maximize_profit`, `maximize_revenue`, `protect_volume`, `control_margin`, `custom_objective`.
- Полноценный **What-if**: baseline + 3 сценария, сравнение сценариев и sensitivity heatmap (price x demand).
- Экспорт в Excel (history, baseline, optimal, metrics, recommendation).

## Каноническая схема (универсальный CSV)

### Обязательные поля
- `date` (или timestamp/date aliases)
- `product_id` (SKU)
- `price`

### Опциональные поля
- `category`, `quantity`, `revenue`, `cost`, `discount`, `freight_value`, `stock`, `promotion`, `rating`, `reviews_count`, `region`, `channel`, `segment`

Если опциональные поля отсутствуют, приложение использует fallback:
- `quantity = 1`
- `revenue = price * quantity`
- `cost = 0.65 * price`

## Как работает recommendation engine
1. Нормализует вход в каноническую схему.
2. Строит дневной ряд по SKU и признаки (лаги, сезонность, динамика цены).
3. Обучает ансамбль моделей и оценивает качество на holdout.
4. Подбирает цену, максимизирующую выбранную objective-функцию.
5. Формирует бизнес-рекомендацию: действие, почему, ожидаемый эффект, риск, confidence, альтернатива, reason codes.

## How-to: What-if панель
1. Загрузите данные и запустите анализ.
2. В блоке What-if задайте baseline + минимум 3 сценария.
3. Для каждого сценария меняйте: price, demand multiplier, freight, cost, horizon.
4. Нажмите **«Сравнить 4 сценария»**.
5. Смотрите:
   - таблицу baseline vs scenarios;
   - график сравнения сценариев;
   - sensitivity heatmap (price x demand).

## Запуск
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Структура
- `app.py` — UI и основная оркестрация
- `data_schema.py` — канонические поля и alias-правила
- `data_adapter.py` — маппинг, нормализация, quality checks
- `recommendation.py` — бизнес-рекомендации + reason codes
- `what_if.py` — multi-scenario расчёт и sensitivity grid

## Примечания по устойчивости
- Приложение не падает при неполных данных и показывает ограничения в виде предупреждений.
- При низком объёме/качестве данных confidence снижается, рекомендации становятся более консервативными.
