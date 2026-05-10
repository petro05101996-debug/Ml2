# ML Pricing What-if v1

Приложение поддерживает production-v1 контур:

Universal CSV → stable weekly baseline (`legacy_baseline`) → scenario recompute / enhanced local factor layer → manual shock overlay.

Дополнительный режим `catboost_full_factors` доступен как расширенный сценарный режим: он повторно прогнозирует спрос по изменённым факторам, но не является production default.

## Возможности

- Universal CSV загрузка с auto-mapping и нормализацией.
- Stable v1 runtime: `legacy_baseline` по умолчанию.
- What-if сценарии: цена, скидка, промо, логистика, внешний спрос.
- Расширенный режим CatBoost full factors: модель переоценивает спрос по изменённым факторам.
- Decision layer: проверяет сценарии, оценивает риск, экономический эффект и формирует план теста.
- Decision layer не является причинным доказательством и не гарантирует глобальный оптимум.

## Контракт v1

- Production default: `legacy_baseline`.
- Non-legacy weekly candidates используются как диагностика, если runtime frozen to legacy.
- CatBoost full factors включается только выбранным режимом анализа.
- Decision layer ищет лучший найденный вариант среди проверенных сценариев, а не математически гарантированный глобальный оптимум.
- Demand shock является ручной гипотезой, а не автоматически выученным причинным эффектом.

## Запуск
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Тесты
```bash
pytest -q
```
