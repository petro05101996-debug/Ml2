# ML Pricing What-if v1

Приложение поддерживает **один активный production-контур**: Universal CSV → weekly baseline (`price_promo_freight_baseline`) → scenario recompute (`price/promo/freight`) → manual shock.

## Возможности
- Universal CSV загрузка с auto-mapping и нормализацией.
- Leak-safe daily pipeline (без backward fill из будущего).
- Прозрачный what-if: `price`, `promo`, `freight`, `manual shock`.
- Раздельные метрики: baseline / as-is / scenario, плюс дельты по units/revenue/profit.
- Learned uplift оставлен только в диагностике (в runtime не участвует).
- **Excel export**: `history`, `neutral_baseline`, `as_is`, `metrics`.
- **CSV export**: `holdout_predictions`, `analysis_baseline_vs_as_is`, `manual_scenario_daily`, `feature_report`.

## Контракт v1
- Активный путь по умолчанию: `price_promo_freight_baseline+scenario_recompute`.
- Legacy/naive путь используется только как аварийный fallback (когда weekly ML не проходит проверку качества или данных недостаточно).
- Для `price/promo/freight` нет двойного учёта: эффекты применяются один раз через scenario engine поверх baseline units.
- `shock` применяется отдельно как прозрачный post-model multiplier.

## Запуск
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Тесты
```bash
pytest -q
```
