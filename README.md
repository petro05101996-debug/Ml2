# ML Pricing What-if v1

Приложение поддерживает **один активный контур**: Universal CSV → baseline + elasticity + uplift → holdout diagnostics → full-history refit → what-if simulation.

## Возможности
- Universal CSV загрузка с auto-mapping и нормализацией.
- Leak-safe daily pipeline (без backward fill из будущего).
- Раздельные метрики прибыли: `profit_raw`, `profit_adjusted`, `uncertainty_penalty`.
- Ручной what-if, мультисценарии, sensitivity heatmap (`price × demand_multiplier`).
- **Excel export**: `history`, `neutral_baseline`, `as_is`, `metrics`.
- **CSV export**: `holdout_predictions`, `scenario_output`, `feature_report`.

## Запуск
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Тесты
```bash
pytest -q
```
