# ML Pricing App — Universal CSV What-if v1

Streamlit-приложение для диагностики ценовых решений: загружает произвольный CSV с продажами, приводит его к единому контракту данных, строит недельный baseline и считает прозрачные what-if сценарии по цене, скидкам, промо, логистике и внешнему demand shock.

Основной production-контур сейчас один: **Universal CSV → нормализация и quality gate → stable weekly baseline (`legacy_baseline`) → `enhanced_local_factors` what-if layer → decision/recommendation guardrails → экспорт результатов**.

`catboost_full_factors` оставлен как расширенный диагностический режим: он переоценивает спрос по изменённым факторам, но не является default-контуром и требует достаточной исторической поддержки.

## Что делает продукт

- Принимает Universal CSV и выполняет auto-mapping колонок к канонической схеме.
- Нормализует транзакции, скидки, выручку, себестоимость, логистику и доступные признаки спроса.
- Строит стабильный недельный baseline без утечки будущих наблюдений в feature/data preparation.
- Считает what-if сценарии v1 через `enhanced_local_factors`: цена, скидка, промо, логистика и ручной внешний demand shock.
- Явно разделяет baseline, raw scenario metrics, adjusted/economic metrics и итоговые business deltas.
- Применяет guardrails исторической поддержки цены: `safe_clip` по умолчанию и экспериментальный `economic_extrapolation` для тестовых сценариев.
- Проверяет качество данных, надёжность модели и применимость рекомендаций через production/decision gates.
- Формирует decision passport: статус решения, риск, ожидаемый эффект, условия пилота и rollback-критерии.
- Поддерживает аудит внешней рекомендации: приложение проверяет предложенную пользователем идею через тот же сценарный и decision-контур.
- Добавляет scenario audit и reproducibility metadata: параметры сценария, режим расчёта, hash датасета и run id.
- Экспортирует результаты в CSV/JSON/Excel, включая листы по decision gate, model quality, scenario audit, feature usage и ограничениям.

## Контракт v1

- Активный пользовательский контур: **Universal CSV what-if v1**.
- Production what-if default: `enhanced_local_factors`.
- Baseline forecast path: stable weekly ML baseline на основе `legacy_baseline`.
- Финальный пользовательский путь: `weekly_ml_baseline + enhanced_local_factor_layer`.
- `legacy_current` используется только для совместимости и базового пересчёта.
- `catboost_full_factors` доступен только как advanced/diagnostic режим и может быть автоматически заменён на `enhanced_local_factors`, если backend или данные недостаточны.
- Production bundle после диагностики строится на full-data refit, чтобы финальный артефакт использовал всю доступную историю.
- What-if multipliers должны материально влиять на output metrics; сценарии без изменений должны совпадать с as-is расчётом.

## Данные и качество

Минимально ожидаемые поля зависят от auto-mapping, но для production-рекомендаций критичны дата, товар/SKU, цена и объём продаж. Если часть полей отсутствует, приложение пытается безопасно восстановить их и помечает источник в контракте данных.

Важные правила:

- `quantity` может быть выведено из `revenue / net_price`; если это невозможно, production-рекомендации блокируются.
- `cost` может быть proxied как `0.65 * price`, но такие profit-рекомендации считаются test-only, а не production.
- Малые датасеты, слабая ценовая вариативность, мало price changes или низкая историческая поддержка сценария снижают статус рекомендации.
- Confidence является эвристической оценкой надёжности, а не гарантией результата.

## Режимы расчёта

| Режим | Назначение | Статус |
| --- | --- | --- |
| `enhanced_local_factors` | Baseline + прозрачные локальные эффекты цены/скидки/промо/логистики/shock | production default |
| `legacy_current` | Старый стабильный пересчёт для совместимости | compatibility |
| `catboost_full_factors` | Модельный what-if по изменённым факторам | advanced diagnostic |

## Guardrails цены

- `safe_clip` — режим по умолчанию: спрос и финансы считаются по цене внутри исторического диапазона поддержки.
- `economic_extrapolation` — экспериментальный режим: спрос ограничивается границей исторической поддержки с elasticity-tail за её пределами, а финансы считаются по введённой цене. Используйте только как тестовую гипотезу.

## Что продукт НЕ делает

- Не гарантирует глобальный оптимум цены.
- Не доказывает причинно-следственную связь.
- Не заменяет A/B-тест, контролируемый пилот или бизнес-экспертизу.
- Не должен автоматически рекомендовать действия при плохих данных, proxy-cost, малой истории или опасной экстраполяции.
- Не является единственным источником управленческого решения.

## Локальный запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Тесты

```bash
pytest -q
```

## Быстрый production-чеклист

Перед использованием результата как рекомендации убедитесь, что:

1. Universal CSV распознан корректно, а data quality gate не содержит hard blockers.
2. `cost` предоставлен в данных, а не заменён proxy `0.65 * price`, если решение основано на profit.
3. Сценарий рассчитан в `enhanced_local_factors` и имеет достаточную историческую поддержку цены.
4. Raw, adjusted и baseline metrics не смешиваются в интерпретации.
5. Decision passport не помечает решение как `not_recommended` или diagnostic/test-only без явного пилотного контекста.
6. Для production-вывода выполнен full-data refit и сохранены scenario audit/reproducibility metadata.
