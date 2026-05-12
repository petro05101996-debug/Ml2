from pathlib import Path


def test_missing_trained_bundle_shows_ui_warning_not_keyerror():
    src = Path("app.py").read_text()
    assert 'not r.get("_trained_bundle")' in src
    assert 'Сначала выполните базовый анализ' in src


def test_audit_ui_uses_action_specific_inputs():
    src = Path("app.py").read_text()
    assert 'Новая скидка, %' in src
    assert 'Ожидаемое внешнее изменение спроса, %' in src
    assert 'Есть внешнее подтверждение гипотезы' in src


def test_ui_has_user_friendly_tabs():
    src = Path("app.py").read_text()
    assert '"Цена-кандидат"' in src
    assert '"Проверка решения"' in src
    assert '"Сравнить варианты"' in src
    assert '"Скачать отчёт"' in src
    assert '"Лучший вариант цены": "Цена-кандидат"' in src
    assert '"Проверка решений": "Проверка решения"' in src


def test_decision_analyzer_has_no_visible_decision_layer_wording():
    src = Path("app.py").read_text()
    assert "Decision layer: проверяет варианты сценариев" not in src
    assert "Покажет, стоит ли запускать изменение" in src
    assert "Анализатор не меняет модель" in src


def test_decision_analyzer_uses_human_metric_labels():
    src = Path("app.py").read_text()
    assert "Надёжность решения" in src or "Надежность решения" in src
    assert "Изменение спроса" in src
    assert "Изменение выручки" in src
    assert "Осторожная оценка прибыли" in src


def test_decision_json_is_hidden_in_technical_expander():
    src = Path("app.py").read_text()
    assert "Для аналитика: технический файл решения" in src
    assert "Технический JSON паспорта" in src or "decision_passport" in src


def test_scenario_screen_has_no_price_sensitivity_artifact():
    src = Path("app.py").read_text()
    assert 'open_surface("Проверка соседних цен")' not in src
    assert 'Дополнительно: чувствительность к цене' not in src


def test_report_json_is_hidden_for_analyst():
    src = Path("app.py").read_text()
    assert 'open_surface("JSON для аналитика"' not in src
    assert 'with st.expander("Для аналитика: технический файл"' in src


def test_scenario_advanced_settings_are_not_main_user_path():
    src = Path("app.py").read_text()
    assert 'with st.expander("Для аналитика: расширенные настройки"' in src
    assert 'with st.expander("Редко используется: логистика и внешний спрос"' in src
