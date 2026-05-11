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
    assert '"Цена"' in src
    assert '"Анализ решений"' in src
    assert '"Сравнение"' in src
    assert '"Отчёт"' in src
    assert '"Лучший вариант цены": "Цена"' in src
    assert '"Проверка решений": "Анализ решений"' in src


def test_decision_analyzer_has_no_visible_decision_layer_wording():
    src = Path("app.py").read_text()
    assert "Decision layer: проверяет варианты сценариев" not in src
    assert "Проверьте управленческую гипотезу" in src
    assert "Анализатор не меняет модель" in src


def test_decision_analyzer_uses_human_metric_labels():
    src = Path("app.py").read_text()
    assert "Надёжность решения" in src or "Надежность решения" in src
    assert "Изменение спроса" in src
    assert "Изменение выручки" in src
    assert "Осторожная оценка прибыли" in src


def test_decision_json_is_hidden_in_technical_expander():
    src = Path("app.py").read_text()
    assert "Для аналитика: технические детали" in src
    assert "Технический JSON паспорта" in src or "decision_passport" in src
