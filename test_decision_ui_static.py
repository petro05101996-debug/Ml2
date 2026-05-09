from pathlib import Path


def test_missing_trained_bundle_shows_ui_warning_not_keyerror():
    src = Path("app.py").read_text()
    assert 'not r.get("_trained_bundle")' in src
    assert 'Сначала выполните базовый анализ' in src


def test_audit_ui_uses_action_specific_inputs():
    src = Path("app.py").read_text()
    assert 'Новая скидка, %' in src
    assert 'Ожидаемое изменение спроса, %' in src
    assert 'Есть внешнее обоснование гипотезы' in src
