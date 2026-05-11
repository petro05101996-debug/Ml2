from pathlib import Path


def test_decision_context_has_user_friendly_labels():
    src = Path("app.py").read_text()
    assert "Что взять за отправную точку" in src
    assert "Базовый прогноз" in src
    assert "Рассчитанный сценарий" in src


def test_upload_mapping_has_human_labels():
    src = Path("app.py").read_text()
    assert "CANONICAL_FIELD_UI" in src
    assert "Дата операции" in src
    assert "SKU / товар" in src
    assert "Цена продажи" in src
    assert "Продажи, шт." in src


def test_saved_scenarios_have_user_friendly_names():
    src = Path("app.py").read_text()
    assert "Вариант 1" in src
    assert "Вариант 2" in src
    assert "Вариант 3" in src


def test_workspace_guide_is_rendered():
    src = Path("app.py").read_text()
    assert "render_workspace_guide" in src


def test_decision_action_labels_exist():
    src = Path("app.py").read_text()
    assert "action_type_label" in src
    assert "Изменить цену" in src
    assert "Изменить скидку" in src
    assert "Изменить промо" in src
    assert "Проверить внешнюю гипотезу спроса" in src


def test_no_duplicate_static_streamlit_keys():
    import re
    src = Path("app.py").read_text()
    keys = re.findall(r'key\s*=\s*["\']([^"\']+)["\']', src)
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicates, f"Duplicate static Streamlit keys: {duplicates}"


def test_decision_results_are_signature_guarded():
    src = Path("app.py").read_text()
    assert "decision_optimizer_signature" in src
    assert "recommendation_audit_signature" in src
    assert "Параметры изменились" in src
    assert "Параметры рекомендации изменились" in src


def test_user_ui_does_not_use_slot_wording_for_saved_scenarios():
    src = Path("app.py").read_text()
    assert 'st.selectbox("Слот сохранения"' not in src
    assert 'st.button("Сохранить в слот"' not in src
    assert "Куда сохранить вариант" in src
    assert "Сохранить вариант" in src


def test_comparison_table_has_human_column_names():
    src = Path("app.py").read_text()
    assert '"Надёжность"' in src or '"Надежность"' in src
    assert '"Поддержка данных"' in src
    assert '"Цена ограничена защитой"' in src
    assert '"Предупреждения"' in src
    assert '"Цена до скидки"' in src
    assert '"Цена после скидки"' in src
    assert '"Confidence"' not in src
    assert '"Support"' not in src
    assert '"Warnings"' not in src
