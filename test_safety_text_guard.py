from investment_lab.engine.safety_text_guard import check_text_safety, sanitize_advisory_text


def test_forbidden_phrase_detected_and_sanitized():
    text = "Сценарий A лучше соответствует ограничениям"
    result = check_text_safety(text)
    assert result.is_safe is False
    assert "лучше соответствует" in result.violations
    assert "Формулировка заменена на нейтральную" in result.sanitized_text


def test_safe_text_remains_safe():
    text = "Расчёт основан на введённых данных и допущениях."
    result = check_text_safety(text)
    assert result.is_safe is True
    assert result.violations == []


def test_sanitize_regex_variations():
    text = "Мы рекомендуем этот вариант"
    out = sanitize_advisory_text(text)
    assert "информационно-аналитический характер" in out
