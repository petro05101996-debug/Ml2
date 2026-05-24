from report_builder import build_business_report_payload


def test_report_contains_required_legal_phrases():
    out = build_business_report_payload(payload={})
    body = " ".join([out.get("title", ""), out.get("subtitle", ""), out.get("disclaimer_top", ""), out.get("disclaimer_bottom", "")]).casefold()
    assert "отчёт проверки условий и сценариев" in body
    assert "не является индивидуальной инвестиционной рекомендацией" in body
    assert "не определяет пригодность инструмента" in body
    assert "не формирует инвестиционный профиль" in body
    assert "не содержит предложения купить, продать или удерживать" in body


def test_report_has_no_forbidden_winner_wording():
    out = build_business_report_payload(payload={})
    body = str(out).casefold()
    for phrase in ["лучше соответствует", "с максимальным соответствием", "балл соответствия", "подходит пользователю"]:
        assert phrase not in body
