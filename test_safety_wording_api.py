import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_compliance_contract():
    out = client.post("/api/compliance/check-text", json={"text": "подходит вам"}).json()
    assert set(out.keys()) == {"is_safe", "violations", "sanitized_text", "disclaimer_required"}
    assert out["is_safe"] is False


def test_health_contract():
    out = client.get("/api/health").json()
    assert out == {"status": "ok", "service": "investment-scenario-lab"}


def test_alias_endpoints_and_proposal_shape():
    payload = {"text": "Доходность до 18% годовых, срок 12 месяцев, только сегодня, без риска.", "source": "реклама"}
    proposal = client.post("/api/analyze/proposal", json=payload).json()
    assert proposal["status"] == "ok"
    assert "parsed" in proposal and "risk_flags" in proposal and "unknown_fields" in proposal
    assert proposal["parsed"]["declared_return"] == "18"
    assert proposal["parsed"]["term_months"] == 12
    assert any(x["code"] == "UNKNOWN_FEES" for x in proposal["unknown_fields"])

    scenario = client.post("/api/analyze/scenario", json={}).json()
    for key in ["base_result", "stress_result", "fees_impact", "tax_impact", "inflation_impact", "liquidity", "risk", "complexity", "risk_flags", "unknown_fields", "assumptions", "limitations", "sensitivity_summary"]:
        assert key in scenario

    portfolio = client.post("/api/analyze/portfolio", json={}).json()
    for key in ["composition", "largest_position_share", "asset_class_shares", "concentration", "liquidity", "stress_result", "risk_flags", "unknown_fields", "checklist"]:
        assert key in portfolio

    assert client.post("/api/analyze/compare", json={}).status_code == 200
    assert client.get("/api/scenarios/templates").status_code == 200
    assert client.post("/api/dialog/start", json={}).status_code == 200
    assert client.post("/api/dialog/answer", json={}).status_code == 200
    assert client.get("/api/knowledge/instrument/ofz").status_code == 200
    assert client.post("/api/report/generate", json={}).status_code == 200


def test_source_risk_telegram_social():
    out = client.post("/api/analyze/proposal", json={"text": "до 15%", "source": "Telegram/соцсети"}).json()
    assert out["parsed"]["source_risk"] == "NON_OFFICIAL_SOURCE"

def test_report_has_sections():
    out = client.post("/api/report/generate", json={"offer": {}, "scenario": {}}).json()
    assert "sections" in out and isinstance(out["sections"], list) and len(out["sections"]) >= 10
