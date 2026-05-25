from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.compliance import router as compliance_router
from backend.app.api.explain import router as explain_router
from investment_lab.engine.offer_text_parser import parse_offer_text
from investment_lab.engine.risk_flag_engine import flags_from_offer, unknown_field_payload
from investment_lab.engine.safety_text_guard import sanitize_advisory_text
from investment_lab.engine.scenario_calculator import calculate_scenario, calculate_portfolio

app = FastAPI(title="investment-scenario-lab")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "investment-scenario-lab"}


@app.get("/api/scenarios/templates")
def scenario_templates() -> dict:
    return {"templates": ["single_scenario", "compare_scenarios", "portfolio_check", "instrument_explain"]}


QUESTIONS = [
    {"id":"amount","title":"Какую сумму вы хотите проверить?","type":"number","why_it_matters":"Сумма влияет на абсолютный размер итогового результата."},
    {"id":"term_months","title":"На какой срок планируется размещение?","type":"number","why_it_matters":"Срок влияет на доход, ликвидность и чувствительность к рискам."},
]

@app.post("/api/dialog/start")
def dialog_start(payload: dict) -> dict:
    return {"status":"ok","step":"question","question":QUESTIONS[0],"cursor":0,"answers":{}}


@app.post("/api/dialog/answer")
def dialog_answer(payload: dict) -> dict:
    cursor = int(payload.get("cursor", 0))
    answers = dict(payload.get("answers", {}))
    if "answer" in payload and "question_id" in payload:
        answers[str(payload.get("question_id"))] = payload.get("answer")
    nxt = cursor + 1
    if nxt >= len(QUESTIONS):
        return {"status":"ok","step":"preview","answers":answers}
    return {"status":"ok","step":"question","question":QUESTIONS[nxt],"cursor":nxt,"answers":answers}


@app.post("/api/analyze/proposal")
def analyze_proposal(payload: dict) -> dict:
    text = str(payload.get("text", ""))
    source = payload.get("source")
    parsed = parse_offer_text(text, source=source)
    risk_flags = flags_from_offer(parsed)
    unknown = [unknown_field_payload(code) for code in parsed.get("unknown_fields", [])]
    summary = sanitize_advisory_text("Сервис нашёл нераскрытые условия и риск-флаги.", context="offer")
    return {"status": "ok", "kind": "proposal", "parsed": parsed, "risk_flags": risk_flags, "unknown_fields": unknown, "summary": summary}


@app.post("/api/analyze/scenario")
@app.post("/api/analyze/compare")
def analyze_scenario(payload: dict) -> dict:
    calc = calculate_scenario(payload)
    return {"status": "ok", "kind": "scenario", **calc, "risk_flags": payload.get("risk_flags", []), "unknown_fields": payload.get("unknown_fields", []), "assumptions": payload.get("assumptions", []), "limitations": payload.get("limitations", [])}


@app.post("/api/analyze/portfolio")
def analyze_portfolio(payload: dict) -> dict:
    calc = calculate_portfolio(payload)
    return {"status": "ok", "kind": "portfolio", **calc, "risk_flags": payload.get("risk_flags", []), "unknown_fields": payload.get("unknown_fields", []), "checklist": ["Проверьте концентрацию", "Проверьте ликвидность", "Сверьте комиссии и налоги"]}



@app.post("/api/report/generate")
def report_generate(payload: dict) -> dict:
    disclaimer = (
        "Отчёт носит информационно-аналитический характер. Он не является индивидуальной инвестиционной рекомендацией, "
        "не содержит предложения купить, продать или удерживать финансовый инструмент, не определяет пригодность инструмента "
        "для пользователя и не формирует инвестиционный профиль. Все расчёты основаны на введённых данных и допущениях."
    )
    offer = payload.get("offer", {}) if isinstance(payload, dict) else {}
    scenario = payload.get("scenario", {}) if isinstance(payload, dict) else {}
    sections = [
        {"title": "Краткое резюме", "items": [f"Обнаружено условий к уточнению: {len(offer.get('unknown_fields', []))}", f"Обнаружено риск-флагов: {len(offer.get('risk_flags', []))}"]},
        {"title": "Введённые данные", "items": [f"Текст предложения: {bool(offer.get('parsed'))}", f"Сценарий рассчитан: {bool(scenario)}"]},
        {"title": "Что удалось определить", "items": [f"{k}: {v}" for k,v in (offer.get('parsed', {}) or {}).items()]},
        {"title": "Что неизвестно", "items": [f"{u.get('title')}: {u.get('plain_explanation')}" for u in offer.get('unknown_fields', [])]},
        {"title": "Допущения", "items": scenario.get("assumptions", ["Использованы консервативные допущения для нераскрытых условий."])},
        {"title": "Базовый сценарий", "metrics": [{"label":"Чистый номинальный результат","value": scenario.get("base_result",{}).get("net_nominal")},{"label":"Результат с учётом инфляции","value": scenario.get("base_result",{}).get("net_real")} ]},
        {"title": "Стресс-сценарий", "metrics": [{"label":"Просадка","value": scenario.get("stress_result",{}).get("drawdown")},{"label":"Падение, %","value": scenario.get("stress_result",{}).get("stress_drop_pct")} ]},
        {"title": "Комиссии", "metrics": [{"label":"Уплачено комиссий","value": scenario.get("fees_impact",{}).get("fees_paid")}]},
        {"title": "Налоги", "metrics": [{"label":"Уплачено налогов","value": scenario.get("tax_impact",{}).get("tax_paid")}]},
        {"title": "Инфляция", "metrics": [{"label":"Инфляционное снижение","value": scenario.get("inflation_impact",{}).get("inflation_loss")}]},
        {"title": "Risk flags", "items": [f"{r.get('title')}: {r.get('why_it_matters')}" for r in offer.get('risk_flags', [])]},
        {"title": "Чеклист вопросов", "items": ["Уточните комиссии", "Уточните условия досрочного выхода", "Попросите официальный документ условий"]},
        {"title": "Ограничения расчёта", "items": scenario.get("limitations", ["Результат зависит от полноты и качества входных данных."])},
    ]
    return {"status": "ok", "kind": "report", "title": "Отчёт проверки условий и сценариев", "subtitle": "Информационно-аналитический расчёт по введённым данным. Не является индивидуальной инвестиционной рекомендацией.", "sections": sections, "disclaimer_top": disclaimer, "disclaimer_bottom": disclaimer, "payload": payload}


app.include_router(compliance_router)

app.include_router(explain_router)


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(FRONTEND_DIST / "index.html")
