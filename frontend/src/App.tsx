import React, { useMemo, useState } from "react";
import LandingPage from "./pages/LandingPage";
import OfferCheckPage from "./pages/OfferCheckPage";
import OfferResultPage from "./pages/OfferResultPage";
import ScenarioModePage from "./pages/ScenarioModePage";
import ScenarioInputPage from "./pages/ScenarioInputPage";
import ScenarioPreviewPage from "./pages/ScenarioPreviewPage";
import ResultsPage from "./pages/ResultsPage";
import ReportPage from "./pages/ReportPage";
import PortfolioCheckPage from "./pages/PortfolioCheckPage";
import ExplainInstrumentPage from "./pages/ExplainInstrumentPage";
import OfferAssumptionsStep from "./components/offer/OfferAssumptionsStep";
import OfferParsedPreview from "./components/offer/OfferParsedPreview";
import { postJson } from "./api";

type Step = "landing" | "offer" | "offer_parsed_preview" | "offer_assumptions" | "offer_result" | "scenario_mode" | "scenario_input" | "scenario_preview" | "scenario_result" | "portfolio" | "explain" | "report";

export default function App() {
  const [step, setStep] = useState<Step>("landing");
  const [form, setForm] = useState({ offerText: "", offerSource: "банк", amount: "100000", term: "12", annual_return_pct: "12", fees_pct: "0", tax_pct: "13", inflation_pct: "7", stress_drop_pct: "10", earlyNeed: "не знаю", includeTaxes: true, includeInflation: true });
  const [offerResult, setOfferResult] = useState<any>(null);
  const [scenarioResult, setScenarioResult] = useState<any>(null);
  const [portfolioResult, setPortfolioResult] = useState<any>(null);
  const [knowledgeResult, setKnowledgeResult] = useState<any>(null);
  const [reportResult, setReportResult] = useState<any>(null);
  const [error, setError] = useState<string>("");
  const [assumptionsConfirmed, setAssumptionsConfirmed] = useState(false);
  const onChange = (patch: any) => setForm((s) => ({ ...s, ...patch }));

  const assumptions = ["Расчёт основан на введённых параметрах.", "Нераскрытые условия учитываются как допущения."];

  const safe = async (fn: () => Promise<void>) => { try { setError(""); await fn(); } catch (e: any) { setError(e.message || String(e)); } };
  async function runOfferParse() { const data = await postJson<any>("/api/analyze/proposal", { text: form.offerText, source: form.offerSource }); setOfferResult(data); setStep("offer_parsed_preview"); }
  async function runScenario() {
    if (!assumptionsConfirmed) throw new Error("Подтвердите допущения перед расчётом");
    const data = await postJson<any>("/api/analyze/scenario", { amount: Number(form.amount), term_months: Number(form.term), annual_return_pct: Number(form.annual_return_pct), fees_pct: Number(form.fees_pct), tax_pct: Number(form.tax_pct), inflation_pct: Number(form.inflation_pct), stress_drop_pct: Number(form.stress_drop_pct), include_taxes: form.includeTaxes, include_inflation: form.includeInflation, unknown_fields_count: (offerResult?.unknown_fields || []).length });
    setScenarioResult(data); setStep("scenario_result");
  }
  async function runPortfolio() { const data = await postJson<any>("/api/analyze/portfolio", { positions: [{ name: "ОФЗ", amount: 60000 }, { name: "Фонд", amount: 40000 }] }); setPortfolioResult(data); setStep("portfolio"); }
  async function runExplain(topic: string) { const res = await fetch(`/api/knowledge/instrument/${encodeURIComponent(topic)}`); if(!res.ok) throw new Error(`HTTP ${res.status}`); setKnowledgeResult(await res.json()); setStep("explain"); }
  async function runReport() { const data = await postJson<any>("/api/report/generate", { offer: offerResult || {}, scenario: scenarioResult || {}, portfolio: portfolioResult || {}, explain: knowledgeResult || {} }); setReportResult(data); setStep("report"); }

  const page = useMemo(() => {
    switch (step) {
      case "offer": return <OfferCheckPage {...form} onChange={onChange} onSubmit={() => safe(runOfferParse)} />;
      case "offer_parsed_preview": return <section><OfferParsedPreview values={offerResult?.parsed || {}} /><ul>{(offerResult?.unknown_fields || []).map((u: any) => <li key={u.code}>{u.title}: {u.plain_explanation}</li>)}</ul><ul>{(offerResult?.risk_flags || []).map((r: any) => <li key={r.code}>{r.title}</li>)}</ul><button onClick={() => setStep("offer_assumptions")}>Перейти к допущениям</button></section>;
      case "offer_assumptions": return <OfferAssumptionsStep unknowns={offerResult?.unknown_fields || []} onBack={() => setStep("offer_parsed_preview")} onContinue={() => { setAssumptionsConfirmed(true); setStep("offer_result"); }} />;
      case "offer_result": return <OfferResultPage offerResult={offerResult} scenarioResult={scenarioResult} onRunScenario={() => safe(runScenario)} onGenerateReport={() => safe(runReport)} />;
      case "scenario_mode": return <ScenarioModePage onSingleScenario={() => setStep("scenario_input")} onCompare={() => setStep("scenario_input")} onPortfolio={() => safe(runPortfolio)} onExplain={() => safe(() => runExplain("офз"))} />;
      case "scenario_input": return <ScenarioInputPage form={form} onChange={onChange} onNext={() => setStep("scenario_preview")} />;
      case "scenario_preview": return <ScenarioPreviewPage form={form} unknownFields={offerResult?.unknown_fields || []} assumptions={assumptions} confirmed={assumptionsConfirmed} onConfirmedChange={setAssumptionsConfirmed} canRun={assumptionsConfirmed} onRun={() => safe(runScenario)} />;
      case "scenario_result": return <section><ResultsPage scenarioResult={scenarioResult} /><button onClick={() => safe(runReport)}>Сформировать отчёт</button></section>;
      case "portfolio": return <PortfolioCheckPage result={portfolioResult} />;
      case "explain": return <ExplainInstrumentPage card={knowledgeResult} />;
      case "report": return <ReportPage report={reportResult} />;
      default: return <LandingPage onStartOffer={() => setStep("offer")} onStartScenario={() => setStep("scenario_mode")} />;
    }
  }, [step, form, offerResult, scenarioResult, portfolioResult, knowledgeResult, reportResult, assumptionsConfirmed]);

  return <main><nav><button onClick={() => setStep("landing")}>Главная</button><button onClick={() => setStep("offer")}>Проверить предложение</button><button onClick={() => setStep("scenario_mode")}>Собрать сценарий</button><button onClick={() => safe(runPortfolio)}>Проверить портфель</button><button onClick={() => safe(() => runExplain("офз"))}>База знаний</button><button onClick={() => safe(runReport)}>Отчёт</button></nav>{error ? <section><strong>{error}</strong></section> : null}{page}</main>;
}
