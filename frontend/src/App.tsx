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
import { getJson, postJson } from "./api";

type Step = "landing" | "offer-check" | "parsed-preview" | "assumptions" | "offer-result" | "scenario-mode" | "manual-input" | "scenario-preview" | "scenario-result" | "portfolio" | "knowledge-base" | "report";

const DISCLAIMER = "Сервис носит информационно-аналитический характер. Расчёты основаны на введённых данных и допущениях. Сервис не является индивидуальной инвестиционной рекомендацией, не предлагает купить, продать или удерживать финансовый инструмент и не определяет пригодность инструмента для пользователя.";
const INSTRUMENTS = ["офз", "депозит", "фонд", "корпоративная облигация", "накопительный счёт", "индексный фонд"];

export default function App() {
  const [step, setStep] = useState<Step>("landing");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ offerText: "", offerSource: "банк", amount: "100000", term: "12", annual_return_pct: "12", fees_pct: "0", tax_pct: "13", inflation_pct: "7", stress_drop_pct: "10", includeTaxes: true, includeInflation: true });
  const [assumptionsConfirmed, setAssumptionsConfirmed] = useState(false);
  const [offerResult, setOfferResult] = useState<any>(null);
  const [scenarioResult, setScenarioResult] = useState<any>(null);
  const [portfolioResult, setPortfolioResult] = useState<any>(null);
  const [knowledgeResult, setKnowledgeResult] = useState<any>(null);
  const [reportResult, setReportResult] = useState<any>(null);
  const [instrument, setInstrument] = useState("офз");
  const [positions, setPositions] = useState([{ name: "ОФЗ", amount: 60000 }, { name: "Фонд", amount: 40000 }]);

  const onChange = (patch: any) => setForm((s) => ({ ...s, ...patch }));
  async function runSafe(action: () => Promise<void>) { try { setError(null); setLoading(true); await action(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setLoading(false); } }

  const unknowns = offerResult?.unknown_fields || [];
  const assumptions = unknowns.length ? unknowns.map((u: any) => `${u.title}: ${u.plain_explanation}`) : ["Расчёт основан только на вручную введённых данных.", "Нераскрытые условия отсутствуют или не были переданы."];

  const page = useMemo(() => {
    switch (step) {
      case "offer-check": return <OfferCheckPage {...form} onChange={onChange} onSubmit={() => runSafe(async()=>{const d = await postJson<any>("/api/analyze/proposal", { text: form.offerText, source: form.offerSource }); setOfferResult(d); setAssumptionsConfirmed(false); setStep("parsed-preview");})} />;
      case "parsed-preview": return <section><OfferParsedPreview values={offerResult?.parsed || {}} /><button onClick={() => setStep("assumptions")}>Продолжить</button></section>;
      case "assumptions": return <OfferAssumptionsStep unknowns={unknowns} onBack={() => setStep("parsed-preview")} onContinue={() => { setAssumptionsConfirmed(true); setStep("offer-result"); }} />;
      case "offer-result": return <OfferResultPage offerResult={offerResult} scenarioResult={scenarioResult} onRunScenario={() => setStep("manual-input")} onGenerateReport={() => runSafe(async()=>{const d=await postJson<any>("/api/report/generate", { offer: offerResult||{}, scenario: scenarioResult||{}, portfolio: portfolioResult||{}, explain: knowledgeResult||{} }); setReportResult(d); setStep("report");})} />;
      case "scenario-mode": return <ScenarioModePage onSingleScenario={() => setStep("manual-input")} onCompare={() => setStep("manual-input")} onPortfolio={() => setStep("portfolio")} onExplain={() => setStep("knowledge-base")} />;
      case "manual-input": return <ScenarioInputPage form={form} onChange={onChange} onNext={() => setStep("scenario-preview")} />;
      case "scenario-preview": return <ScenarioPreviewPage form={form} unknownFields={unknowns} assumptions={assumptions} confirmed={assumptionsConfirmed} onConfirmedChange={setAssumptionsConfirmed} canRun={assumptionsConfirmed} onRun={() => runSafe(async()=>{const d = await postJson<any>("/api/analyze/scenario", { amount:Number(form.amount), term_months:Number(form.term), annual_return_pct:Number(form.annual_return_pct), fees_pct:Number(form.fees_pct), tax_pct:Number(form.tax_pct), inflation_pct:Number(form.inflation_pct), stress_drop_pct:Number(form.stress_drop_pct), include_taxes:form.includeTaxes, include_inflation:form.includeInflation, unknown_fields_count:unknowns.length }); setScenarioResult(d); setStep("scenario-result");})} />;
      case "scenario-result": return <section><ResultsPage scenarioResult={scenarioResult} /><div><button onClick={()=>setStep("scenario-preview")}>Изменить параметры</button><button onClick={() => runSafe(async()=>{const d=await postJson<any>("/api/report/generate", { offer: offerResult||{}, scenario: scenarioResult||{}, portfolio: portfolioResult||{}, explain: knowledgeResult||{} }); setReportResult(d); setStep("report");})}>Сформировать отчёт</button></div></section>;
      case "portfolio": return <section><h2>Портфель</h2>{positions.map((p,idx)=><div key={idx}><input value={p.name} onChange={(e)=>setPositions((s)=>s.map((x,i)=>i===idx?{...x,name:e.target.value}:x))}/><input type="number" value={p.amount} onChange={(e)=>setPositions((s)=>s.map((x,i)=>i===idx?{...x,amount:Number(e.target.value)}:x))}/><button onClick={()=>setPositions((s)=>s.filter((_,i)=>i!==idx))}>Удалить</button></div>)}<button onClick={()=>setPositions((s)=>[...s,{name:"",amount:0}])}>Добавить позицию</button><button onClick={()=>runSafe(async()=>{const d=await postJson<any>("/api/analyze/portfolio",{positions});setPortfolioResult(d);})}>Рассчитать портфель</button><PortfolioCheckPage result={portfolioResult} /></section>;
      case "knowledge-base": return <section><h2>База знаний</h2><select value={instrument} onChange={(e)=>setInstrument(e.target.value)}>{INSTRUMENTS.map((x)=><option key={x}>{x}</option>)}</select><button onClick={()=>runSafe(async()=>{const d=await getJson<any>(`/api/knowledge/instrument/${encodeURIComponent(instrument)}`);setKnowledgeResult(d);})}>Показать</button><ExplainInstrumentPage card={knowledgeResult} /></section>;
      case "report": return <ReportPage report={reportResult} />;
      default: return <LandingPage onStartOffer={() => setStep("offer-check")} onStartScenario={() => setStep("scenario-mode")} />;
    }
  }, [step, form, offerResult, scenarioResult, portfolioResult, knowledgeResult, reportResult, assumptionsConfirmed, instrument, positions]);

  return <main><nav><button onClick={() => setStep("landing")}>Главная</button><button onClick={() => setStep("offer-check")}>Проверка предложения</button><button onClick={() => setStep("scenario-mode")}>Сценарий</button><button onClick={() => setStep("portfolio")}>Портфель</button><button onClick={() => setStep("knowledge-base")}>База знаний</button><button onClick={() => setStep("report")}>Отчёт</button></nav><p>{DISCLAIMER}</p>{loading && <p>Загрузка...</p>}{error && <p>{error}</p>}{page}</main>;
}
