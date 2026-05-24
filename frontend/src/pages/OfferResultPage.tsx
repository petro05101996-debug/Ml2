import React from "react";
import MetricCard from "../components/ui/MetricCard";

export default function OfferResultPage({ offerResult, scenarioResult, onRunScenario, onGenerateReport }: any) {
  const unknownCount = offerResult?.unknown_fields?.length || 0;
  const riskCount = offerResult?.risk_flags?.length || 0;
  return <section><h1>Результат проверки предложения</h1><p>Сервис обнаружил: нераскрытые условия — {unknownCount}, риск-флаги — {riskCount}. Это не означает, что предложение подходит или не подходит пользователю.</p><h2>Что удалось определить</h2><ul>{Object.entries(offerResult?.parsed||{}).map(([k,v])=><li key={k}>{k}: {String(v)}</li>)}</ul><h2>Что неизвестно</h2><ul>{(offerResult?.unknown_fields||[]).map((u:any)=><li key={u.code}><b>{u.title}</b> — {u.plain_explanation}. {u.why_it_matters}</li>)}</ul><h2>Risk flags</h2><ul>{(offerResult?.risk_flags||[]).map((r:any)=><li key={r.code}>{r.title} ({r.severity}): {r.why_it_matters}</li>)}</ul>{scenarioResult && <><h2>Сценарные метрики</h2><MetricCard label="Чистый номинальный результат" value={String(scenarioResult?.base_result?.net_nominal ?? "—")} /><MetricCard label="Результат с учётом инфляции" value={String(scenarioResult?.base_result?.net_real ?? "—")} /><MetricCard label="Стресс-просадка" value={String(scenarioResult?.stress_result?.drawdown ?? "—")} /></>}<button onClick={onRunScenario}>Рассчитать сценарий</button><button onClick={onGenerateReport}>Сформировать отчёт</button></section>;
}
