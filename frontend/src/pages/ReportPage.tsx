import React from "react";

export default function ReportPage({ report }: { report: any }) {
  return (
    <section>
      <h1>{report?.title || "Отчёт"}</h1>
      <p>{report?.subtitle}</p>
      {(report?.sections || []).map((s: any, i: number) => (
        <div key={i} className="card">
          <h3>{s.title}</h3>
          {s.metrics ? <ul>{s.metrics.map((m: any, idx: number) => <li key={idx}>{m.label}: {String(m.value ?? "—")}</li>)}</ul> : null}
          {s.items ? <ul>{s.items.map((x: string, idx: number) => <li key={idx}>{x}</li>)}</ul> : null}
          {s.risk_flags ? <ul>{s.risk_flags.map((x: any, idx: number) => <li key={idx}>{x.title}</li>)}</ul> : null}
          {s.unknown_fields ? <ul>{s.unknown_fields.map((x: any, idx: number) => <li key={idx}>{x.title}</li>)}</ul> : null}
          {s.checklist ? <ul>{s.checklist.map((x: string, idx: number) => <li key={idx}>{x}</li>)}</ul> : null}
        </div>
      ))}
      <p>{report?.disclaimer_top}</p>
      <p>{report?.disclaimer_bottom}</p>
    </section>
  );
}
