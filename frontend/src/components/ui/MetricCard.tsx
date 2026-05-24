import React from "react";
export default function MetricCard({ label, value, hint }: { label: string; value: string; hint?: string }) { return <div className="card metric-card"><div className="muted">{label}</div><div className="metric-value"><strong>{value}</strong></div>{hint ? <div className="muted">{hint}</div> : null}</div>; }
