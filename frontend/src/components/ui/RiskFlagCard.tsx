import React from "react";
export default function RiskFlagCard({ title, severity, why, check }: { title: string; severity: string; why: string; check: string }) { return <div><h4>{title}</h4><p>Уровень: {severity}</p><p>{why}</p><p>{check}</p></div>; }
