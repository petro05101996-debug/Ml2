import React from "react";
export default function PortfolioCheckPage({ result }: { result: any }) {
  return <section><h1>Проверка портфеля</h1><h3>Состав портфеля</h3><ul>{(result?.composition||[]).map((x:any,i:number)=><li key={i}>{x.name}: {x.amount} ({x.share_pct}%)</li>)}</ul><p>Доля крупнейшей позиции: {result?.largest_position_share ?? '—'}</p><p>Концентрация: {result?.concentration?.status ?? '—'}</p><p>Ликвидность: {result?.liquidity?.score ?? '—'}</p><p>Стоимость в стресс-сценарии: {result?.stress_result?.stress_value ?? '—'}</p><h3>Checklist</h3><ul>{(result?.checklist||[]).map((x:string)=><li key={x}>{x}</li>)}</ul></section>;
}
