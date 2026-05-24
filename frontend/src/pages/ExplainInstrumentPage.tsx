import React from "react";

export default function ExplainInstrumentPage({ card }: { card: any }) {
  return (
    <section>
      <h1>Разбор инструмента</h1>
      <h2>{card?.title}</h2>
      <p>{card?.simple_explanation}</p>
      <h3>Как работает</h3><ul>{(card?.how_it_works || []).map((x: string) => <li key={x}>{x}</li>)}</ul>
      <h3>Риски</h3><ul>{(card?.key_risks || []).map((x: string) => <li key={x}>{x}</li>)}</ul>
      <h3>Что проверить</h3><ul>{(card?.what_to_check || []).map((x: string) => <li key={x}>{x}</li>)}</ul>
      <h3>Частые ошибки</h3><ul>{(card?.common_misunderstandings || []).map((x: string) => <li key={x}>{x}</li>)}</ul>
      <p>{card?.not_advice_disclaimer}</p>
    </section>
  );
}
