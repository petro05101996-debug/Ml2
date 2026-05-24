import React from "react";

export default function OfferAssumptionsStep({ unknowns, onBack, onContinue }: { unknowns: any[]; onBack: () => void; onContinue: () => void }) {
  return (
    <section>
      <h2>Подтверждение допущений</h2>
      {unknowns.map((u, i) => (
        <div key={i} className="card">
          <h4>{u.title || u.code}</h4>
          <p>{u.plain_explanation}</p>
          <p><strong>Почему важно:</strong> {u.why_it_matters}</p>
          <ul>{(u.what_to_check || []).map((x: string) => <li key={x}>{x}</li>)}</ul>
          <p><strong>Допущение:</strong> временно используется консервативный параметр до уточнения.</p>
          <p><strong>Влияние:</strong> итоговый результат может измениться после уточнения условий.</p>
        </div>
      ))}
      <button onClick={onBack}>Изменить данные</button>
      <button onClick={onContinue}>Продолжить с допущениями</button>
    </section>
  );
}
