import React from "react";

type Props = { values: Record<string, string | number | null | undefined> };

const fallback = "Не указано в предложении";

export default function OfferParsedPreview({ values }: Props) {
  const row = (label: string, key: string) => (
    <li>
      <strong>{label}:</strong> {values[key] ?? fallback}
    </li>
  );

  return (
    <section>
      <h2>Что удалось определить</h2>
      <ul>
        {row("Тип инструмента", "instrument_type")}
        {row("Заявленная доходность", "declared_return")}
        {row("Срок", "term_months")}
        {row("Комиссии", "fees")}
        {row("Досрочный выход", "early_exit_conditions")}
        {row("Налоги", "tax_notes")}
        {row("Источник", "source")}
        {row("Маркетинговые формулировки", "marketing_phrases")}
      </ul>
    </section>
  );
}
