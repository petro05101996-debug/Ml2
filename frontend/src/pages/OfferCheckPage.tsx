import React, { useMemo } from "react";

type Props = {
  offerText: string;
  offerSource: string;
  amount: string;
  term: string;
  earlyNeed: string;
  includeTaxes: boolean;
  includeInflation: boolean;
  onChange: (patch: Partial<Props>) => void;
  onSubmit: () => void;
};

export default function OfferCheckPage(props: Props) {
  const errors = useMemo(() => {
    const out: string[] = [];
    if (!props.offerText.trim()) out.push("Добавьте текст предложения.");
    if (!props.offerSource.trim()) out.push("Выберите источник предложения.");
    if (props.amount && Number(props.amount) <= 0) out.push("Сумма должна быть больше 0.");
    if (props.term && Number(props.term) <= 0) out.push("Срок должен быть больше 0.");
    return out;
  }, [props.offerText, props.offerSource, props.amount, props.term]);

  return (
    <section>
      <h1>Проверка финансового предложения</h1>
      <textarea value={props.offerText} onChange={(e) => props.onChange({ offerText: e.target.value })} />
      <select value={props.offerSource} onChange={(e) => props.onChange({ offerSource: e.target.value })}>
        <option>банк</option><option>брокер</option><option>приложение</option><option>реклама</option><option>Telegram/соцсети</option><option>другое</option>
      </select>
      <input placeholder="Сумма" value={props.amount} onChange={(e) => props.onChange({ amount: e.target.value })} />
      <input placeholder="Срок" value={props.term} onChange={(e) => props.onChange({ term: e.target.value })} />
      <select value={props.earlyNeed} onChange={(e) => props.onChange({ earlyNeed: e.target.value })}><option>да</option><option>нет</option><option>не знаю</option></select>
      <label><input type="checkbox" checked={props.includeTaxes} onChange={(e) => props.onChange({ includeTaxes: e.target.checked })} /> учитывать налоги</label>
      <label><input type="checkbox" checked={props.includeInflation} onChange={(e) => props.onChange({ includeInflation: e.target.checked })} /> учитывать инфляцию</label>
      {errors.length > 0 && <ul>{errors.map((e) => <li key={e}>{e}</li>)}</ul>}
      <button disabled={errors.length > 0} onClick={props.onSubmit}>Разобрать предложение</button>
    </section>
  );
}
