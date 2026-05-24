import React from "react";

export default function ScenarioModePage({ onSingleScenario, onCompare, onPortfolio, onExplain }: { onSingleScenario: () => void; onCompare: () => void; onPortfolio: () => void; onExplain: () => void; }) {
  return <section><h1>Выберите режим проверки</h1><button onClick={onSingleScenario}>Проверить один сценарий</button><button onClick={onCompare}>Сравнить несколько сценариев</button><button onClick={onPortfolio}>Проверить портфель</button><button onClick={onExplain}>Разобраться в инструменте</button></section>;
}
