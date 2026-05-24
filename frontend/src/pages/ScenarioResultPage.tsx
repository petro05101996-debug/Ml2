import React from "react";
import DisclaimerBlock from "../components/ui/DisclaimerBlock";

export default function ScenarioResultPage() {
  return (
    <section>
      <h1>Результат сценария</h1>
      <h2>Базовый результат</h2>
      <h2>Стресс-сценарий</h2>
      <h2>Ликвидность</h2>
      <h2>Risk flags</h2>
      <h2>Неизвестные параметры</h2>
      <h2>Чеклист перед самостоятельным решением</h2>
      <DisclaimerBlock />
    </section>
  );
}
