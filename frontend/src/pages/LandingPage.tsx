import React from "react";

export default function LandingPage({ onStartOffer, onStartScenario }: { onStartOffer: () => void; onStartScenario: () => void }) {
  return (
    <main>
      <h1>Проверьте финансовое предложение до вложения денег</h1>
      <p>
        Сервис помогает разобрать условия, комиссии, налоги, инфляцию, досрочный выход, стресс-сценарий и нераскрытые параметры.
        Без подбора инструментов, без продажи продуктов и без инвестиционных рекомендаций.
      </p>
      <button onClick={onStartOffer}>Проверить предложение</button>
      <button onClick={onStartScenario}>Собрать сценарий вручную</button>
    </main>
  );
}
