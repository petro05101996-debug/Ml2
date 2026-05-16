"""Product copy and navigation labels for the Decision Workspace UI."""

PAGE_OVERVIEW = "Обзор"
PAGE_DECISION = "Проверить решение"
PAGE_WHAT_IF = "What-if"
PAGE_PRICE = "Подбор цены"
PAGE_COMPARE = "Сравнение"
PAGE_REPORT = "Отчёт"
PAGE_DIAGNOSTICS = "Диагностика"

PRODUCT_TITLE = "What-if Cloud"
PRODUCT_PROMISE = "Проверьте бизнес-решение до запуска"
PRODUCT_SUBTITLE = (
    "Загрузите историю продаж, измените цену, скидку, промо или внешний спрос — "
    "система покажет влияние на спрос, выручку, прибыль и риск."
)

PRIMARY_CTA = "Проверить решение"
SECONDARY_CTA_WHAT_IF = "Быстрый what-if"

WORKSPACE_PAGES = [
    PAGE_OVERVIEW,
    PAGE_DECISION,
    PAGE_WHAT_IF,
    PAGE_PRICE,
    PAGE_COMPARE,
    PAGE_REPORT,
    PAGE_DIAGNOSTICS,
]

TAB_ALIASES = {
    "Главное": PAGE_OVERVIEW,
    "Дашборд": PAGE_OVERVIEW,
    "Итог": PAGE_OVERVIEW,
    PAGE_OVERVIEW: PAGE_OVERVIEW,
    "Проверить сценарий": PAGE_WHAT_IF,
    "Сценарий": PAGE_WHAT_IF,
    "What-if сценарий": PAGE_WHAT_IF,
    PAGE_WHAT_IF: PAGE_WHAT_IF,
    "Цена": PAGE_PRICE,
    "Лучший вариант цены": PAGE_PRICE,
    PAGE_PRICE: PAGE_PRICE,
    "Проверка решения": PAGE_DECISION,
    "Проверка решений": PAGE_DECISION,
    "Аналитик решений": PAGE_DECISION,
    "Анализ решений": PAGE_DECISION,
    PAGE_DECISION: PAGE_DECISION,
    "Сравнить варианты": PAGE_COMPARE,
    PAGE_COMPARE: PAGE_COMPARE,
    "Экспорт": PAGE_REPORT,
    "Скачать отчёт": PAGE_REPORT,
    "Отчет": PAGE_REPORT,
    PAGE_REPORT: PAGE_REPORT,
    PAGE_DIAGNOSTICS: PAGE_DIAGNOSTICS,
}


def normalize_workspace_page(page: object, default: str = PAGE_OVERVIEW) -> str:
    """Map legacy session-state tab labels to the current workspace page label."""
    return TAB_ALIASES.get(str(page), default)
