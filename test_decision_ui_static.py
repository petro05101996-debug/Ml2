from pathlib import Path


def test_missing_trained_bundle_shows_ui_warning_not_keyerror():
    src = Path("app.py").read_text()
    assert 'not r.get("_trained_bundle")' in src
    assert "Сначала выполните базовый анализ" in src


def test_audit_ui_uses_action_specific_inputs():
    src = Path("app.py").read_text()
    assert "Новая скидка, %" in src
    assert "Ожидаемое внешнее изменение спроса, %" in src
    assert "Есть внешнее подтверждение гипотезы" in src


def test_new_workspace_pages_exist():
    text = Path("app.py").read_text(encoding="utf-8") + Path("ui/copy.py").read_text(encoding="utf-8")
    for label in ["Обзор", "Проверить решение", "What-if", "Подбор цены", "Сравнение", "Отчёт", "Диагностика"]:
        assert label in text


def test_primary_cta_is_decision_not_scenario():
    text = Path("ui/components.py").read_text(encoding="utf-8")
    assert 'primary_label = "Проверить решение"' in text
    assert 'primary_label = "Проверить сценарий"' not in text


def test_decorative_chips_removed():
    all_ui = (
        Path("app.py").read_text(encoding="utf-8")
        + Path("ui/theme.py").read_text(encoding="utf-8")
        + Path("ui/components.py").read_text(encoding="utf-8")
    )
    for forbidden in ["landing-chip", "landing-chip-row", "floating-chip", "cta-float", "scenario-divider", "scenario-pill"]:
        assert forbidden not in all_ui


def test_decision_before_what_if_in_navigation():
    text = Path("ui/copy.py").read_text(encoding="utf-8")
    assert text.find("Проверить решение") < text.find("What-if")


def test_decision_analyzer_has_no_visible_decision_layer_wording():
    src = Path("app.py").read_text()
    assert "Decision layer: проверяет варианты сценариев" not in src
    assert "Покажет, стоит ли запускать изменение" in src
    assert "Анализатор не меняет модель" in src


def test_decision_analyzer_uses_human_metric_labels():
    src = Path("app.py").read_text()
    assert "render_verdict_panel" in src
    assert "Изменение спроса" in src or "Спрос" in src
    assert "Изменение выручки" in src or "Выручка" in src
    assert "Осторожная оценка прибыли" in src or "Прибыль" in src


def test_decision_json_is_hidden_in_technical_expander():
    src = Path("app.py").read_text()
    assert "Для аналитика" in src
    assert "decision_passport" in src


def test_scenario_screen_has_no_price_sensitivity_artifact():
    src = Path("app.py").read_text()
    assert 'open_surface("Проверка соседних цен")' not in src
    assert "Дополнительно: чувствительность к цене" not in src


def test_report_json_is_hidden_for_analyst():
    src = Path("app.py").read_text()
    assert 'open_surface("JSON для аналитика"' not in src
    assert 'with st.expander("Для аналитика: технический файл"' in src


def test_scenario_advanced_settings_are_not_main_user_path():
    src = Path("app.py").read_text()
    assert 'with st.expander("Дополнительно: логистика"' in src
    assert 'with st.expander("Для аналитика: режим расчёта и защитные настройки"' in src
    assert 'with st.expander("Расширенные настройки"' not in src
    assert 'with st.expander("Для аналитика: расширенные настройки"' not in src


def test_decision_screen_uses_mode_cards_not_tabs():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "render_decision_mode_cards" in src
    assert "st.tabs" not in src


def test_landing_uses_decision_workspace_copy():
    src = Path("ui/components.py").read_text(encoding="utf-8")
    assert "Проверьте бизнес-решение до запуска" in src
    assert "Запускать через пилот" in src
    assert "Проверьте цену до запуска акции" not in src


def test_price_screen_is_honest_about_profit_objective():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "Цель подбора: найти вариант цены с лучшей ожидаемой прибылью" in src
    assert "цена-кандидат" not in src
    assert "Что оптимизируем?" not in src
    assert "Подобрать цену" in src
    assert "Проверить как решение" in src


def test_no_dead_navigation_or_view_model_modules():
    assert not Path("ui/navigation.py").exists()
    assert not Path("ui/view_models.py").exists()


def test_decision_margin_objective_not_faked_as_profit():
    src = Path("app.py").read_text(encoding="utf-8")
    assert '"Маржа": "profit"' not in src
    assert '"Прибыль": "profit"' in src


def test_overview_uses_mode_and_human_quality_labels():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "scenario_mode_label(mode)" in src
    assert "data_quality_ui_label" in src
    assert 'o3.metric("Метод", "Базовый прогноз")' not in src


def test_upload_wizard_requires_explicit_column_confirmation():
    src = Path("app.py").read_text(encoding="utf-8")
    upload_fn = src[src.index("def render_upload_screen"):src.index("def build_ui_decision_summary")]
    assert "2 if columns_done and sku_done else 1" not in upload_fn
    assert "upload_columns_confirmed" in upload_fn
    assert "active_step = 0 if not upload_done else (1 if not columns_confirmed else 2)" in upload_fn
    assert "Подтвердить колонки и продолжить" in upload_fn
    assert "Изменить сопоставление колонок" in upload_fn
    assert "on_change=_reset_upload_columns_confirmation" in upload_fn
    assert upload_fn.index("render_page_header") < upload_fn.index("st.file_uploader")


def test_upload_gate_blockers_are_shown():
    src = Path("app.py").read_text(encoding="utf-8")
    assert 'data_gate["status"] in {"blocked", "diagnostic_only"}' in src
    assert "schema_errors.extend([str(x) for x in blockers])" in src
    assert 'data_gate.get("warnings", [])' in src


def test_find_best_does_not_offer_external_demand_as_control():
    src = Path("app.py").read_text(encoding="utf-8")
    action_block = src[src.index("action_map = {"):src.index("reverse_action_map = {")]
    reverse_block = src[src.index("reverse_action_map = {"):src.index("def _show_decision_passport")]
    assert '"Внешний спрос": "demand_shock"' not in action_block
    assert '"Внешний спрос": "demand_shock"' in reverse_block


def test_landing_has_no_dead_example_button():
    src = Path("ui/components.py").read_text(encoding="utf-8")
    assert "Посмотреть пример результата" not in src


def test_landing_nav_start_is_handled_explicitly():
    components = Path("ui/components.py").read_text(encoding="utf-8")
    app_src = Path("app.py").read_text(encoding="utf-8")
    assert "def render_landing_nav() -> str | None" in components
    assert 'return "app"' in components
    assert "nav_action = render_landing_nav()" in app_src
    assert 'nav_action == "app"' in app_src


def test_workspace_defaults_to_canonical_overview_page():
    app_src = Path("app.py").read_text(encoding="utf-8")
    components = Path("ui/components.py").read_text(encoding="utf-8")
    assert "st.session_state.active_workspace_tab = PAGE_OVERVIEW" in app_src
    assert "active_tab == PAGE_WHAT_IF" in components
    assert 'active_tab == "What-if"' not in components


def test_workspace_navigation_uses_product_buttons_not_radio_pills():
    components = Path("ui/components.py").read_text(encoding="utf-8")
    theme = Path("ui/theme.py").read_text(encoding="utf-8")
    render_tabs = components[components.index("def render_tabs"):components.index("def render_chart_card")]
    assert "st.radio" not in render_tabs
    assert "st.button(tab" in render_tabs
    assert "app-nav" in render_tabs
    assert ".app-nav" in theme


def test_overview_states_main_product_job_and_decision_first_paths():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "Главная задача системы" in src
    assert "проверить бизнес-решение до запуска" in src.lower()
    assert "1. Найти лучшее решение" in src
    assert "2. Проверить мою идею" in src
    assert "3. Быстрый what-if" in src
    assert "Главное действие — «Проверить решение»" in src


def test_what_if_mode_is_display_only_after_analysis():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "Текущий режим расчёта" in src
    assert "Режим выбран перед запуском анализа и в этом разделе не меняется." in src
    assert "Оставьте без изменений, если не уверены. Эти настройки нужны для продвинутых сценариев." not in src


def test_landing_outputs_has_single_heading_and_old_app_landing_removed():
    components = Path("ui/components.py").read_text(encoding="utf-8")
    app_src = Path("app.py").read_text(encoding="utf-8")
    assert components.count("Что вы получите после расчёта") == 1
    assert "def render_landing_mockup" not in app_src
    assert "def inject_landing_css" not in app_src


def test_price_chart_uses_calculated_variant_not_candidate_label():
    src = Path("app.py").read_text(encoding="utf-8")
    copy = Path("ui/copy.py").read_text(encoding="utf-8")
    assert 'recommended_label = "Рекомендация" if status in ACTIONABLE_PRICE_OPT_STATUSES else "Расчётный вариант"' in src
    assert "Кандидат" not in src
    assert "Цена-кандидат" not in copy
