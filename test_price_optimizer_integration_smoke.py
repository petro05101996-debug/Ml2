from pathlib import Path


def test_app_imports_price_optimizer_functions():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'from price_optimizer import analyze_price_optimization, build_price_optimizer_signature' in app_text


def test_stale_signature_guard_present_for_apply_button():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'is_price_optimizer_stale' in app_text
    assert 'disabled=not can_apply' in app_text


def test_scenario_tab_routes_to_price_candidate_section():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'Открыть раздел «Подбор цены»' in app_text
    assert 'go_price_candidate_from_scenario' in app_text
    assert 'st.session_state.active_workspace_tab = PAGE_PRICE' in app_text


def test_price_optimizer_uses_base_state_only_after_ux_split():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'price_optimizer_result_base' in app_text
    assert 'price_optimizer_result_scenario' not in app_text


def test_scenario_tab_has_no_inline_price_optimizer():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'run_price_optimizer_from_scenario_tab' not in app_text
    assert 'scenario_reference_price' not in app_text
    assert 'is_scenario_price_optimizer_stale' not in app_text
    assert 'Сценарная рекомендация по цене устарела' not in app_text
    assert 'render_price_optimizer_chart(st.session_state.get("price_optimizer_result_scenario"))' not in app_text
    assert 'render_price_optimizer_table(st.session_state.get("price_optimizer_result_scenario"))' not in app_text


def test_price_candidate_context_text_is_not_misleading():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'Контекст: рекомендация рассчитана для параметров, указанных в разделе «What-if»' in app_text
    assert 'Контекст: рекомендация рассчитана для базовых условий.' not in app_text


def test_non_actionable_chart_label_uses_calculated_variant():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'recommended_label = "Рекомендация" if status in ACTIONABLE_PRICE_OPT_STATUSES else "Расчётный вариант"' in app_text
