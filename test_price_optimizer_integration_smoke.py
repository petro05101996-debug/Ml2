from pathlib import Path


def test_app_imports_price_optimizer_functions():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'from price_optimizer import analyze_price_optimization, build_price_optimizer_signature' in app_text


def test_stale_signature_guard_present_for_apply_button():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'is_price_optimizer_stale' in app_text
    assert 'disabled=not can_apply' in app_text


def test_scenario_optimizer_uses_what_if_price_reference():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'scenario_reference_price = float(st.session_state.get("what_if_price"' in app_text


def test_base_and_scenario_optimizer_states_are_separated():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'price_optimizer_result_base' in app_text
    assert 'price_optimizer_result_scenario' in app_text


def test_scenario_stale_warning_and_inline_chart_table_present():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'is_scenario_price_optimizer_stale' in app_text
    assert 'Сценарная рекомендация по цене устарела' in app_text
    assert 'render_price_optimizer_chart(st.session_state.get("price_optimizer_result_scenario"))' in app_text
    assert 'render_price_optimizer_table(st.session_state.get("price_optimizer_result_scenario"))' in app_text


def test_non_actionable_chart_label_uses_candidate():
    app_text = Path('app.py').read_text(encoding='utf-8')
    assert 'recommended_label = "Рекомендация" if status in ACTIONABLE_PRICE_OPT_STATUSES else "Кандидат"' in app_text
