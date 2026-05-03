import pandas as pd

from price_optimizer import analyze_price_optimization, build_price_candidate_grid


def _runner_factory(conf=0.8, clipped=False, ood=False, support='high', fail_below=None):
    def runner(_bundle, manual_price, **kwargs):
        price = float(manual_price)
        if fail_below is not None and price < fail_below:
            raise RuntimeError('forced failure')
        demand = max(1.0, 1000.0 - (price - 100.0) * 5.0)
        revenue = demand * price
        profit = demand * (price - 60.0)
        return {
            "demand_total": demand,
            "revenue_total": revenue,
            "profit_total": profit,
            "profit_total_adjusted": profit,
            "confidence": conf,
            "confidence_label": "Высокая" if conf >= 0.45 else "Низкая",
            "support_label": support,
            "price_clipped": clipped,
            "ood_flag": ood,
            "validation_gate": {"ok": True},
            "effective_scenario": {"applied_price_gross": price},
        }
    return runner


def test_price_grid_contains_current_price():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110]})}
    grid, meta = build_price_candidate_grid(bundle, current_price=100.0, candidate_count=11)
    assert any(abs(float(x) - 100.0) < 1e-9 for x in grid)
    assert meta["safe_price_min"] > 0


def test_optimizer_does_not_mutate_bundle():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110]})}
    before = bundle["daily_base"].copy(deep=True)
    analyze_price_optimization(bundle, 100.0, _runner_factory(), 30, "enhanced_local_factors", "safe_clip")
    pd.testing.assert_frame_equal(bundle["daily_base"], before)


def test_optimizer_blocks_clipped_prices():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(clipped=True), 30, "enhanced_local_factors", "safe_clip")
    assert opt["status"] in {"risky_only", "insufficient_support", "no_valid_candidates"}


def test_optimizer_blocks_ood_prices():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(ood=True), 30, "enhanced_local_factors", "safe_clip")
    assert opt["status"] in {"risky_only", "insufficient_support", "no_valid_candidates"}


def test_low_support_label_is_not_recommended():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(support='low'), 30, "enhanced_local_factors", "safe_clip")
    assert opt["status"] in {"risky_only", "insufficient_support", "no_valid_candidates"}


def test_optimizer_recommends_increase_when_profit_gain_is_material():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(), 30, "enhanced_local_factors", "safe_clip", min_profit_uplift_pct=1.0)
    assert opt["recommended_price"] >= 100.0


def test_optimizer_returns_price_range():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(), 30, "enhanced_local_factors", "safe_clip")
    assert opt["recommended_price_min"] <= opt["recommended_price_max"]


def test_optimizer_keeps_candidates_when_some_runner_calls_fail():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(fail_below=98), 30, "enhanced_local_factors", "safe_clip")
    assert len(opt["candidates"]) > 0


def test_low_historical_price_support_has_no_actionable_recommendation():
    bundle = {"daily_base": pd.DataFrame({"price": [100, 100, 101, 101]})}
    opt = analyze_price_optimization(bundle, 100.0, _runner_factory(), 30, "enhanced_local_factors", "safe_clip")
    assert opt["status"] == "insufficient_support"


def test_current_price_ok_not_actionable_status():
    bundle = {"daily_base": pd.DataFrame({"price": [90, 95, 100, 105, 110, 115, 120]})}

    def flat_runner(_bundle, manual_price, **kwargs):
        price = float(manual_price)
        return {
            "demand_total": 100.0,
            "revenue_total": 10000.0,
            "profit_total": 1000.0,
            "profit_total_adjusted": 1000.0,
            "confidence": 0.9,
            "confidence_label": "Высокая",
            "support_label": "high",
            "price_clipped": False,
            "ood_flag": False,
            "validation_gate": {"ok": True},
            "effective_scenario": {"applied_price_gross": price},
        }

    opt = analyze_price_optimization(bundle, 100.0, flat_runner, 30, "enhanced_local_factors", "safe_clip")
    assert opt["status"] == "current_price_ok"
