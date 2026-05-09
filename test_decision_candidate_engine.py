import copy
import numpy as np
import pandas as pd

from decision_candidate_engine import generate_decision_candidates


def tb():
    return {"base_ctx":{"price":100.0,"discount":0.1,"promotion":0.0,"cost":65}, "daily_base": pd.DataFrame({"price":np.linspace(90,110,30),"sales":10,"discount":0.1,"promotion":[0,1]*15})}


def test_baseline_candidate_always_created():
    cs=generate_decision_candidates(tb(),None)
    assert any(c["action_type"]=="baseline" for c in cs)


def test_price_candidates_finite_positive():
    cs=[c for c in generate_decision_candidates(tb(),None,allowed_actions=["price_change"]) if c["action_type"]=="price_change"]
    assert cs and all(c["target_value"]>0 for c in cs)


def test_discount_candidates_clamped():
    b=tb(); b["base_ctx"]["discount"]=0.9
    cs=[c for c in generate_decision_candidates(b,None,allowed_actions=["discount_change"]) if c["action_type"]=="discount_change"]
    assert cs and all(0 <= c["target_value"] <= 0.95 for c in cs)


def test_promotion_candidates_only_when_data_exists():
    b=tb(); cs=generate_decision_candidates(b,None,allowed_actions=["promotion_change"])
    assert any(c["action_type"]=="promotion_change" for c in cs)
    b2={"base_ctx":{"price":100,"discount":0.1}, "daily_base": pd.DataFrame({"price":[100],"sales":[1]})}
    cs2=generate_decision_candidates(b2,None,allowed_actions=["promotion_change"])
    assert not any(c["action_type"]=="promotion_change" for c in cs2)


def test_no_mutation_of_trained_bundle():
    b=tb(); before=copy.deepcopy(b)
    generate_decision_candidates(b,None)
    assert b["base_ctx"] == before["base_ctx"]
    assert b["daily_base"].equals(before["daily_base"])

from decision_candidate_engine import evaluate_decision_candidates


def simple_runner(trained_bundle, manual_price, **kwargs):
    price = float(manual_price)
    current = float(trained_bundle["base_ctx"]["price"])
    demand = 1000 * (price / current) ** -1.2
    profit = demand * (price - trained_bundle["base_ctx"].get("cost", 65))
    # Mutate local copy to ensure Decision Layer protects original bundle.
    trained_bundle["mutated_by_runner"] = True
    trained_bundle["base_ctx"]["price"] = -999
    return {"demand_total": demand, "revenue_total": demand * price, "profit_total_adjusted": profit, "confidence": 0.85, "validation_gate": {"ok": True}, "effective_scenario": {"price_out_of_range": False}}


def test_zero_discount_still_generates_positive_discount_candidates():
    b=tb(); b["base_ctx"]["discount"] = 0.0; b["daily_base"]["discount"] = [0, .05, .1] * 10
    cs=generate_decision_candidates(b,None,allowed_actions=["discount_change"])
    vals={round(c["target_value"],2) for c in cs if c["action_type"]=="discount_change"}
    assert {0.05, 0.10}.issubset(vals)


def test_price_grid_finds_known_profit_optimum_with_fake_elasticity_runner():
    b=tb()
    b["daily_base"] = pd.DataFrame({
        "price": np.linspace(70, 150, 90),
        "sales": 100 + np.sin(np.arange(90)),
        "discount": 0.1,
        "promotion": np.resize([0, 1], 90),
    })
    before_keys=set(b.keys()); before_ctx=copy.deepcopy(b["base_ctx"]); before_daily=b["daily_base"].copy()
    cs=generate_decision_candidates(b,None,allowed_actions=["price_change"])
    evaluated=evaluate_decision_candidates({"history_daily":b["daily_base"],"quality_report":{"holdout_metrics":{"wape":10}}},b,cs,simple_runner,"enhanced_local_factors","safe_clip",30,"profit")
    best=max(evaluated, key=lambda e: e.get("expected_effect",{}).get("profit_delta_pct",-999))
    # For elasticity -1.2 and cost 65, profit optimum is materially above current price.
    assert best["candidate"]["target_value"] > 115
    assert set(b.keys()) == before_keys
    assert b["base_ctx"] == before_ctx
    assert b["daily_base"].equals(before_daily)


def test_evaluate_decision_candidates_does_not_mutate_trained_bundle():
    b=tb(); before_keys=set(b.keys()); before_ctx=copy.deepcopy(b["base_ctx"]); before_daily=b["daily_base"].copy()
    cs=generate_decision_candidates(b,None,allowed_actions=["price_change"])
    evaluate_decision_candidates({"history_daily":b["daily_base"],"quality_report":{"holdout_metrics":{"wape":10}}},b,cs[:4],simple_runner,"enhanced_local_factors","safe_clip",30,"profit")
    assert set(b.keys()) == before_keys
    assert b["base_ctx"] == before_ctx
    assert b["daily_base"].equals(before_daily)


def test_discount_candidate_has_absolute_delta_pp_from_zero():
    b=tb(); b["base_ctx"]["discount"] = 0.0; b["daily_base"]["discount"] = [0, .05, .1] * 10
    cs=generate_decision_candidates(b,None,allowed_actions=["discount_change"])
    c=next(c for c in cs if c["action_type"] == "discount_change" and round(c["target_value"], 2) == 0.10)
    assert c["change_pct"] == 0.0
    assert round(c["metadata"]["absolute_delta_pp"], 2) == 10.0


def test_failed_candidate_evaluation_has_technical_error_metadata():
    b=tb()
    candidates=generate_decision_candidates(b,None,allowed_actions=["price_change"])[:2]
    def failing_runner(*args, **kwargs):
        raise ValueError("boom")
    out=evaluate_decision_candidates({"history_daily":b["daily_base"]},b,candidates,failing_runner,"enhanced_local_factors","safe_clip",30,"profit")
    assert out
    assert all(e["reliability"].get("technical_error") for e in out)
    assert all(e["reliability"].get("error_type") == "ValueError" for e in out)
