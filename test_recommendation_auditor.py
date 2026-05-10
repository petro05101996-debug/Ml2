import numpy as np
import pandas as pd

from recommendation_auditor import audit_and_improve_recommendation, build_candidate_from_recommendation, generate_alternatives_around_recommendation


def make_bundle(max_price=110):
    prices=np.linspace(90,max_price,180)
    df=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=180),"sales":100+np.sin(np.arange(180))*5,"price":prices,"discount":0.0,"promotion":[0,1]*90})
    return {"base_ctx":{"price":100.0,"discount":0.0,"promotion":0.0,"cost":65.0},"daily_base":df}, {"history_daily":df,"quality_report":{"holdout_metrics":{"wape":18}}}


def fake_runner(trained_bundle, manual_price, freight_multiplier=1.0, demand_multiplier=1.0,
                horizon_days=30, discount_multiplier=1.0, cost_multiplier=1.0,
                stock_cap=0.0, overrides=None, factor_overrides=None,
                scenario_calc_mode=None, price_guardrail_mode="safe_clip"):
    current_price = trained_bundle["base_ctx"]["price"]
    base_demand = 1000
    elasticity = -1.2
    price_ratio = manual_price / current_price
    demand = base_demand * (price_ratio ** elasticity) * demand_multiplier
    discount = overrides.get("discount", trained_bundle["base_ctx"].get("discount", 0.0)) if overrides else trained_bundle["base_ctx"].get("discount", 0.0)
    net_price = manual_price * (1 - discount)
    cost = trained_bundle["base_ctx"].get("cost", current_price * 0.65)
    revenue = demand * net_price
    profit = demand * (net_price - cost)
    maxp = float(trained_bundle["daily_base"]["price"].max())
    ood = manual_price > maxp
    return {"demand_total": demand, "revenue_total": revenue, "profit_total_adjusted": profit, "confidence": 0.8, "confidence_label": "medium", "support_label": "medium", "price_clipped": False, "ood_flag": ood, "validation_gate": {"ok": True}, "effective_scenario": {"requested_price_gross": manual_price, "applied_price_gross": manual_price, "price_out_of_range": ood}}


def test_external_15pct_outside_support_gets_improved_solution():
    tb,res=make_bundle(max_price=108)
    rec={"source_name":"analyst","action_type":"price_change","target_value":115.0,"change_pct":15.0,"objective":"profit","comment":"+15%","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","economic_extrapolation")
    assert out["audit_verdict"]["verdict"] in {"modify","test_only"}
    assert out["improved_solution"] is not None
    if out["improved_solution"].get("target_value"):
        assert out["improved_solution"]["target_value"] <= 115.0


def test_external_unprofitable_rejected():
    tb,res=make_bundle(max_price=110)
    rec={"source_name":"manual","action_type":"discount_change","target_value":0.8,"change_pct":800,"objective":"profit","comment":"huge discount","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","safe_clip")
    assert out["audit_verdict"]["verdict"] == "reject"
    assert out["improved_solution"] is not None


def test_good_external_recommendation_accept():
    tb,res=make_bundle(max_price=110)
    rec={"source_name":"manual","action_type":"price_change","target_value":105.0,"change_pct":5,"objective":"profit","comment":"+5%","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","safe_clip")
    assert out["audit_verdict"]["verdict"] in {"accept","modify"}
    assert out["improved_solution"] is not None


def test_mode2_returns_more_than_evaluation():
    tb,res=make_bundle()
    rec={"source_name":"manual","action_type":"price_change","target_value":105.0,"change_pct":5,"objective":"profit","comment":"+5%","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","safe_clip")
    for key in ["audit_verdict","improved_solution","alternatives_table","decision_passport"]:
        assert key in out


def test_external_bad_recommendation_gets_rejected():
    tb,res=make_bundle(max_price=110)
    rec={"source_name":"manual","action_type":"discount_change","target_value":0.8,"change_pct":800,"objective":"profit","comment":"huge discount","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","safe_clip")
    assert out["audit_verdict"]["verdict"] == "reject"


def test_external_good_recommendation_gets_accepted_or_test_recommended():
    tb,res=make_bundle(max_price=120)
    rec={"source_name":"manual","action_type":"price_change","target_value":108.0,"change_pct":8,"objective":"profit","comment":"+8%","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","safe_clip")
    assert out["audit_verdict"]["verdict"] in {"accept","modify","test_only"}
    assert out["input_evaluation"]["reliability"]["decision_status"] in {"recommended","test_recommended","experimental_only"}


def test_external_overaggressive_recommendation_gets_modified():
    tb,res=make_bundle(max_price=108)
    rec={"source_name":"manual","action_type":"price_change","target_value":130.0,"change_pct":30,"objective":"profit","comment":"+30%","metadata":{}}
    out=audit_and_improve_recommendation(res,tb,rec,fake_runner,"enhanced_local_factors","economic_extrapolation")
    assert out["audit_verdict"]["verdict"] in {"modify","test_only","reject"}
    assert out["improved_solution"] is not None


def test_audit_discount_candidate_has_absolute_delta_pp_from_zero():
    tb,res=make_bundle()
    tb["base_ctx"]["discount"] = 0.0
    rec={"source_name":"manual","action_type":"discount_change","target_value":0.10,"change_pct":0,"objective":"profit","comment":"10 pct discount","metadata":{}}
    c=build_candidate_from_recommendation(rec,tb,None,30)
    assert c["change_pct"] == 0.0
    assert round(c["metadata"]["absolute_delta_pp"], 2) == 10.0
    alts=generate_alternatives_around_recommendation(c,tb,None,"profit",30)
    alt=next(a for a in alts if a["action_type"] == "discount_change" and round(a["target_value"], 2) == 0.10)
    assert round(alt["metadata"]["absolute_delta_pp"], 2) == 10.0


def test_recommendation_candidate_preserves_factor_overrides_from_context():
    tb, _ = make_bundle()
    rec = {
        "source_name": "manual",
        "action_type": "price_change",
        "target_value": 105.0,
        "objective": "profit",
        "comment": "+5%",
        "metadata": {},
    }
    ctx = {"price": 100.0, "discount": 0.0, "promotion": 0.0, "factor_overrides": {"factor__weather": 1.2}}

    cand = build_candidate_from_recommendation(rec, tb, ctx, 30)
    alts = generate_alternatives_around_recommendation(cand, tb, ctx, "profit", 30)

    assert cand["scenario_params"]["factor_overrides"] == {"factor__weather": 1.2}
    assert all(a["scenario_params"].get("factor_overrides") == {"factor__weather": 1.2} for a in alts)


def test_freight_recommendation_uses_absolute_current_context_freight_override():
    tb, _ = make_bundle()
    tb["base_ctx"]["freight_value"] = 10.0
    ctx = {"price": 100.0, "discount": 0.0, "promotion": 0.0, "freight_value": 20.0}
    rec = {
        "source_name": "manual",
        "action_type": "freight_change",
        "target_value": 18.0,
        "objective": "profit",
        "comment": "freight 18",
        "metadata": {},
    }

    cand = build_candidate_from_recommendation(rec, tb, ctx, 30)
    alts = generate_alternatives_around_recommendation(cand, tb, ctx, "profit", 30)
    baseline = next(a for a in alts if a["action_type"] == "baseline")

    assert cand["scenario_params"]["overrides"]["freight_value"] == 18.0
    assert cand["scenario_params"]["freight_multiplier"] == 1.0
    assert baseline["scenario_params"]["overrides"]["freight_value"] == 20.0
    assert baseline["scenario_params"]["freight_multiplier"] == 1.0
