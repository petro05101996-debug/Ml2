import numpy as np
import pandas as pd

from decision_reliability import evaluate_decision_reliability, estimate_effect_uncertainty


def bundle(days=180, prices=None, include_cost=True):
    prices = prices if prices is not None else np.linspace(90, 110, days)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=days), "sales": 100 + np.sin(np.arange(days))*10, "price": np.resize(prices, days), "discount": np.linspace(0.02,0.08,days), "promotion": np.resize([0,1], days)})
    base = {"price": 100.0, "discount": 0.05, "promotion": 0.0}
    if include_cost: base["cost"] = 65.0
    return {"daily_base": df, "base_ctx": base}, {"history_daily": df, "quality_report": {"holdout_metrics": {"wape": 18}}}


def cand(action="price_change", target=105, current=100, change=5):
    return {"candidate_id":"c", "action_type":action, "current_value":current, "target_value":target, "change_pct":change, "metadata":{}, "scenario_params":{"manual_price":target}}


def scenario(price=105, demand=950, revenue=99750, profit=38000, **kw):
    d={"demand_total":demand,"revenue_total":revenue,"profit_total_adjusted":profit,"confidence":0.8,"validation_gate":{"ok":True},"effective_scenario":{"price_out_of_range":False}}
    d.update(kw); return d

BASE={"demand_total":1000,"revenue_total":100000,"profit_total_adjusted":35000,"confidence":0.8}


def test_high_support_price_inside_range():
    tb,res=bundle()
    out=evaluate_decision_reliability(res,tb,cand(),scenario(),BASE)
    assert out["score"] >= 70
    assert out["decision_status"] in {"recommended","test_recommended"}
    assert out["risk_level"] != "high"


def test_price_outside_historical_range_high_risk():
    tb,res=bundle()
    c=cand(target=150, change=50)
    out=evaluate_decision_reliability(res,tb,c,scenario(price=150, ood_flag=True, effective_scenario={"price_out_of_range":True}),BASE)
    assert out["risk_level"] == "high"
    assert out["decision_status"] in {"experimental_only","not_recommended"}
    assert any("диапаз" in w.lower() or "экстра" in w.lower() for w in out["warnings"])


def test_low_unique_prices_factor_support_low():
    tb,res=bundle(prices=[100, 101])
    out=evaluate_decision_reliability(res,tb,cand(),scenario(),BASE)
    assert out["components"]["factor_support"] < 55
    assert out["decision_status"] in {"experimental_only","not_recommended","test_recommended"}


def test_no_profit_blocks_profit_optimization():
    tb,res=bundle(include_cost=False)
    out=evaluate_decision_reliability(res,tb,cand(),{"demand_total":100,"revenue_total":1000,"confidence":0.8}, {"demand_total":90,"revenue_total":900}, objective="profit")
    assert out["decision_status"] == "not_recommended"
    assert out["blockers"]


def test_demand_shock_max_experimental_without_evidence():
    tb,res=bundle()
    out=evaluate_decision_reliability(res,tb,cand("demand_shock",1.1,1.0,10),scenario(),BASE)
    assert out["decision_status"] == "experimental_only"
    assert any("Demand shock" in w for w in out["warnings"])


def test_price_guardrail_mode_is_passed_to_reliability():
    tb,res=bundle()
    c=cand(target=150, change=50)
    clipped = scenario(price=150, price_clipped=True, effective_scenario={"price_out_of_range": True}, extrapolation_applied=True)
    safe = evaluate_decision_reliability(res,tb,c,clipped,BASE,price_guardrail_mode="safe_clip")
    extrap = evaluate_decision_reliability(res,tb,c,clipped,BASE,price_guardrail_mode="economic_extrapolation")
    assert any("guardrail" in b for b in safe["blockers"])
    assert not any("guardrail" in b for b in extrap["blockers"])


def test_outside_historical_range_is_not_auto_recommended():
    tb,res=bundle()
    out=evaluate_decision_reliability(res,tb,cand(target=145,change=45),scenario(ood_flag=True,effective_scenario={"price_out_of_range":True}),BASE,price_guardrail_mode="economic_extrapolation")
    assert out["decision_status"] in {"experimental_only","not_recommended","test_recommended"}
    assert out["decision_status"] != "recommended"


def test_effect_smaller_than_model_noise_is_test_only_or_not_recommended():
    tb,res=bundle(); res["quality_report"]["holdout_metrics"]["wape"] = 35
    small=scenario(profit=36000)  # +2.86%, below WAPE-derived total-horizon noise
    out=evaluate_decision_reliability(res,tb,cand(),small,BASE)
    assert out["decision_status"] in {"experimental_only","not_recommended","test_recommended"}
    assert out["decision_status"] != "recommended"
    assert out["economic_significance"]["uncertainty"]["expected_model_error_pct"] > abs(out["economic_significance"]["profit_delta_pct"])


def test_flat_history_never_gets_recommended_status():
    tb,res=bundle()
    tb["daily_base"]["sales"] = 100.0
    res["history_daily"] = tb["daily_base"]
    out=evaluate_decision_reliability(res,tb,cand(),scenario(profit=45000),BASE)
    assert out["decision_status"] != "recommended"
    assert any("плос" in w.lower() for w in out["warnings"])


def test_revenue_objective_blocks_large_profit_drop():
    tb,res=bundle()
    bad={"demand_total":1300,"revenue_total":130000,"profit_total_adjusted":30000,"confidence":0.8,"validation_gate":{"ok":True},"effective_scenario":{"price_out_of_range":False}}
    out=evaluate_decision_reliability(res,tb,cand(),bad,BASE,objective="revenue")
    assert out["decision_status"] == "not_recommended"
    assert any("Прибыль падает более чем на 5%" in b for b in out["blockers"])


def test_revenue_objective_profit_decline_not_recommended():
    tb,res=bundle()
    slight={"demand_total":1200,"revenue_total":120000,"profit_total_adjusted":34500,"confidence":0.95,"validation_gate":{"ok":True},"effective_scenario":{"price_out_of_range":False}}
    out=evaluate_decision_reliability(res,tb,cand(),slight,BASE,objective="revenue")
    assert out["decision_status"] != "recommended"
    assert any("Прибыль снижается" in w for w in out["warnings"])


def test_unknown_model_features_do_not_claim_price_is_used_by_ml():
    results = {
        "history_daily": pd.DataFrame({
            "price": [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            "sales": [10, 11, 9, 10, 12, 8, 9, 7, 8, 7],
            "date": pd.date_range("2025-01-01", periods=10),
        }),
        "quality_report": {"holdout_metrics": {"wape": 30}},
    }
    trained_bundle = {"daily_base": results["history_daily"]}
    candidate = {
        "action_type": "price_change",
        "target_value": 110,
        "change_pct": 10,
        "metadata": {},
    }

    rel = evaluate_decision_reliability(
        results,
        trained_bundle,
        candidate,
        {"profit_total": 110, "demand_total": 90, "revenue_total": 1000, "confidence": 0.7},
        {"profit_total": 100, "demand_total": 100, "revenue_total": 900},
        objective="profit",
    )

    details = rel["component_details"]["factor_support"]
    assert details["model_feature_known"] is False
    assert details["model_uses_price"] is False
    assert any("Список признаков модели неизвестен" in x for x in rel["warnings"])


def test_model_quality_wape_is_already_percent_not_rescaled():
    tb, res = bundle()
    res["quality_report"]["holdout_metrics"]["wape"] = 0.67
    out = evaluate_decision_reliability(res, tb, cand(), scenario(), BASE)
    assert out["component_details"]["model_quality"]["wape"] == 0.67
    assert not any("WAPE высокий" in w for w in out["warnings"])


def test_strong_economic_extrapolation_blocks_auto_recommendation():
    tb, res = bundle()
    c = cand(target=150, change=50)
    sc = scenario(
        price=150,
        ood_flag=True,
        extrapolation_applied=True,
        effective_scenario={"price_out_of_range": True},
    )
    sc["price_out_of_range"] = True
    sc["extrapolation_price_ratio"] = 1.36
    out = evaluate_decision_reliability(res, tb, c, sc, BASE, price_guardrail_mode="economic_extrapolation")
    assert out["decision_status"] == "experimental_only"


def test_uncertainty_wape_stays_percent_points_not_fraction_scaled():
    out = estimate_effect_uncertainty({}, {"profit_total": 100.0}, {"profit_total": 103.0}, horizon_days=30, profit_delta_pct=3.0, wape=0.67)
    assert out["method"] == "wape_approximation"
    assert out["expected_model_error_pct"] < 2.5


def test_fallback_elasticity_extrapolation_is_experimental_only():
    tb, res = bundle()
    c = cand(target=125, change=25)
    sc = scenario(price=125, ood_flag=True, extrapolation_applied=True, effective_scenario={"price_out_of_range": True, "elasticity_source": "fallback_prior_invalid_elasticity"})
    sc["price_out_of_range"] = True
    sc["extrapolation_price_ratio"] = 1.12
    sc["elasticity_source"] = "fallback_prior_invalid_elasticity"
    out = evaluate_decision_reliability(res, tb, c, sc, BASE, price_guardrail_mode="economic_extrapolation")
    assert out["decision_status"] == "experimental_only"
    assert out["effect_nature"] == "prior_based"


def test_out_of_safe_range_fallback_elasticity_extrapolation_is_experimental_only():
    tb, res = bundle()
    c = cand(target=125, change=25)
    sc = scenario(price=125, ood_flag=True, extrapolation_applied=True, effective_scenario={"price_out_of_range": True, "elasticity_source": "fallback_prior_out_of_safe_range"})
    sc["price_out_of_range"] = True
    sc["extrapolation_price_ratio"] = 1.12
    sc["elasticity_source"] = "fallback_prior_out_of_safe_range"
    out = evaluate_decision_reliability(res, tb, c, sc, BASE, price_guardrail_mode="economic_extrapolation")
    assert out["decision_status"] == "experimental_only"
    assert out["recommendation_gate"] == "experimental_only"
    assert out["effect_nature"] == "prior_based"
