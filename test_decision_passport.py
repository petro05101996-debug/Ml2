import json
from decision_passport import build_decision_passport, build_validation_plan


def test_validation_plan_by_objective():
    plan=build_validation_plan({},{"risk_level":"medium","label":"medium"},"profit")
    assert plan["success_metric"] == "gross_profit"
    assert "control_recommendation" in plan


def test_passport_json_serializable_and_causal_note():
    opt={"best_action":{"candidate_id":"c","title":"Raise price","action_type":"price_change","current_value":100,"target_value":105,"change_pct":5,"expected_effect":{"profit_delta_pct":5,"conservative_profit_delta_pct":3},"reliability":{"score":75,"label":"medium","risk_level":"medium","decision_status":"test_recommended","statistical_support":"moderate"},"warnings":[],"blockers":[]},"safe_option":None,"balanced_option":None,"aggressive_option":None}
    p=build_decision_passport("find_best_decision", opt)
    assert p["decision_status"] == "test_recommended"
    assert any("модельной оценкой" in x for x in p["limitations"])
    json.dumps(p, ensure_ascii=False)
