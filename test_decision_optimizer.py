from decision_optimizer import rank_decision_candidates


def ev(cid, profit, rel, risk="low", status="test_recommended", blockers=None):
    return {"candidate":{"candidate_id":cid,"title":cid,"action_type":"price_change","target_value":100},"expected_effect":{"profit_delta_pct":profit,"conservative_profit_delta_pct":profit-1},"reliability":{"score":rel,"risk_level":risk,"decision_status":status,"components":{"scenario_support":rel,"validation_readiness":80},"economic_significance":{"conservative_profit_delta_pct":profit-1}},"warnings":[],"blockers":blockers or []}


def test_not_choose_max_profit_if_high_risk():
    out=rank_decision_candidates([ev("risky",40,50,"high","experimental_only"), ev("safe",8,82,"low","recommended")])
    assert out["best_action"]["candidate_id"] == "safe"
    assert out["aggressive_option"]["candidate_id"] == "risky"


def test_choose_balanced_lower_profit_higher_reliability():
    out=rank_decision_candidates([ev("high_profit",10,60,"medium","experimental_only"), ev("balanced",8,85,"low","recommended")])
    assert out["best_action"]["candidate_id"] == "balanced"


def test_aggressive_high_upside_not_best_when_high_risk():
    out=rank_decision_candidates([ev("aggr",25,55,"high","experimental_only"), ev("ok",6,75,"medium","test_recommended")])
    assert out["aggressive_option"]["candidate_id"] == "aggr"
    assert out["best_action"]["candidate_id"] == "ok"


def test_no_valid_candidates_summary():
    out=rank_decision_candidates([ev("bad",-2,40,"high","not_recommended")])
    assert out["best_action"] is None
    assert "Надёжного решения" in out["summary"]["message"]


def test_compact_preserves_objective_for_passport():
    candidate = ev("rev",6,80,"low","recommended")
    candidate["candidate"]["objective"] = "revenue"
    out=rank_decision_candidates([candidate], objective="revenue")
    assert out["best_action"]["objective"] == "revenue"


def test_aggressive_option_filters_blocked_and_technical_errors():
    blocked = ev("blocked",50,80,"low","recommended", blockers=["boom"])
    tech = ev("tech",45,90,"low","recommended")
    tech["reliability"]["technical_error"] = True
    ok = ev("ok_aggressive",10,70,"medium","test_recommended")
    out=rank_decision_candidates([blocked, tech, ok], objective="profit")
    assert out["aggressive_option"]["candidate_id"] == "ok_aggressive"
