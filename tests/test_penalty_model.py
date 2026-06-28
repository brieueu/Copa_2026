from tools.penalty_model import knockout_resolution_probability, penalty_home_probability, draw_after_extra_time_probability

def test_penalty_probability_is_separate_from_regular_probability():
    probs = knockout_resolution_probability(0.60, 0.55, 0.50, 0.65)
    assert {'p_draw_after_extra_time', 'p_penalty_home', 'p_penalty_away'}.issubset(probs)
    assert probs['p_penalty_home'] != probs['p_home_regular_extra']
    assert abs(probs['p_penalty_home'] + probs['p_penalty_away'] - 1.0) < 1e-12

def test_draw_after_extra_time_is_clipped():
    assert 0.10 <= draw_after_extra_time_probability(10) <= 0.28
    assert penalty_home_probability(0.8, 0.2) > 0.5
