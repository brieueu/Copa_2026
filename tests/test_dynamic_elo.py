from tools.dynamic_elo import update_elo, margin_multiplier

def test_three_goal_win_changes_more_than_one_goal_win():
    base = {'Brazil': 1900.0, 'Scotland': 1700.0}
    one_goal = update_elo(base, 'Brazil', 'Scotland', 1, 0, k=60, local_advantage=0)
    three_goals = update_elo(base, 'Brazil', 'Scotland', 3, 0, k=60, local_advantage=0)
    assert three_goals['Brazil'] - base['Brazil'] > one_goal['Brazil'] - base['Brazil']

def test_elo_is_conserved_pairwise_without_rounding():
    base = {'A': 1800.0, 'B': 1800.0}
    updated = update_elo(base, 'A', 'B', 2, 1, k=60, local_advantage=0)
    assert abs((updated['A'] + updated['B']) - 3600.0) < 1e-9

def test_margin_multiplier_world_elo_style():
    assert margin_multiplier(3, 0) > margin_multiplier(1, 0)
