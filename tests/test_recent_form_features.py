from tools.build_recent_form_features import build_recent_form_features

def test_recent_form_has_required_columns():
    df = build_recent_form_features()
    assert len(df) == 48
    required = {'last_10_points_per_match','last_10_goal_difference_per_match','last_10_opponent_elo_mean','last_10_win_rate','six_month_points_per_match','six_month_goal_difference_per_match','six_month_match_count'}
    assert required.issubset(df.columns)
    assert df['last_10_points_per_match'].notna().all()
