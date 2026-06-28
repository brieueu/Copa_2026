import pandas as pd


def test_round_of_32_uses_real_openfootball_bracket_after_june_28():
    bracket = pd.read_csv('outputs/updated_round_of_32_bracket.csv')
    round32 = bracket[bracket['stage'].eq('Round of 32')].sort_values('match_id')
    assert len(round32) == 16
    first = round32.iloc[0]
    assert first['match_id'] == 73
    assert first['team_home'] == 'South Africa'
    assert first['team_away'] == 'Canada'
    assert round32['bracket_source'].eq('openfootball_real_bracket_2026-06-28').all()


def test_real_round_of_32_teams_have_probability_one():
    bracket = pd.read_csv('outputs/updated_round_of_32_bracket.csv')
    probs = pd.read_csv('outputs/updated_2026_probabilities.csv')
    round32_teams = set(bracket.loc[bracket['stage'].eq('Round of 32'), 'team_home']) | set(bracket.loc[bracket['stage'].eq('Round of 32'), 'team_away'])
    round32_probs = probs[probs['team'].isin(round32_teams)]
    assert len(round32_probs) == 32
    assert round32_probs['round_of_32_probability'].eq(1.0).all()
    assert probs.loc[~probs['team'].isin(round32_teams), 'round_of_32_probability'].eq(0.0).all()
