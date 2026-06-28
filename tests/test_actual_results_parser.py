from pathlib import Path
from tools.build_actual_2026_results import load_actual_results

def test_actual_results_contains_group_stage_scores():
    df = load_actual_results(Path('Data/raw/external/v0.2.0_dynamic_elo/openfootball_worldcup_2026.json'))
    assert {'home_team', 'away_team', 'home_score', 'away_score', 'round'}.issubset(df.columns)
    assert df[df['stage_type'].eq('group')]['home_score'].notna().any()
    assert {'host_country', 'neutral_for_model', 'city'}.issubset(df.columns)
