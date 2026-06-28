import json
from pathlib import Path
import pandas as pd

def test_model_weights_sum_to_one():
    weights = json.loads(Path('Data/processed/model_weights_v0.2.0.json').read_text())['weights']
    assert abs(sum(weights.values()) - 1.0) < 1e-12

def test_dynamic_elo_exists_for_all_teams():
    df = pd.read_csv('Data/processed/dynamic_elo_after_group_stage.csv')
    assert df['dynamic_elo_rating'].notna().all()
    assert df['team'].nunique() >= 48
