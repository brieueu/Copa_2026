from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_updated_vencedor_notebook_outputs_exist_and_are_valid():
    expected_files = [
        ROOT / "Vencedor_Copa_2026_Notebook.ipynb",
        ROOT / "outputs/updated_2026_probabilities.csv",
        ROOT / "outputs/updated_round_of_32_bracket.csv",
        ROOT / "outputs/updated_2026_simulation_summary.md",
        ROOT / "outputs/updated_model_card.md",
        ROOT / "outputs/updated_champion_probabilities.png",
        ROOT / "outputs/updated_phase_progression_heatmap.png",
        ROOT / "outputs/round_of_32_bracket_probabilities.png",
        ROOT / "outputs/monte_carlo_champion_distribution.png",
    ]
    for path in expected_files:
        assert path.exists(), f"Missing artifact: {path}"
        assert path.stat().st_size > 0, f"Empty artifact: {path}"

    probs = pd.read_csv(ROOT / "outputs/updated_2026_probabilities.csv")
    assert probs["team"].nunique() == 48
    assert abs(probs["champion_probability"].sum() - 1.0) < 1e-6
    assert probs["champion_probability"].between(0, 1).all()

    required_phase_columns = {
        "group_stage_probability",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    }
    assert required_phase_columns.issubset(probs.columns)

    bracket = pd.read_csv(ROOT / "outputs/updated_round_of_32_bracket.csv")
    assert len(bracket[bracket["stage"] == "Round of 32"]) == 16
    assert {"team_home", "team_away", "p_home_win", "p_away_win", "most_likely_winner"}.issubset(bracket.columns)
