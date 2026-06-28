from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

try:
    from team_name_normalization import canonical_team_name
except ImportError:
    from tools.team_name_normalization import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]
SHOOTOUTS = ROOT / "Data/raw/external/v0.2.0_dynamic_elo/international_shootouts.csv"


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def draw_after_extra_time_probability(strength_diff: float) -> float:
    return max(0.10, min(0.28, 0.24 - 0.10 * abs(strength_diff)))


def build_historical_shootout_scores(path: Path = SHOOTOUTS) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["home_team", "away_team", "winner"]:
        df[col] = df[col].map(canonical_team_name)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]) | set(df["winner"]))
    rows = []
    for team in teams:
        played = df[df["home_team"].eq(team) | df["away_team"].eq(team)]
        wins = played["winner"].eq(team).sum()
        rows.append({"team": team, "shootouts": int(len(played)), "shootout_wins": int(wins), "historical_shootout_score": float((wins + 1) / (len(played) + 2))})
    return pd.DataFrame(rows)


def penalty_strength(dynamic_elo_score: float, goalkeeper_score: float = 0.5, historical_shootout_score: float = 0.5, recent_form_score: float = 0.5) -> float:
    return 0.45 * dynamic_elo_score + 0.25 * goalkeeper_score + 0.15 * historical_shootout_score + 0.15 * recent_form_score


def penalty_home_probability(home_penalty_strength: float, away_penalty_strength: float, scale_penalty: float = 4.0) -> float:
    return logistic(scale_penalty * (home_penalty_strength - away_penalty_strength))


def knockout_resolution_probability(home_strength: float, away_strength: float, home_penalty_strength: float, away_penalty_strength: float) -> dict[str, float]:
    diff = home_strength - away_strength
    p_home_non_penalty_raw = logistic(5.0 * diff)
    p_draw = draw_after_extra_time_probability(diff)
    p_home_regular = (1 - p_draw) * p_home_non_penalty_raw
    p_away_regular = (1 - p_draw) * (1 - p_home_non_penalty_raw)
    p_pen_home = penalty_home_probability(home_penalty_strength, away_penalty_strength)
    return {"p_home_regular_extra": p_home_regular, "p_away_regular_extra": p_away_regular, "p_draw_after_extra_time": p_draw, "p_penalty_home": p_pen_home, "p_penalty_away": 1-p_pen_home}
