from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

try:
    from location_advantage import match_location_adjustment
    from team_name_normalization import canonical_team_name
except ImportError:
    from tools.location_advantage import match_location_adjustment
    from tools.team_name_normalization import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL = ROOT / "Data/processed/copa_2026_master_team_dataset.csv"
DEFAULT_MATCHES = ROOT / "Data/processed/actual_2026_matches.csv"
DEFAULT_OUT = ROOT / "Data/processed/dynamic_elo_after_group_stage.csv"


def expected_score(rating_for: float, rating_against: float) -> float:
    return 1 / (10 ** (-(rating_for - rating_against) / 400) + 1)


def margin_multiplier(home_score: int | float, away_score: int | float) -> float:
    margin = abs(float(home_score) - float(away_score))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    if margin == 3:
        return 1.75
    return (11 + margin) / 8


def tournament_k(tournament: str | None = None, stage_type: str | None = None) -> float:
    text = f"{tournament or ''} {stage_type or ''}".lower()
    if "world cup" in text or "group" in text or "knockout" in text:
        return 60.0
    if "qual" in text or "continental" in text:
        return 40.0
    return 20.0


def update_elo(ratings: dict[str, float], home_team: str, away_team: str, home_score: int | float, away_score: int | float, k: float = 60.0, local_advantage: float = 0.0, away_local_advantage: float = 0.0) -> dict[str, float]:
    home = canonical_team_name(home_team)
    away = canonical_team_name(away_team)
    updated = dict(ratings)
    rh = float(updated.get(home, 1500.0))
    ra = float(updated.get(away, 1500.0))
    eh = expected_score(rh + local_advantage, ra + away_local_advantage)
    if home_score > away_score:
        sh = 1.0
    elif home_score == away_score:
        sh = 0.5
    else:
        sh = 0.0
    g = margin_multiplier(home_score, away_score)
    delta = k * g * (sh - eh)
    updated[home] = rh + delta
    updated[away] = ra - delta
    return updated


def load_initial_ratings(path: Path = DEFAULT_INITIAL) -> dict[str, float]:
    df = pd.read_csv(path)
    rating_col = "dynamic_elo_rating" if "dynamic_elo_rating" in df.columns else "elo_rating"
    return {canonical_team_name(r.team): float(getattr(r, rating_col)) for r in df.itertuples() if pd.notna(getattr(r, rating_col))}


def calculate_dynamic_elo(matches: pd.DataFrame, initial_ratings: dict[str, float]) -> pd.DataFrame:
    ratings = dict(initial_ratings)
    rows = []
    played = matches[matches.get("stage_type", "group").eq("group") if "stage_type" in matches.columns else matches.index == matches.index].copy()
    played = played[pd.to_numeric(played["home_score"], errors="coerce").notna() & pd.to_numeric(played["away_score"], errors="coerce").notna()]
    for match in played.sort_values(["date", "match_id"], na_position="last").itertuples():
        home = canonical_team_name(match.home_team)
        away = canonical_team_name(match.away_team)
        before_home = ratings.get(home, 1500.0)
        before_away = ratings.get(away, 1500.0)
        host = getattr(match, "host_country", "")
        adv_home, adv_away = match_location_adjustment(home, away, host)
        ratings = update_elo(ratings, home, away, match.home_score, match.away_score, k=60, local_advantage=adv_home, away_local_advantage=adv_away)
        rows.append({"match_id": getattr(match, "match_id", None), "home_team": home, "away_team": away, "home_score": match.home_score, "away_score": match.away_score, "home_rating_before": before_home, "away_rating_before": before_away, "home_rating_after": ratings[home], "away_rating_after": ratings[away], "home_delta": ratings[home]-before_home, "away_delta": ratings[away]-before_away, "margin_multiplier": margin_multiplier(match.home_score, match.away_score), "host_country": host})
    out = pd.DataFrame({"team": list(ratings.keys()), "dynamic_elo_rating": list(ratings.values())})
    initial = pd.DataFrame({"team": list(initial_ratings.keys()), "initial_elo_rating": list(initial_ratings.values())})
    out = out.merge(initial, on="team", how="left")
    out["dynamic_elo_delta"] = out["dynamic_elo_rating"] - out["initial_elo_rating"].fillna(1500.0)
    out = out.sort_values("dynamic_elo_rating", ascending=False).reset_index(drop=True)
    out.attrs["match_updates"] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    matches = pd.read_csv(args.matches)
    initial = load_initial_ratings()
    out = calculate_dynamic_elo(matches, initial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} shape={out.shape}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
