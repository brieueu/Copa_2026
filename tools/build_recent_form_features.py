from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    from team_name_normalization import canonical_team_name
except ImportError:
    from tools.team_name_normalization import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS = ROOT / "Data/raw/external/v0.2.0_dynamic_elo/international_results.csv"
MASTER = ROOT / "Data/processed/copa_2026_master_team_dataset.csv"
OUT = ROOT / "Data/processed/recent_form_features.csv"
CUTOFF = pd.Timestamp("2026-06-11")
SIX_MONTH_START = CUTOFF - pd.DateOffset(months=6)


def _team_rows(results: pd.DataFrame, team: str) -> pd.DataFrame:
    home = results[results["home_team"].eq(team)].copy()
    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["opponent"] = home["away_team"]
    away = results[results["away_team"].eq(team)].copy()
    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["opponent"] = away["home_team"]
    df = pd.concat([home, away], ignore_index=True).sort_values("date")
    df["points"] = df.apply(lambda r: 3 if r.goals_for > r.goals_against else 1 if r.goals_for == r.goals_against else 0, axis=1)
    return df


def _features_for(team_rows: pd.DataFrame, elo_lookup: dict[str, float]) -> dict[str, float]:
    last10 = team_rows.tail(10)
    six = team_rows[team_rows["date"] >= SIX_MONTH_START]
    def pack(prefix, df):
        if df.empty:
            return {f"{prefix}_points_per_match": 0.0, f"{prefix}_goal_difference_per_match": 0.0, f"{prefix}_win_rate": 0.0, f"{prefix}_opponent_elo_mean": 1500.0, f"{prefix}_match_count": 0}
        return {f"{prefix}_points_per_match": float(df["points"].mean()), f"{prefix}_goal_difference_per_match": float((df["goals_for"]-df["goals_against"]).mean()), f"{prefix}_win_rate": float((df["points"] == 3).mean()), f"{prefix}_opponent_elo_mean": float(df["opponent"].map(elo_lookup).fillna(1500.0).mean()), f"{prefix}_match_count": int(len(df))}
    out = pack("last_10", last10)
    six_out = pack("six_month", six)
    out.update({k: v for k, v in six_out.items() if k != "six_month_win_rate"})
    return out


def build_recent_form_features(results_path: Path = RAW_RESULTS, master_path: Path = MASTER) -> pd.DataFrame:
    results = pd.read_csv(results_path, parse_dates=["date"])
    results["home_team"] = results["home_team"].map(canonical_team_name)
    results["away_team"] = results["away_team"].map(canonical_team_name)
    results = results[results["date"] < CUTOFF]
    master = pd.read_csv(master_path)
    teams = master["team"].map(canonical_team_name).tolist()
    elo_lookup = dict(zip(master["team"].map(canonical_team_name), pd.to_numeric(master.get("elo_rating", 1500), errors="coerce").fillna(1500)))
    rows = []
    for team in teams:
        tr = _team_rows(results, team)
        rows.append({"team": team, **_features_for(tr, elo_lookup)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build_recent_form_features()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} shape={df.shape}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
