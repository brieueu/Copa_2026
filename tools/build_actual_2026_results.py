from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

try:
    from team_name_normalization import canonical_team_name
except ImportError:
    from tools.team_name_normalization import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "Data/raw/external/v0.2.0_dynamic_elo/openfootball_worldcup_2026.json"
OUT_CSV = ROOT / "Data/processed/actual_2026_matches.csv"
HOST_BY_GROUND = {
    "Mexico City": "Mexico",
    "Guadalajara (Zapopan)": "Mexico",
    "Monterrey (Guadalupe)": "Mexico",
    "Toronto": "Canada",
    "Vancouver": "Canada",
}
US_CITIES = {"Atlanta", "Boston (Foxborough)", "Dallas (Arlington)", "Houston", "Kansas City", "Los Angeles (Inglewood)", "Miami", "New York New Jersey (East Rutherford)", "Philadelphia", "San Francisco Bay Area (Santa Clara)", "Seattle"}


def _score(match: dict) -> tuple[float | None, float | None]:
    score = match.get("score") or {}
    ft = score.get("ft") or score.get("score")
    if isinstance(ft, list) and len(ft) >= 2:
        return ft[0], ft[1]
    return None, None


def infer_host_country(ground: object) -> str:
    ground = str(ground or "").strip()
    if ground in HOST_BY_GROUND:
        return HOST_BY_GROUND[ground]
    if ground in US_CITIES:
        return "United States"
    return ""


def load_actual_results(path: Path = RAW_JSON) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for i, match in enumerate(data.get("matches", []), start=1):
        home_score, away_score = _score(match)
        group = match.get("group")
        stage_type = "group" if group else "knockout"
        ground = match.get("ground") or match.get("city") or ""
        rows.append({
            "match_id": match.get("num") or i,
            "date": match.get("date"),
            "time": match.get("time"),
            "round": match.get("round"),
            "group": group,
            "stage_type": stage_type,
            "home_team": canonical_team_name(match.get("team1")),
            "away_team": canonical_team_name(match.get("team2")),
            "home_score": home_score,
            "away_score": away_score,
            "city": ground,
            "stadium": match.get("stadium") or "",
            "host_country": infer_host_country(ground),
            "neutral_for_model": canonical_team_name(match.get("team1")) != infer_host_country(ground) and canonical_team_name(match.get("team2")) != infer_host_country(ground),
            "has_result": pd.notna(home_score) and pd.notna(away_score),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_actual_results(RAW_JSON)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} shape={df.shape}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
