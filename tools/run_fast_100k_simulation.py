from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEED = 42
SIMULATIONS = 100_000


def minmax(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0 or s.max() == s.min():
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - s.min()) / (s.max() - s.min())
    if invert:
        out = 1 - out
    return out.fillna(0.5)


def logistic(x):
    return 1 / (1 + np.exp(-x))


teams_raw = pd.read_csv(PROCESSED / "copa_2026_master_team_dataset.csv")
features = teams_raw.copy()
features["elo_score"] = minmax(features["elo_rating"])
features["elo_rank_score"] = minmax(features["elo_rank"], invert=True)
features["squad_fc26_score"] = minmax(features["fc26_top23_overall_mean"])
features["squad_fc25_score"] = minmax(features["fc25_top23_overall_mean"])
features["market_fc25_score"] = minmax(np.log1p(features["fc25_total_value_eur_top23"].fillna(0)))
features["attack_score"] = minmax(features.get("fc26_attack_mean", features["fc26_top23_overall_mean"]))
features["defense_score"] = minmax(features.get("fc26_defense_mean", features["fc26_top23_overall_mean"]))
features["goalkeeper_score"] = minmax(features.get("fc26_top3_gk_mean", features["fc26_top23_overall_mean"]))
features["kaggle_prior_score"] = minmax(features.get("kaggle_champion_probability", features.get("champion_probability", features["combined_strength"])))
features["technical_weighted_strength"] = (
    0.18 * features["elo_score"]
    + 0.08 * features["elo_rank_score"]
    + 0.28 * features["squad_fc26_score"]
    + 0.12 * features["squad_fc25_score"]
    + 0.14 * features["market_fc25_score"]
    + 0.08 * features["attack_score"]
    + 0.06 * features["defense_score"]
    + 0.04 * features["goalkeeper_score"]
    + 0.02 * features["kaggle_prior_score"]
)
features["technical_rank"] = features["technical_weighted_strength"].rank(ascending=False, method="min").astype(int)
features = features.sort_values("technical_rank").reset_index(drop=True)

team_names = features["team"].tolist()
team_to_idx = {t: i for i, t in enumerate(team_names)}
idx_to_team = {i: t for t, i in team_to_idx.items()}
strength = features["technical_weighted_strength"].to_numpy(float)
confed = features.get("elo_confederation", pd.Series([None] * len(features))).to_numpy()
groups = sorted(features["group"].unique())
group_to_indices = {g: np.array([team_to_idx[t] for t in features.loc[features["group"] == g, "team"]], dtype=int) for g in groups}

fixtures = pd.read_csv(DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_group_fixtures.csv")
slots = pd.read_csv(DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_knockout_slots.csv").sort_values("match_id")

rng = np.random.default_rng(SEED)
N = SIMULATIONS
T = len(team_names)
points = np.zeros((N, T), dtype=np.int16)
wins = np.zeros((N, T), dtype=np.int16)
draws = np.zeros((N, T), dtype=np.int16)
losses = np.zeros((N, T), dtype=np.int16)
gf = np.zeros((N, T), dtype=np.int16)
ga = np.zeros((N, T), dtype=np.int16)
first_match_rows = []


def technical_multiplier_idx(a, b):
    ca, cb = confed[a], confed[b]
    if pd.isna(ca) or pd.isna(cb):
        return 1.10
    return 1.00 if ca == cb else 1.15


def knockout_p_home(home_idx, away_idx):
    diff = (strength[home_idx] - strength[away_idx]) * technical_multiplier_idx(home_idx, away_idx)
    return float(logistic(5.0 * diff))


for row in fixtures.sort_values("match_id").itertuples():
    a = team_to_idx[row.home_team]
    b = team_to_idx[row.away_team]
    diff = (strength[a] - strength[b]) * technical_multiplier_idx(a, b)
    p_a_raw = float(logistic(5.0 * diff))
    draw_p = float(np.clip(0.26 - 0.14 * abs(diff), 0.14, 0.30))
    p_a = (1 - draw_p) * p_a_raw
    p_b = (1 - draw_p) * (1 - p_a_raw)

    sample = rng.random(N)
    outcome = np.where(sample < p_a, 0, np.where(sample < p_a + draw_p, 1, 2))
    lambda_a = float(np.clip(0.85 + 1.55 * strength[a] - 0.70 * strength[b], 0.25, 3.5))
    lambda_b = float(np.clip(0.85 + 1.55 * strength[b] - 0.70 * strength[a], 0.25, 3.5))
    goals_a = rng.poisson(lambda_a, N).astype(np.int16)
    goals_b = rng.poisson(lambda_b, N).astype(np.int16)

    mask = (outcome == 0) & (goals_a <= goals_b)
    goals_a[mask] = goals_b[mask] + 1
    mask = (outcome == 2) & (goals_b <= goals_a)
    goals_b[mask] = goals_a[mask] + 1
    mask = outcome == 1
    avg = np.rint((goals_a[mask] + goals_b[mask]) / 2).astype(np.int16)
    goals_a[mask] = avg
    goals_b[mask] = avg

    gf[:, a] += goals_a
    ga[:, a] += goals_b
    gf[:, b] += goals_b
    ga[:, b] += goals_a

    a_win = outcome == 0
    draw = outcome == 1
    b_win = outcome == 2
    points[a_win, a] += 3
    points[b_win, b] += 3
    points[draw, a] += 1
    points[draw, b] += 1
    wins[a_win, a] += 1
    wins[b_win, b] += 1
    draws[draw, a] += 1
    draws[draw, b] += 1
    losses[b_win, a] += 1
    losses[a_win, b] += 1

    first_match_rows.append({
        "match_id": row.match_id,
        "group": row.group,
        "team_a": row.home_team,
        "team_b": row.away_team,
        "goals_a": int(goals_a[0]),
        "goals_b": int(goals_b[0]),
        "outcome": ["team_a_win", "draw", "team_b_win"][int(outcome[0])],
        "p_team_a_win": p_a,
        "p_draw": draw_p,
        "p_team_b_win": p_b,
    })

# Rank group tables for all simulations.
gd = gf - ga
group_rankings: dict[str, np.ndarray] = {}
position = np.zeros((N, T), dtype=np.int8)
for g, idxs in group_to_indices.items():
    # lexsort sorts ascending and uses last key as primary; negative means descending.
    order_local = np.lexsort((
        -strength[idxs][None, :].repeat(N, axis=0),
        -wins[:, idxs],
        -gf[:, idxs],
        -gd[:, idxs],
        -points[:, idxs],
    ), axis=1)
    ranked = idxs[order_local]
    group_rankings[g] = ranked
    for pos in range(4):
        position[np.arange(N), ranked[:, pos]] = pos + 1

phase_cols = [
    "group_stage_probability",
    "round_of_32_probability",
    "round_of_16_probability",
    "quarter_final_probability",
    "semi_final_probability",
    "final_probability",
    "champion_probability",
]
counts = {col: np.zeros(T, dtype=np.int32) for col in phase_cols}
counts["group_stage_probability"][:] = N
counts["round_of_32_probability"] += (position <= 2).sum(axis=0)

# Best thirds per simulation.
third_candidates = np.column_stack([group_rankings[g][:, 2] for g in groups])
third_order = np.lexsort((
    -strength[third_candidates],
    -wins[np.arange(N)[:, None], third_candidates],
    -gf[np.arange(N)[:, None], third_candidates],
    -gd[np.arange(N)[:, None], third_candidates],
    -points[np.arange(N)[:, None], third_candidates],
), axis=1)
best_thirds_by_sim = third_candidates[np.arange(N)[:, None], third_order[:, :8]]
for sim in range(N):
    counts["round_of_32_probability"][best_thirds_by_sim[sim]] += 1

# First simulation tables.
first_table_rows = []
for g in groups:
    for pos, idx in enumerate(group_rankings[g][0], start=1):
        first_table_rows.append({
            "group": g,
            "team": idx_to_team[int(idx)],
            "played": 3,
            "wins": int(wins[0, idx]),
            "draws": int(draws[0, idx]),
            "losses": int(losses[0, idx]),
            "goals_for": int(gf[0, idx]),
            "goals_against": int(ga[0, idx]),
            "points": int(points[0, idx]),
            "technical_weighted_strength": float(strength[idx]),
            "goal_difference": int(gd[0, idx]),
            "group_position": pos,
        })
first_table = pd.DataFrame(first_table_rows)

stage_to_col = {
    "Round of 32": "round_of_16_probability",
    "Round of 16": "quarter_final_probability",
    "Quarter-final": "semi_final_probability",
    "Semi-final": "final_probability",
}
champion_names = []
first_bracket_rows = None

for sim in range(N):
    group_winners = {g: int(group_rankings[g][sim, 0]) for g in groups}
    group_runners = {g: int(group_rankings[g][sim, 1]) for g in groups}
    available_thirds = {features.loc[idx, "group"]: int(idx) for idx in best_thirds_by_sim[sim]}
    third_assignments = {}
    third_slots = slots[(slots["stage"] == "Round of 32") & ((slots["home_slot_type"] == "best_third") | (slots["away_slot_type"] == "best_third"))]
    for slot in third_slots.itertuples():
        side = "home" if slot.home_slot_type == "best_third" else "away"
        allowed_raw = getattr(slot, f"{side}_allowed_third_groups")
        allowed = str(allowed_raw).split(",") if pd.notna(allowed_raw) else []
        chosen = None
        for group in [features.loc[idx, "group"] for idx in best_thirds_by_sim[sim]]:
            if group in allowed and group in available_thirds:
                chosen = group
                break
        if chosen is None and available_thirds:
            chosen = next(iter(available_thirds))
        third_assignments[(int(slot.match_id), side)] = available_thirds.pop(chosen) if chosen else None

    match_results = {}
    rows = []
    for slot in slots.itertuples():
        def resolve(side):
            typ = getattr(slot, f"{side}_slot_type")
            if typ == "winner_group":
                return group_winners[getattr(slot, f"{side}_group_ref")]
            if typ == "runner_up_group":
                return group_runners[getattr(slot, f"{side}_group_ref")]
            if typ == "best_third":
                return third_assignments[(int(slot.match_id), side)]
            if typ == "winner_match":
                return match_results[int(getattr(slot, f"{side}_match_ref"))]["winner"]
            if typ == "loser_match":
                return match_results[int(getattr(slot, f"{side}_match_ref"))]["loser"]
            raise ValueError(typ)

        home = resolve("home")
        away = resolve("away")
        p_home = knockout_p_home(home, away)
        winner = home if rng.random() < p_home else away
        loser = away if winner == home else home
        match_results[int(slot.match_id)] = {"winner": winner, "loser": loser}
        if slot.stage in stage_to_col:
            counts[stage_to_col[slot.stage]][winner] += 1
        if slot.stage == "Final":
            counts["champion_probability"][winner] += 1
            champion_names.append(idx_to_team[winner])
        if sim == 0:
            rows.append({
                "match_id": int(slot.match_id),
                "stage": slot.stage,
                "slot_home": slot.slot_home,
                "slot_away": slot.slot_away,
                "team_home": idx_to_team[home],
                "team_away": idx_to_team[away],
                "p_home_win": p_home,
                "p_away_win": 1 - p_home,
                "most_likely_winner": idx_to_team[home if p_home >= 0.5 else away],
                "simulated_winner": idx_to_team[winner],
                "simulated_loser": idx_to_team[loser],
            })
    if sim == 0:
        first_bracket_rows = pd.DataFrame(rows)

probabilities = pd.DataFrame({"team": team_names})
for col in phase_cols:
    probabilities[col] = counts[col] / N
probabilities = probabilities.merge(features[["team", "group", "technical_rank", "technical_weighted_strength"]], on="team", how="left")
probabilities = probabilities.sort_values("champion_probability", ascending=False).reset_index(drop=True)
assert probabilities["team"].nunique() == 48
assert abs(probabilities["champion_probability"].sum() - 1.0) < 1e-9

probabilities.to_csv(OUTPUTS / "updated_2026_probabilities.csv", index=False)
first_bracket_rows.to_csv(OUTPUTS / "updated_round_of_32_bracket.csv", index=False)
pd.DataFrame(first_match_rows).to_csv(OUTPUTS / "updated_group_stage_matches_seed42.csv", index=False)
first_table.to_csv(OUTPUTS / "updated_group_stage_table_seed42.csv", index=False)

summary = "# Simulação probabilística Copa 2026\n\n"
summary += f"Simulações: {SIMULATIONS}\n"
summary += f"Seed: {SEED}\n"
summary += f"Seleções: {features['team'].nunique()}\n"
summary += "Formato: 48 seleções, 12 grupos de 4, 32 no mata-mata.\n\n"
summary += "## Top 10 favoritos\n\n"
summary += probabilities[["team", "champion_probability", "final_probability", "semi_final_probability"]].head(10).to_string(index=False)
summary += "\n\n## Round of 32 da primeira simulação\n\n"
r32 = first_bracket_rows[first_bracket_rows["stage"] == "Round of 32"]
summary += r32[["match_id", "team_home", "team_away", "most_likely_winner", "simulated_winner"]].to_string(index=False)
(OUTPUTS / "updated_2026_simulation_summary.md").write_text(summary, encoding="utf-8")

model_card = """# Model Card — Copa 2026 Predictor Atualizado

## Dados
Usa datasets Kaggle em `Data/raw/kaggle` e base combinada em `Data/processed/copa_2026_master_team_dataset.csv`.

## Método
Simulação Monte Carlo estocástica com força por seleção baseada em Elo, rankings, ratings EA FC 26, EA FC 25, valor de mercado, ataque, defesa e goleiros.

## Ponto metodológico central
Para o formato de 48 seleções, a qualidade individual recebe peso alto porque várias seleções de África, Ásia e Oceania terão poucos confrontos recentes contra potências europeias e sul-americanas. O multiplicador intercontinental aumenta a influência da diferença técnica quando o histórico direto é provavelmente escasso.

## Limitações
Resultado é previsão probabilística, não resultado oficial. Lesões, convocações definitivas, forma imediatamente anterior ao torneio e decisões oficiais finais de chaveamento podem alterar as probabilidades. Ratings de jogos e valor de mercado são proxies imperfeitos de qualidade futebolística.
"""
(OUTPUTS / "updated_model_card.md").write_text(model_card, encoding="utf-8")

print(probabilities.head(10).to_string(index=False))
print(f"Simulações: {SIMULATIONS}")
