from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from build_actual_2026_results import load_actual_results
from build_recent_form_features import build_recent_form_features
from dynamic_elo import calculate_dynamic_elo, load_initial_ratings
from location_advantage import match_location_adjustment
from penalty_model import build_historical_shootout_scores, penalty_strength, knockout_resolution_probability
from team_name_normalization import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEED = 42
SIMULATIONS = 100_000
WEIGHTS_PATH = PROCESSED / "model_weights_v0.2.0.json"

WEIGHTS = {
    "dynamic_elo": 0.42,
    "squad_fc26": 0.18,
    "squad_fc25_market_experience": 0.08,
    "recent_form_last_10": 0.16,
    "recent_form_six_month": 0.08,
    "location": 0.05,
    "external_prior_v0_1": 0.03,
}


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


def write_weights() -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v0.2.0",
        "description": "Pesos auditáveis para força composta com Elo dinâmico, forma recente, sede/local e prior v0.1.0.",
        "weights": WEIGHTS,
    }
    WEIGHTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_feature_table() -> pd.DataFrame:
    write_weights()
    actual = load_actual_results(DATA / "raw/external/v0.2.0_dynamic_elo/openfootball_worldcup_2026.json")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    actual.to_csv(PROCESSED / "actual_2026_matches.csv", index=False)
    initial = load_initial_ratings(PROCESSED / "copa_2026_master_team_dataset.csv")
    dynamic_elo = calculate_dynamic_elo(actual, initial)
    dynamic_elo.to_csv(PROCESSED / "dynamic_elo_after_group_stage.csv", index=False)
    recent = build_recent_form_features()
    recent.to_csv(PROCESSED / "recent_form_features.csv", index=False)

    features = pd.read_csv(PROCESSED / "copa_2026_master_team_dataset.csv")
    features["team"] = features["team"].map(canonical_team_name)
    features = features.merge(dynamic_elo[["team", "dynamic_elo_rating", "initial_elo_rating", "dynamic_elo_delta"]], on="team", how="left")
    features = features.merge(recent, on="team", how="left")

    features["dynamic_elo_rating"] = features["dynamic_elo_rating"].fillna(features["elo_rating"]).fillna(1500)
    features["dynamic_elo_score"] = minmax(features["dynamic_elo_rating"])
    features["squad_fc26_score"] = minmax(features["fc26_top23_overall_mean"])
    features["squad_fc25_market_experience_score"] = (
        0.45 * minmax(features["fc25_top23_overall_mean"])
        + 0.35 * minmax(np.log1p(features["fc25_total_value_eur_top23"].fillna(0)))
        + 0.20 * minmax(features["fc25_int_caps_top23"].fillna(0))
    )
    features["last_10_form_score"] = (
        0.50 * minmax(features["last_10_points_per_match"])
        + 0.30 * minmax(features["last_10_goal_difference_per_match"])
        + 0.20 * minmax(features["last_10_opponent_elo_mean"])
    )
    features["six_month_form_score"] = (
        0.60 * minmax(features["six_month_points_per_match"])
        + 0.30 * minmax(features["six_month_goal_difference_per_match"])
        + 0.10 * minmax(features["six_month_match_count"])
    )
    features["location_base_score"] = features["is_host_team"].fillna(False).astype(bool).map({True: 1.0, False: 0.35})
    features["external_prior_score"] = minmax(features.get("kaggle_champion_probability", features.get("champion_probability", 0.5)))
    features["model_strength_v0_2"] = (
        WEIGHTS["dynamic_elo"] * features["dynamic_elo_score"]
        + WEIGHTS["squad_fc26"] * features["squad_fc26_score"]
        + WEIGHTS["squad_fc25_market_experience"] * features["squad_fc25_market_experience_score"]
        + WEIGHTS["recent_form_last_10"] * features["last_10_form_score"]
        + WEIGHTS["recent_form_six_month"] * features["six_month_form_score"]
        + WEIGHTS["location"] * features["location_base_score"]
        + WEIGHTS["external_prior_v0_1"] * features["external_prior_score"]
    )
    features["technical_weighted_strength"] = features["model_strength_v0_2"]
    features["technical_rank"] = features["model_strength_v0_2"].rank(ascending=False, method="min").astype(int)
    features = features.sort_values("technical_rank").reset_index(drop=True)
    enriched_cols = ["team", "group", "technical_rank", "model_strength_v0_2", "dynamic_elo_rating", "dynamic_elo_delta", "last_10_points_per_match", "six_month_points_per_match"]
    features[enriched_cols].to_csv(PROCESSED / "v0.2.0_team_strength_features.csv", index=False)
    return features


def main() -> None:
    features = load_feature_table()
    team_names = features["team"].tolist()
    team_to_idx = {t: i for i, t in enumerate(team_names)}
    idx_to_team = {i: t for t, i in team_to_idx.items()}
    strength = features["model_strength_v0_2"].to_numpy(float)
    dynamic_elo = features["dynamic_elo_rating"].to_numpy(float)
    goalkeeper_score = minmax(features.get("fc26_top3_gk_mean", features["fc26_top23_overall_mean"])).to_numpy(float)
    recent_score = features["last_10_form_score"].to_numpy(float)
    confed = features.get("elo_confederation", pd.Series([None] * len(features))).to_numpy()
    groups = sorted(features["group"].unique())
    group_to_indices = {g: np.array([team_to_idx[t] for t in features.loc[features["group"] == g, "team"]], dtype=int) for g in groups}

    fixtures = pd.read_csv(DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_group_fixtures.csv")
    slots = pd.read_csv(DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_knockout_slots.csv").sort_values("match_id")
    for c in ["home_team", "away_team"]:
        if c in fixtures.columns:
            fixtures[c] = fixtures[c].map(canonical_team_name)

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

    def adjusted_diff(a, b, host_country=None):
        ha, hb = match_location_adjustment(idx_to_team[a], idx_to_team[b], host_country, confed[a] if not pd.isna(confed[a]) else None, confed[b] if not pd.isna(confed[b]) else None)
        elo_location_component = (ha - hb) / 4000.0
        return (strength[a] - strength[b] + elo_location_component) * technical_multiplier_idx(a, b)

    for row in fixtures.sort_values("match_id").itertuples():
        a = team_to_idx[canonical_team_name(row.home_team)]
        b = team_to_idx[canonical_team_name(row.away_team)]
        host = canonical_team_name(getattr(row, "host_country", ""))
        diff = adjusted_diff(a, b, host)
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
        mask = (outcome == 0) & (goals_a <= goals_b); goals_a[mask] = goals_b[mask] + 1
        mask = (outcome == 2) & (goals_b <= goals_a); goals_b[mask] = goals_a[mask] + 1
        mask = outcome == 1
        avg = np.rint((goals_a[mask] + goals_b[mask]) / 2).astype(np.int16); goals_a[mask] = avg; goals_b[mask] = avg
        gf[:, a] += goals_a; ga[:, a] += goals_b; gf[:, b] += goals_b; ga[:, b] += goals_a
        a_win = outcome == 0; draw = outcome == 1; b_win = outcome == 2
        points[a_win, a] += 3; points[b_win, b] += 3; points[draw, a] += 1; points[draw, b] += 1
        wins[a_win, a] += 1; wins[b_win, b] += 1; draws[draw, a] += 1; draws[draw, b] += 1; losses[b_win, a] += 1; losses[a_win, b] += 1
        first_match_rows.append({"match_id": row.match_id, "group": row.group, "team_a": row.home_team, "team_b": row.away_team, "host_country": host, "venue": getattr(row, "venue", ""), "stadium": getattr(row, "stadium", ""), "city": getattr(row, "city", ""), "neutral_for_model": getattr(row, "neutral_for_model", True), "goals_a": int(goals_a[0]), "goals_b": int(goals_b[0]), "outcome": ["team_a_win", "draw", "team_b_win"][int(outcome[0])], "p_team_a_win": p_a, "p_draw": draw_p, "p_team_b_win": p_b})

    gd = gf - ga
    group_rankings = {}
    position = np.zeros((N, T), dtype=np.int8)
    for g, idxs in group_to_indices.items():
        order_local = np.lexsort((-strength[idxs][None, :].repeat(N, axis=0), -wins[:, idxs], -gf[:, idxs], -gd[:, idxs], -points[:, idxs]), axis=1)
        ranked = idxs[order_local]
        group_rankings[g] = ranked
        for pos in range(4):
            position[np.arange(N), ranked[:, pos]] = pos + 1

    phase_cols = ["group_stage_probability", "round_of_32_probability", "round_of_16_probability", "quarter_final_probability", "semi_final_probability", "final_probability", "champion_probability"]
    counts = {col: np.zeros(T, dtype=np.int32) for col in phase_cols}
    counts["group_stage_probability"][:] = N
    counts["round_of_32_probability"] += (position <= 2).sum(axis=0)
    third_candidates = np.column_stack([group_rankings[g][:, 2] for g in groups])
    third_order = np.lexsort((-strength[third_candidates], -wins[np.arange(N)[:, None], third_candidates], -gf[np.arange(N)[:, None], third_candidates], -gd[np.arange(N)[:, None], third_candidates], -points[np.arange(N)[:, None], third_candidates]), axis=1)
    best_thirds_by_sim = third_candidates[np.arange(N)[:, None], third_order[:, :8]]
    for sim in range(N):
        counts["round_of_32_probability"][best_thirds_by_sim[sim]] += 1

    first_table_rows = []
    for g in groups:
        for pos, idx in enumerate(group_rankings[g][0], start=1):
            first_table_rows.append({"group": g, "team": idx_to_team[int(idx)], "played": 3, "wins": int(wins[0, idx]), "draws": int(draws[0, idx]), "losses": int(losses[0, idx]), "goals_for": int(gf[0, idx]), "goals_against": int(ga[0, idx]), "points": int(points[0, idx]), "model_strength_v0_2": float(strength[idx]), "dynamic_elo_rating": float(dynamic_elo[idx]), "goal_difference": int(gd[0, idx]), "group_position": pos})
    first_table = pd.DataFrame(first_table_rows)

    shootout_scores = build_historical_shootout_scores()
    shootout_lookup = dict(zip(shootout_scores["team"], shootout_scores["historical_shootout_score"]))
    dyn_score = minmax(pd.Series(dynamic_elo)).to_numpy(float)
    pen_strength = np.array([penalty_strength(dyn_score[i], goalkeeper_score[i], shootout_lookup.get(idx_to_team[i], 0.5), recent_score[i]) for i in range(T)])
    stage_to_col = {"Round of 32": "round_of_16_probability", "Round of 16": "quarter_final_probability", "Quarter-final": "semi_final_probability", "Semi-final": "final_probability"}
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
                    chosen = group; break
            if chosen is None and available_thirds:
                chosen = next(iter(available_thirds))
            third_assignments[(int(slot.match_id), side)] = available_thirds.pop(chosen) if chosen else None
        match_results = {}; rows = []
        for slot in slots.itertuples():
            def resolve(side):
                typ = getattr(slot, f"{side}_slot_type")
                if typ == "winner_group": return group_winners[getattr(slot, f"{side}_group_ref")]
                if typ == "runner_up_group": return group_runners[getattr(slot, f"{side}_group_ref")]
                if typ == "best_third": return third_assignments[(int(slot.match_id), side)]
                if typ == "winner_match": return match_results[int(getattr(slot, f"{side}_match_ref"))]["winner"]
                if typ == "loser_match": return match_results[int(getattr(slot, f"{side}_match_ref"))]["loser"]
                raise ValueError(typ)
            home = resolve("home"); away = resolve("away")
            host = canonical_team_name(getattr(slot, "host_country", ""))
            probs = knockout_resolution_probability(float(strength[home]), float(strength[away]), float(pen_strength[home]), float(pen_strength[away]))
            sample = rng.random()
            if sample < probs["p_home_regular_extra"]:
                winner, loser, resolution = home, away, "regular_time_or_extra_time"
            elif sample < probs["p_home_regular_extra"] + probs["p_away_regular_extra"]:
                winner, loser, resolution = away, home, "regular_time_or_extra_time"
            else:
                winner = home if rng.random() < probs["p_penalty_home"] else away
                loser = away if winner == home else home
                resolution = "penalties"
            match_results[int(slot.match_id)] = {"winner": winner, "loser": loser}
            if slot.stage in stage_to_col: counts[stage_to_col[slot.stage]][winner] += 1
            if slot.stage == "Final": counts["champion_probability"][winner] += 1; champion_names.append(idx_to_team[winner])
            if sim == 0:
                p_home_total = probs["p_home_regular_extra"] + probs["p_draw_after_extra_time"] * probs["p_penalty_home"]
                rows.append({"match_id": int(slot.match_id), "stage": slot.stage, "slot_home": slot.slot_home, "slot_away": slot.slot_away, "team_home": idx_to_team[home], "team_away": idx_to_team[away], "host_country": host, "stadium": getattr(slot, "stadium", ""), "city": getattr(slot, "city", ""), "p_home_win": p_home_total, "p_away_win": 1 - p_home_total, "p_penalty_home": probs["p_penalty_home"], "p_penalty_away": probs["p_penalty_away"], "p_draw_after_extra_time": probs["p_draw_after_extra_time"], "knockout_resolution": resolution, "won_by_penalties": resolution == "penalties", "most_likely_winner": idx_to_team[home if p_home_total >= 0.5 else away], "simulated_winner": idx_to_team[winner], "simulated_loser": idx_to_team[loser]})
        if sim == 0: first_bracket_rows = pd.DataFrame(rows)

    probabilities = pd.DataFrame({"team": team_names})
    for col in phase_cols:
        probabilities[col] = counts[col] / N
    probabilities = probabilities.merge(features[["team", "group", "technical_rank", "model_strength_v0_2", "dynamic_elo_rating", "dynamic_elo_delta"]], on="team", how="left")
    probabilities = probabilities.sort_values("champion_probability", ascending=False).reset_index(drop=True)
    assert probabilities["team"].nunique() == 48
    assert abs(probabilities["champion_probability"].sum() - 1.0) < 1e-9
    probabilities.to_csv(OUTPUTS / "updated_2026_probabilities.csv", index=False)
    first_bracket_rows.to_csv(OUTPUTS / "updated_round_of_32_bracket.csv", index=False)
    pd.DataFrame(first_match_rows).to_csv(OUTPUTS / "updated_group_stage_matches_seed42.csv", index=False)
    first_table.to_csv(OUTPUTS / "updated_group_stage_table_seed42.csv", index=False)

    summary = "# Simulação probabilística Copa 2026 — v0.2.0\n\n"
    summary += f"Simulações: {SIMULATIONS}\nSeed: {SEED}\nSeleções: {features['team'].nunique()}\n"
    summary += "Modelo: Monte Carlo com força composta v0.2.0, Elo dinâmico pós-fase de grupos, forma recente, sede/local e pênaltis explícitos.\n\n"
    summary += "## Top 10 favoritos\n\n" + probabilities[["team", "champion_probability", "final_probability", "semi_final_probability", "dynamic_elo_rating"]].head(10).to_string(index=False)
    summary += "\n\n## Mata-mata da primeira simulação\n\n" + first_bracket_rows[["match_id", "team_home", "team_away", "knockout_resolution", "won_by_penalties", "simulated_winner"]].head(32).to_string(index=False)
    (OUTPUTS / "updated_2026_simulation_summary.md").write_text(summary, encoding="utf-8")
    model_card = """# Model Card — Copa 2026 Predictor v0.2.0

## Dados
Usa Kaggle para estrutura do torneio/elencos, OpenFootball para resultados/fixtures da Copa 2026, World Football Elo e international-results para forma recente e ratings.

## Método
A v0.2.0 mantém Monte Carlo com 100.000 torneios, mas troca a força majoritariamente estática por `model_strength_v0_2`: Elo dinâmico pós-fase de grupos, força de elenco EA FC, forma recente, sede/local e prior externo v0.1.0.

## Pênaltis
O mata-mata separa vitória em 90/120 minutos, empate após prorrogação e disputa de pênaltis. O bracket contém `knockout_resolution`, `won_by_penalties`, `p_penalty_home` e `p_penalty_away`.

## Limitações
Resultado é previsão probabilística, não fato. Pesos de sede/local e forma são hipóteses iniciais e devem ser calibrados em backtests futuros.
"""
    (OUTPUTS / "updated_model_card.md").write_text(model_card, encoding="utf-8")
    print(probabilities.head(10).to_string(index=False))
    print(f"Simulações: {SIMULATIONS}")


if __name__ == "__main__":
    main()
