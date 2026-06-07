from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path.cwd()
NOTEBOOK = ROOT / "Vencedor_Copa_2026_Notebook.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(r"""
# Previsor probabilístico da Copa do Mundo FIFA 2026 — dados Kaggle + Monte Carlo

Este notebook é a versão principal atualizada do projeto. Ele substitui a antiga base embutida por dados em `Data/`, usa uma base combinada gerada a partir de datasets Kaggle e executa simulações estocásticas para estimar possíveis resultados da Copa 2026.

Importante: os resultados são **previsões probabilísticas**, não resultados oficiais.

Ideia central da modelagem: no formato de 48 seleções, muitas equipes de África, Ásia e Oceania enfrentarão potências europeias e sul-americanas sem histórico direto recente suficiente. Por isso, o modelo dá peso alto à qualidade individual do elenco, usando ratings EA FC 26/EA FC 25 e valor de mercado como proxies técnicos.
"""))

    cells.append(md("""
## 1. Configuração geral e leitura da pasta Data
"""))

    cells.append(code(r'''
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
SIMULATIONS = 10_000
rng = np.random.default_rng(SEED)

ROOT = Path.cwd()
DATA = ROOT / "Data"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

base_path = PROCESSED / "copa_2026_master_team_dataset.csv"
manifest_path = DATA / "dataset_manifest.json"
assert base_path.exists(), f"Base processada não encontrada: {base_path}"

teams = pd.read_csv(base_path)
print("Projeto:", ROOT)
print("Base:", base_path)
print("Shape:", teams.shape)
print("Seleções:", teams["team"].nunique())
print("Grupos:", teams["group"].nunique())
teams.head()
'''))

    cells.append(md("""
## 2. Fontes de dados Kaggle

O notebook consome a pasta `Data/` recriada no projeto. Os dados brutos vieram dos datasets Kaggle indicados e a base final está em `Data/processed/copa_2026_master_team_dataset.csv`.
"""))

    cells.append(code(r'''
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fontes = pd.DataFrame([
        {
            "label": item.get("label"),
            "dataset": item.get("dataset"),
            "local_path": item.get("local_path"),
            "arquivos": len(item.get("files", [])),
        }
        for item in manifest
    ])
else:
    fontes = pd.DataFrame()
fontes
'''))

    cells.append(md("""
## 3. Auditoria da estrutura de 48 seleções
"""))

    cells.append(code(r'''
group_counts = teams.groupby("group")["team"].nunique().sort_index()
assert teams["team"].nunique() == 48, teams["team"].nunique()
assert teams["group"].nunique() == 12, teams["group"].nunique()
assert group_counts.eq(4).all(), group_counts.to_dict()
print("OK: 48 seleções, 12 grupos, 4 seleções por grupo")
group_counts.to_frame("seleções")
'''))

    cells.append(md("""
## 4. Engenharia de features com peso técnico individual

A força atualizada combina Elo, ranking, ratings de elenco, valor de mercado e ratings por setor. O prior probabilístico do dataset Kaggle entra com peso baixo para evitar circularidade.
"""))

    cells.append(code(r'''
def minmax(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0 or s.max() == s.min():
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - s.min()) / (s.max() - s.min())
    if invert:
        out = 1 - out
    return out.fillna(0.5)

features = teams.copy()
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

assert features["technical_weighted_strength"].between(0, 1).all()
assert features["technical_rank"].nunique() >= 40

features[[
    "team", "group", "technical_rank", "technical_weighted_strength", "elo_rating",
    "fc26_top23_overall_mean", "fc25_top23_overall_mean", "fc25_total_value_eur_top23",
]].head(15)
'''))

    cells.append(md("""
## 5. Modelo probabilístico de partida

A função abaixo gera probabilidades de vitória/empate/derrota. Para confrontos entre confederações diferentes, a diferença técnica recebe multiplicador maior, representando a ausência de histórico direto recente suficiente no novo formato de 48 seleções.
"""))

    cells.append(code(r'''
team_index = features.set_index("team")


def logistic(x):
    return 1 / (1 + np.exp(-x))


def technical_multiplier(team_a, team_b):
    confed_a = team_index.loc[team_a].get("elo_confederation", None)
    confed_b = team_index.loc[team_b].get("elo_confederation", None)
    if pd.isna(confed_a) or pd.isna(confed_b):
        return 1.10
    return 1.00 if confed_a == confed_b else 1.15


def match_probabilities(team_a, team_b, knockout=False):
    a = team_index.loc[team_a]
    b = team_index.loc[team_b]
    diff = float(a["technical_weighted_strength"] - b["technical_weighted_strength"])
    diff *= technical_multiplier(team_a, team_b)
    p_a_raw = float(logistic(5.0 * diff))

    if knockout:
        return {"team_a_win": p_a_raw, "team_b_win": 1 - p_a_raw}

    draw = float(np.clip(0.26 - 0.14 * abs(diff), 0.14, 0.30))
    return {
        "team_a_win": float((1 - draw) * p_a_raw),
        "draw": draw,
        "team_b_win": float((1 - draw) * (1 - p_a_raw)),
    }

sample_p = match_probabilities("Spain", "France")
assert abs(sum(sample_p.values()) - 1.0) < 1e-9
sample_p
'''))

    cells.append(md("""
## 6. Simulação estocástica da fase de grupos
"""))

    cells.append(code(r'''
group_fixtures_path = DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_group_fixtures.csv"
knockout_slots_path = DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_knockout_slots.csv"
assert group_fixtures_path.exists(), group_fixtures_path
assert knockout_slots_path.exists(), knockout_slots_path

group_fixtures = pd.read_csv(group_fixtures_path)
knockout_slots = pd.read_csv(knockout_slots_path)
assert len(group_fixtures) == 72
assert len(knockout_slots[knockout_slots["stage"] == "Round of 32"]) == 16
print("Jogos de fase de grupos:", len(group_fixtures))
print("Jogos no chaveamento:", len(knockout_slots))
group_fixtures.head()
'''))

    cells.append(code(r'''
def simulate_group_match(team_a, team_b, rng):
    probs = match_probabilities(team_a, team_b, knockout=False)
    outcome = rng.choice(
        ["team_a_win", "draw", "team_b_win"],
        p=[probs["team_a_win"], probs["draw"], probs["team_b_win"]],
    )

    a_strength = team_index.loc[team_a, "technical_weighted_strength"]
    b_strength = team_index.loc[team_b, "technical_weighted_strength"]
    lambda_a = float(np.clip(0.85 + 1.55 * a_strength - 0.70 * b_strength, 0.25, 3.5))
    lambda_b = float(np.clip(0.85 + 1.55 * b_strength - 0.70 * a_strength, 0.25, 3.5))
    ga = int(rng.poisson(lambda_a))
    gb = int(rng.poisson(lambda_b))

    if outcome == "team_a_win" and ga <= gb:
        ga = gb + 1
    elif outcome == "team_b_win" and gb <= ga:
        gb = ga + 1
    elif outcome == "draw":
        g = int(round((ga + gb) / 2))
        ga = gb = g
    return ga, gb, outcome, probs


def simulate_group_stage_once(rng):
    stats = {
        row.team: {
            "group": row.group,
            "team": row.team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "points": 0,
            "technical_weighted_strength": row.technical_weighted_strength,
        }
        for row in features.itertuples()
    }
    match_rows = []
    for row in group_fixtures.sort_values("match_id").itertuples():
        a, b = row.home_team, row.away_team
        ga, gb, outcome, probs = simulate_group_match(a, b, rng)
        stats[a]["played"] += 1
        stats[b]["played"] += 1
        stats[a]["goals_for"] += ga
        stats[a]["goals_against"] += gb
        stats[b]["goals_for"] += gb
        stats[b]["goals_against"] += ga
        if ga > gb:
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
            stats[a]["points"] += 3
        elif gb > ga:
            stats[b]["wins"] += 1
            stats[a]["losses"] += 1
            stats[b]["points"] += 3
        else:
            stats[a]["draws"] += 1
            stats[b]["draws"] += 1
            stats[a]["points"] += 1
            stats[b]["points"] += 1
        match_rows.append({
            "match_id": row.match_id,
            "group": row.group,
            "team_a": a,
            "team_b": b,
            "goals_a": ga,
            "goals_b": gb,
            "outcome": outcome,
            "p_team_a_win": probs["team_a_win"],
            "p_draw": probs["draw"],
            "p_team_b_win": probs["team_b_win"],
        })

    table = pd.DataFrame(stats.values())
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    ranked = []
    for group, df in table.groupby("group"):
        df = df.sort_values(
            ["points", "goal_difference", "goals_for", "wins", "technical_weighted_strength"],
            ascending=False,
        ).reset_index(drop=True)
        df["group_position"] = np.arange(1, len(df) + 1)
        ranked.append(df)
    return pd.DataFrame(match_rows), pd.concat(ranked, ignore_index=True)

example_matches, example_table = simulate_group_stage_once(np.random.default_rng(SEED))
assert len(example_matches) == 72
assert example_table["team"].nunique() == 48
example_table.head()
'''))

    cells.append(md("""
## 7. Classificados, melhores terceiros e chaveamento inicial
"""))

    cells.append(code(r'''
def get_qualified(group_table):
    direct = group_table[group_table["group_position"] <= 2].copy()
    thirds = group_table[group_table["group_position"] == 3].sort_values(
        ["points", "goal_difference", "goals_for", "wins", "technical_weighted_strength"],
        ascending=False,
    ).head(8)
    return direct, thirds, pd.concat([direct, thirds], ignore_index=True)


def assign_best_thirds(best_thirds, slots):
    available = {row.group: row.team for row in best_thirds.itertuples()}
    assignments = {}
    third_slots = slots[(slots["stage"] == "Round of 32") & ((slots["home_slot_type"] == "best_third") | (slots["away_slot_type"] == "best_third"))]
    for row in third_slots.itertuples():
        side = "home" if row.home_slot_type == "best_third" else "away"
        allowed_raw = getattr(row, f"{side}_allowed_third_groups")
        allowed = str(allowed_raw).split(",") if pd.notna(allowed_raw) else []
        chosen = None
        for group in best_thirds["group"].tolist():
            if group in allowed and group in available:
                chosen = group
                break
        if chosen is None and available:
            chosen = next(iter(available))
        assignments[(int(row.match_id), side)] = available.pop(chosen) if chosen else None
    return assignments


def resolve_slot(row, side, group_table, match_results, third_assignments):
    typ = row[f"{side}_slot_type"]
    if typ == "winner_group":
        return group_table[(group_table["group"] == row[f"{side}_group_ref"]) & (group_table["group_position"] == 1)]["team"].iloc[0]
    if typ == "runner_up_group":
        return group_table[(group_table["group"] == row[f"{side}_group_ref"]) & (group_table["group_position"] == 2)]["team"].iloc[0]
    if typ == "best_third":
        return third_assignments.get((int(row["match_id"]), side))
    if typ == "winner_match":
        return match_results[int(row[f"{side}_match_ref"])] ["winner"]
    if typ == "loser_match":
        return match_results[int(row[f"{side}_match_ref"])] ["loser"]
    raise ValueError(f"Tipo de slot desconhecido: {typ}")


def simulate_knockout_once(group_table, rng):
    direct, best_thirds, qualified = get_qualified(group_table)
    third_assignments = assign_best_thirds(best_thirds, knockout_slots)
    match_results = {}
    rows = []
    for _, row in knockout_slots.sort_values("match_id").iterrows():
        home = resolve_slot(row, "home", group_table, match_results, third_assignments)
        away = resolve_slot(row, "away", group_table, match_results, third_assignments)
        probs = match_probabilities(home, away, knockout=True)
        p_home = probs["team_a_win"]
        winner = home if rng.random() < p_home else away
        loser = away if winner == home else home
        most_likely = home if p_home >= 0.5 else away
        match_results[int(row["match_id"])] = {"winner": winner, "loser": loser}
        rows.append({
            "match_id": int(row["match_id"]),
            "stage": row["stage"],
            "slot_home": row["slot_home"],
            "slot_away": row["slot_away"],
            "team_home": home,
            "team_away": away,
            "p_home_win": p_home,
            "p_away_win": 1 - p_home,
            "most_likely_winner": most_likely,
            "simulated_winner": winner,
            "simulated_loser": loser,
        })
    return pd.DataFrame(rows)

direct, thirds, qualified = get_qualified(example_table)
assert len(direct) == 24
assert len(thirds) == 8
assert len(qualified) == 32
example_bracket = simulate_knockout_once(example_table, np.random.default_rng(SEED))
assert len(example_bracket) == 32
assert len(example_bracket[example_bracket["stage"] == "Round of 32"]) == 16
example_bracket[example_bracket["stage"] == "Round of 32"]
'''))

    cells.append(md("""
## 8. Monte Carlo do torneio completo

A célula abaixo roda milhares de torneios simulados e agrega a frequência de avanço por fase.
"""))

    cells.append(code(r'''
def run_monte_carlo(simulations=SIMULATIONS, seed=SEED):
    phase_cols = [
        "group_stage_probability",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    counts = {team: {phase: 0 for phase in phase_cols} for team in features["team"]}
    champion_rows = []
    first_bracket_rows = None
    first_matches = None
    first_table = None

    for sim in range(simulations):
        sim_rng = np.random.default_rng(seed + sim)
        matches, table = simulate_group_stage_once(sim_rng)
        bracket = simulate_knockout_once(table, sim_rng)
        if sim == 0:
            first_bracket_rows = bracket.copy()
            first_matches = matches.copy()
            first_table = table.copy()

        for team in features["team"]:
            counts[team]["group_stage_probability"] += 1

        _, _, qualified = get_qualified(table)
        for team in qualified["team"]:
            counts[team]["round_of_32_probability"] += 1

        stage_to_next_col = {
            "Round of 32": "round_of_16_probability",
            "Round of 16": "quarter_final_probability",
            "Quarter-final": "semi_final_probability",
            "Semi-final": "final_probability",
        }
        for stage, col in stage_to_next_col.items():
            winners = bracket.loc[bracket["stage"] == stage, "simulated_winner"]
            for team in winners:
                counts[team][col] += 1

        final_row = bracket[bracket["stage"] == "Final"].iloc[0]
        champion = final_row["simulated_winner"]
        counts[champion]["champion_probability"] += 1
        champion_rows.append({"simulation": sim + 1, "champion": champion})

    probabilities = pd.DataFrame([
        {"team": team, **{col: value / simulations for col, value in phase_counts.items()}}
        for team, phase_counts in counts.items()
    ])
    probabilities = probabilities.merge(
        features[["team", "group", "technical_rank", "technical_weighted_strength"]],
        on="team",
        how="left",
    )
    probabilities = probabilities.sort_values("champion_probability", ascending=False).reset_index(drop=True)
    champions = pd.DataFrame(champion_rows)
    return probabilities, champions, first_bracket_rows, first_matches, first_table

probabilities, champions, first_bracket, first_matches, first_table = run_monte_carlo(SIMULATIONS, SEED)
assert probabilities["team"].nunique() == 48
assert abs(probabilities["champion_probability"].sum() - 1.0) < 1e-9
probabilities.head(15)
'''))

    cells.append(md("""
## 9. Resultados de uma simulação exemplo — fase de grupos
"""))

    cells.append(code(r'''
for group in sorted(first_table["group"].unique()):
    print(f"Grupo {group}")
    display(first_table[first_table["group"] == group][[
        "group_position", "team", "played", "wins", "draws", "losses", "points", "goal_difference", "goals_for", "goals_against",
    ]])
'''))

    cells.append(md("""
## 10. Possíveis vencedores do chaveamento inicial
"""))

    cells.append(code(r'''
r32 = first_bracket[first_bracket["stage"] == "Round of 32"].copy()
r32[["match_id", "team_home", "team_away", "p_home_win", "p_away_win", "most_likely_winner", "simulated_winner"]]
'''))

    cells.append(md("""
## 11. Gráficos de probabilidades
"""))

    cells.append(code(r'''
plt.style.use("seaborn-v0_8-whitegrid")

top = probabilities.head(15).sort_values("champion_probability")
plt.figure(figsize=(10, 7))
plt.barh(top["team"], top["champion_probability"], color="#2454a6")
plt.xlabel("Probabilidade de título")
plt.title("Copa 2026 — chances de título por Monte Carlo")
plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.savefig(OUTPUTS / "updated_champion_probabilities.png", dpi=180)
plt.show()

phase_cols = [
    "round_of_32_probability",
    "round_of_16_probability",
    "quarter_final_probability",
    "semi_final_probability",
    "final_probability",
    "champion_probability",
]
heat = probabilities.head(20).set_index("team")[phase_cols]
plt.figure(figsize=(11, 8))
if sns is not None:
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="viridis")
else:
    plt.imshow(heat.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Probabilidade")
    plt.xticks(range(len(phase_cols)), phase_cols, rotation=45, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
plt.title("Probabilidade de avanço por fase")
plt.tight_layout()
plt.savefig(OUTPUTS / "updated_phase_progression_heatmap.png", dpi=180)
plt.show()

r32_plot = r32.copy()
r32_plot["label"] = r32_plot["team_home"] + " vs " + r32_plot["team_away"]
r32_plot["favorite_probability"] = r32_plot[["p_home_win", "p_away_win"]].max(axis=1)
r32_plot = r32_plot.sort_values("favorite_probability")
plt.figure(figsize=(12, 9))
plt.barh(r32_plot["label"], r32_plot["favorite_probability"], color="#148f77")
for i, row in enumerate(r32_plot.itertuples()):
    plt.text(min(row.favorite_probability + 0.01, 0.98), i, row.most_likely_winner, va="center", fontsize=9)
plt.xlabel("Probabilidade do favorito no confronto")
plt.title("Round of 32 — possíveis vencedores do chaveamento inicial")
plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.savefig(OUTPUTS / "round_of_32_bracket_probabilities.png", dpi=180)
plt.show()

champion_dist = champions["champion"].value_counts(normalize=True).head(15).sort_values()
plt.figure(figsize=(10, 7))
plt.barh(champion_dist.index, champion_dist.values, color="#8e44ad")
plt.xlabel("Frequência como campeão")
plt.title("Distribuição de campeões nas simulações Monte Carlo")
plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.savefig(OUTPUTS / "monte_carlo_champion_distribution.png", dpi=180)
plt.show()

for file in [
    "updated_champion_probabilities.png",
    "updated_phase_progression_heatmap.png",
    "round_of_32_bracket_probabilities.png",
    "monte_carlo_champion_distribution.png",
]:
    path = OUTPUTS / file
    assert path.exists() and path.stat().st_size > 0, path
'''))

    cells.append(md("""
## 12. Salvar artefatos finais
"""))

    cells.append(code(r'''
prob_path = OUTPUTS / "updated_2026_probabilities.csv"
bracket_path = OUTPUTS / "updated_round_of_32_bracket.csv"
summary_path = OUTPUTS / "updated_2026_simulation_summary.md"
model_card_path = OUTPUTS / "updated_model_card.md"

probabilities.to_csv(prob_path, index=False)
first_bracket.to_csv(bracket_path, index=False)
first_matches.to_csv(OUTPUTS / "updated_group_stage_matches_seed42.csv", index=False)
first_table.to_csv(OUTPUTS / "updated_group_stage_table_seed42.csv", index=False)

summary = "# Simulação probabilística Copa 2026\n\n"
summary += f"Simulações: {SIMULATIONS}\n"
summary += f"Seed: {SEED}\n"
summary += f"Seleções: {features['team'].nunique()}\n"
summary += "Formato: 48 seleções, 12 grupos de 4, 32 no mata-mata.\n\n"
summary += "## Top 10 favoritos\n\n"
summary += probabilities[["team", "champion_probability", "final_probability", "semi_final_probability"]].head(10).to_string(index=False)
summary += "\n\n## Round of 32 da primeira simulação\n\n"
summary += r32[["match_id", "team_home", "team_away", "most_likely_winner", "simulated_winner"]].to_string(index=False)
summary_path.write_text(summary, encoding="utf-8")

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
model_card_path.write_text(model_card, encoding="utf-8")

for path in [prob_path, bracket_path, summary_path, model_card_path]:
    assert path.exists() and path.stat().st_size > 0, path

print(prob_path.resolve())
print(summary_path.resolve())
print(model_card_path.resolve())
probabilities.head(10)
'''))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK)
    print(f"OK wrote {NOTEBOOK}")
