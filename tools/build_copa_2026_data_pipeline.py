from __future__ import annotations

import json
import math
import random
import re
import shutil
import unicodedata
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path.cwd()
DATA = ROOT / "Data"
RAW = DATA / "raw" / "kaggle"
PROCESSED = DATA / "processed"
NOTEBOOK = ROOT / "Copa_2026_Data_Pipeline_e_Simulacao.ipynb"
OUTPUTS = ROOT / "outputs"
for p in [RAW, PROCESSED, OUTPUTS, ROOT / "tools"]:
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "fc26": "justdhia/ea-sports-fc-26-player-ratings",
    "elo": "afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings",
    "fc25": "samandarabdujabbar/ea-sports-fc-25-complete-player-stats-and-analysis",
    "wc2026": "pranishkessi/fifa-world-cup-2026-prediction-simulator",
}

ALIASES = {
    "usa": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "czechia": "czech republic",
    "cote divoire": "cote d ivoire",
    "côte d'ivoire": "cote d ivoire",
    "ivory coast": "cote d ivoire",
    "dr congo": "congo dr",
    "democratic republic of congo": "congo dr",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "korea republic": "south korea",
    "republic of korea": "south korea",
}


def norm_name(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def ensure_kaggle_data() -> list[dict]:
    manifest = []
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError("kagglehub não está instalado no ambiente") from exc

    for label, dataset in DATASETS.items():
        dest = RAW / dataset.replace("/", "__")
        if not dest.exists() or not any(dest.rglob("*.csv")):
            cache_path = Path(kagglehub.dataset_download(dataset))
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(cache_path, dest)
        files = [
            {"path": str(p.relative_to(dest)), "size": p.stat().st_size}
            for p in sorted(dest.rglob("*")) if p.is_file()
        ]
        manifest.append({"label": label, "dataset": dataset, "local_path": str(dest), "files": files})
    (DATA / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_sources():
    wc_dir = RAW / "pranishkessi__fifa-world-cup-2026-prediction-simulator" / "data" / "worldcup_2026"
    groups = pd.read_csv(wc_dir / "worldcup_2026_groups.csv")
    group_fixtures = pd.read_csv(wc_dir / "worldcup_2026_group_fixtures.csv")
    all_fixtures = pd.read_csv(wc_dir / "worldcup_2026_all_fixtures.csv")
    knockout_slots = pd.read_csv(wc_dir / "worldcup_2026_knockout_slots.csv")
    team_probs = pd.read_csv(wc_dir / "model_v5_team_probabilities.csv")
    group_match_probs = pd.read_csv(wc_dir / "model_v5_group_match_probabilities.csv")
    group_finish_probs = pd.read_csv(wc_dir / "model_v5_group_finish_probabilities.csv")
    champion_distribution = pd.read_csv(wc_dir / "model_v5_champion_distribution.csv")

    elo = pd.read_csv(RAW / "afonsofernandescruz__2026-fifa-world-cup-historical-elo-ratings" / "elo_ratings_wc2026.csv")
    fc26_players = pd.read_csv(RAW / "justdhia__ea-sports-fc-26-player-ratings" / "ea_fc26_players.csv")
    fc26_outfield = pd.read_csv(RAW / "justdhia__ea-sports-fc-26-player-ratings" / "ea_fc26_outfield.csv")
    fc26_goalkeepers = pd.read_csv(RAW / "justdhia__ea-sports-fc-26-player-ratings" / "ea_fc26_goalkeepers.csv")
    fc25 = pd.read_csv(RAW / "samandarabdujabbar__ea-sports-fc-25-complete-player-stats-and-analysis" / "ea_sports_fc25_full.csv")
    return locals()


def aggregate_sources(src: dict):
    groups = src["groups"].copy()
    groups["team_key"] = groups["team"].map(norm_name)

    team_probs = src["team_probs"].copy()
    team_probs["team_key"] = team_probs["team"].map(norm_name)

    group_finish = src["group_finish_probs"].copy()
    group_finish["team_key"] = group_finish["team"].map(norm_name)

    champ = src["champion_distribution"].copy()
    champ["team_key"] = champ["team"].map(norm_name)
    champ = champ.rename(columns={"champion_probability": "kaggle_champion_probability"})

    elo = src["elo"].copy()
    elo["team_key"] = elo["country"].map(norm_name)
    latest_year = elo["year"].max()
    elo_latest = (
        elo[elo["year"] == latest_year]
        .sort_values(["team_key", "snapshot_date"])
        .groupby("team_key", as_index=False)
        .tail(1)
        [["team_key", "country", "year", "snapshot_date", "rank", "rating", "rank_avg", "rating_avg", "matches_total", "wins", "losses", "draws", "goals_for", "goals_against", "confederation", "is_host"]]
        .rename(columns={
            "country": "elo_country", "year": "elo_year", "snapshot_date": "elo_snapshot_date", "rank": "elo_rank", "rating": "elo_rating",
            "rank_avg": "elo_rank_avg", "rating_avg": "elo_rating_avg", "matches_total": "elo_matches_total",
            "wins": "elo_wins", "losses": "elo_losses", "draws": "elo_draws", "goals_for": "elo_goals_for", "goals_against": "elo_goals_against",
            "confederation": "elo_confederation", "is_host": "elo_is_host",
        })
    )

    fc26 = src["fc26_players"].copy()
    fc26["team_key"] = fc26["nationality"].map(norm_name)
    fc26_agg = fc26.groupby("team_key").apply(lambda g: pd.Series({
        "fc26_players_count": len(g),
        "fc26_top23_overall_mean": g.nlargest(min(23, len(g)), "overallRating")["overallRating"].mean(),
        "fc26_top11_overall_mean": g.nlargest(min(11, len(g)), "overallRating")["overallRating"].mean(),
        "fc26_max_overall": g["overallRating"].max(),
        "fc26_attack_mean": g.nlargest(min(23, len(g)), "overallRating")["sho"].mean(),
        "fc26_midfield_mean": g.nlargest(min(23, len(g)), "overallRating")["pas"].mean(),
        "fc26_defense_mean": g.nlargest(min(23, len(g)), "overallRating")["def"].mean(),
        "fc26_physical_mean": g.nlargest(min(23, len(g)), "overallRating")["phy"].mean(),
    })).reset_index()

    gk = src["fc26_goalkeepers"].copy()
    gk["team_key"] = gk["nationality"].map(norm_name)
    gk_agg = gk.groupby("team_key").apply(lambda g: pd.Series({
        "fc26_goalkeepers_count": len(g),
        "fc26_top3_gk_mean": g.nlargest(min(3, len(g)), "overallRating")["overallRating"].mean(),
    })).reset_index()

    fc25 = src["fc25"].copy()
    fc25["team_key"] = fc25["Nationality"].map(norm_name)
    fc25_agg = fc25.groupby("team_key").apply(lambda g: pd.Series({
        "fc25_players_count": len(g),
        "fc25_top23_overall_mean": g.nlargest(min(23, len(g)), "Overall")["Overall"].mean(),
        "fc25_top11_overall_mean": g.nlargest(min(11, len(g)), "Overall")["Overall"].mean(),
        "fc25_total_value_eur_top23": g.nlargest(min(23, len(g)), "Overall")["Value_EUR"].sum(),
        "fc25_int_caps_top23": g.nlargest(min(23, len(g)), "Overall")["IntCaps"].fillna(0).sum(),
    })).reset_index()

    master = groups.merge(team_probs.drop(columns=["team", "canonical_team"], errors="ignore"), on="team_key", how="left")
    master = master.merge(group_finish.drop(columns=["team", "group"], errors="ignore"), on="team_key", how="left")
    master = master.merge(champ.drop(columns=["team"], errors="ignore"), on="team_key", how="left")
    master = master.merge(elo_latest, on="team_key", how="left")
    master = master.merge(fc26_agg, on="team_key", how="left")
    master = master.merge(gk_agg, on="team_key", how="left")
    master = master.merge(fc25_agg, on="team_key", how="left")

    # Força combinada: usa probabilidades do dataset de simulação, Elo e elenco EA FC.
    def minmax(s, invert=False):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() == 0 or s.max() == s.min():
            out = pd.Series(np.nan, index=s.index)
        else:
            out = (s - s.min()) / (s.max() - s.min())
        if invert:
            out = 1 - out
        return out

    master["elo_score"] = minmax(master["elo_rating"])
    master["elo_rank_score"] = minmax(master["elo_rank"], invert=True)
    master["squad_fc26_score"] = minmax(master["fc26_top23_overall_mean"])
    master["squad_fc25_score"] = minmax(master["fc25_top23_overall_mean"])
    master["market_fc25_score"] = minmax(np.log1p(master["fc25_total_value_eur_top23"].fillna(0)))
    master["kaggle_prob_score"] = minmax(master["champion_probability"].fillna(master["kaggle_champion_probability"]))

    score_cols = ["elo_score", "elo_rank_score", "squad_fc26_score", "squad_fc25_score", "market_fc25_score", "kaggle_prob_score"]
    weights = np.array([0.25, 0.10, 0.20, 0.10, 0.10, 0.25])
    values = master[score_cols].fillna(master[score_cols].median(numeric_only=True)).fillna(0.5).to_numpy()
    master["combined_strength"] = values.dot(weights) / weights.sum()
    master["combined_rank"] = master["combined_strength"].rank(ascending=False, method="min").astype(int)
    master = master.sort_values(["combined_rank", "group", "team_slot"]).reset_index(drop=True)
    return master


def simulate_group_stage(master: pd.DataFrame, group_match_probs: pd.DataFrame, seed: int = 42):
    rng = random.Random(seed)
    strength = dict(zip(master["team"], master["combined_strength"]))
    rows = []
    for _, m in group_match_probs.sort_values("match_id").iterrows():
        teams = [m["team_a"], m["team_b"]]
        probs = [float(m["p_team_a_win"]), float(m["p_draw"]), float(m["p_team_b_win"])]
        outcome = rng.choices(["team_a_win", "draw", "team_b_win"], weights=probs, k=1)[0]
        sa, sb = strength.get(teams[0], 0.5), strength.get(teams[1], 0.5)
        lam_a = max(0.2, 0.85 + 1.55 * sa - 0.75 * sb)
        lam_b = max(0.2, 0.85 + 1.55 * sb - 0.75 * sa)
        ga = int(np.random.default_rng(seed + int(m["match_id"])).poisson(lam_a))
        gb = int(np.random.default_rng(seed + 1000 + int(m["match_id"])).poisson(lam_b))
        if outcome == "team_a_win" and ga <= gb:
            ga = gb + 1
        elif outcome == "team_b_win" and gb <= ga:
            gb = ga + 1
        elif outcome == "draw":
            g = int(round((ga + gb) / 2))
            ga = gb = g
        winner = teams[0] if ga > gb else teams[1] if gb > ga else "Draw"
        rows.append({**m.to_dict(), "sim_goals_team_a": ga, "sim_goals_team_b": gb, "sim_result": outcome, "sim_winner": winner})

    matches = pd.DataFrame(rows)
    table_rows = []
    for group, teams_df in master.sort_values("team_slot").groupby("group"):
        stats = {t: {"group": group, "team": t, "played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0, "points": 0, "combined_strength": strength.get(t, 0.5)} for t in teams_df["team"]}
        gm = matches[matches["group"] == group]
        for _, m in gm.iterrows():
            a, b = m["team_a"], m["team_b"]
            ga, gb = int(m["sim_goals_team_a"]), int(m["sim_goals_team_b"])
            for t in [a, b]:
                stats[t]["played"] += 1
            stats[a]["goals_for"] += ga; stats[a]["goals_against"] += gb
            stats[b]["goals_for"] += gb; stats[b]["goals_against"] += ga
            if ga > gb:
                stats[a]["wins"] += 1; stats[b]["losses"] += 1; stats[a]["points"] += 3
            elif gb > ga:
                stats[b]["wins"] += 1; stats[a]["losses"] += 1; stats[b]["points"] += 3
            else:
                stats[a]["draws"] += 1; stats[b]["draws"] += 1; stats[a]["points"] += 1; stats[b]["points"] += 1
        df = pd.DataFrame(stats.values())
        df["goal_difference"] = df["goals_for"] - df["goals_against"]
        df = df.sort_values(["points", "goal_difference", "goals_for", "wins", "combined_strength"], ascending=False).reset_index(drop=True)
        df["group_position"] = np.arange(1, len(df) + 1)
        table_rows.append(df)
    group_tables = pd.concat(table_rows, ignore_index=True)
    return matches, group_tables


def assign_best_thirds_to_slots(best_thirds: pd.DataFrame, slots: pd.DataFrame):
    available = {r["group"]: r["team"] for _, r in best_thirds.iterrows()}
    assignments = {}
    third_slot_rows = slots[(slots["stage"] == "Round of 32") & ((slots["home_slot_type"] == "best_third") | (slots["away_slot_type"] == "best_third"))]
    for _, row in third_slot_rows.iterrows():
        side = "home" if row["home_slot_type"] == "best_third" else "away"
        allowed = str(row[f"{side}_allowed_third_groups"]).split(",")
        chosen_group = None
        # Prefer melhor terceiro com grupo permitido ainda disponível.
        for group in best_thirds["group"].tolist():
            if group in allowed and group in available:
                chosen_group = group
                break
        if chosen_group is None and available:
            chosen_group = next(iter(available))
        assignments[(int(row["match_id"]), side)] = available.pop(chosen_group) if chosen_group else None
    return assignments


def resolve_slot(row, side: str, group_tables: pd.DataFrame, match_results: dict, third_assignments: dict):
    typ = row[f"{side}_slot_type"]
    if typ == "winner_group":
        return group_tables[(group_tables["group"] == row[f"{side}_group_ref"]) & (group_tables["group_position"] == 1)]["team"].iloc[0]
    if typ == "runner_up_group":
        return group_tables[(group_tables["group"] == row[f"{side}_group_ref"]) & (group_tables["group_position"] == 2)]["team"].iloc[0]
    if typ == "best_third":
        return third_assignments.get((int(row["match_id"]), side))
    if typ == "winner_match":
        return match_results[int(row[f"{side}_match_ref"])]["winner"]
    if typ == "loser_match":
        return match_results[int(row[f"{side}_match_ref"])]["loser"]
    return None


def p_win(team_a, team_b, strength):
    diff = strength.get(team_a, 0.5) - strength.get(team_b, 0.5)
    return logistic(4.5 * diff)


def simulate_knockout(master: pd.DataFrame, group_tables: pd.DataFrame, knockout_slots: pd.DataFrame, seed: int = 42):
    rng = random.Random(seed + 2026)
    strength = dict(zip(master["team"], master["combined_strength"]))
    thirds = group_tables[group_tables["group_position"] == 3].sort_values(["points", "goal_difference", "goals_for", "wins", "combined_strength"], ascending=False).head(8)
    third_assignments = assign_best_thirds_to_slots(thirds, knockout_slots)
    match_results = {}
    rows = []
    for _, row in knockout_slots.sort_values("match_id").iterrows():
        a = resolve_slot(row, "home", group_tables, match_results, third_assignments)
        b = resolve_slot(row, "away", group_tables, match_results, third_assignments)
        prob_a = p_win(a, b, strength) if a and b else np.nan
        winner = a if rng.random() < prob_a else b
        loser = b if winner == a else a
        match_results[int(row["match_id"])] = {"winner": winner, "loser": loser}
        rows.append({
            "match_id": int(row["match_id"]), "stage": row["stage"], "slot_home": row["slot_home"], "slot_away": row["slot_away"],
            "team_home": a, "team_away": b, "p_home_win": prob_a, "winner": winner, "loser": loser,
            "date_utc": row.get("date_utc"), "stadium": row.get("stadium"), "city": row.get("city"),
        })
    return pd.DataFrame(rows), thirds


def build_outputs(seed: int = 42):
    manifest = ensure_kaggle_data()
    src = load_sources()
    master = aggregate_sources(src)
    group_matches, group_tables = simulate_group_stage(master, src["group_match_probs"], seed=seed)
    bracket, best_thirds = simulate_knockout(master, group_tables, src["knockout_slots"], seed=seed)

    master.to_csv(PROCESSED / "copa_2026_master_team_dataset.csv", index=False)
    group_matches.to_csv(PROCESSED / "copa_2026_group_stage_simulated_matches.csv", index=False)
    group_tables.to_csv(PROCESSED / "copa_2026_group_stage_tables.csv", index=False)
    best_thirds.to_csv(PROCESSED / "copa_2026_best_thirds.csv", index=False)
    bracket.to_csv(PROCESSED / "copa_2026_resolved_bracket_simulation.csv", index=False)

    with pd.ExcelWriter(PROCESSED / "copa_2026_base_combinada.xlsx", engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="base_combinada", index=False)
        group_tables.to_excel(writer, sheet_name="fase_grupos_tabelas", index=False)
        group_matches.to_excel(writer, sheet_name="fase_grupos_jogos", index=False)
        best_thirds.to_excel(writer, sheet_name="melhores_terceiros", index=False)
        bracket.to_excel(writer, sheet_name="chaveamento", index=False)
        pd.DataFrame(manifest).to_excel(writer, sheet_name="fontes", index=False)

    summary = {
        "teams": int(master["team"].nunique()),
        "groups": int(master["group"].nunique()),
        "teams_per_group_min": int(master.groupby("group")["team"].nunique().min()),
        "teams_per_group_max": int(master.groupby("group")["team"].nunique().max()),
        "group_matches": int(len(group_matches)),
        "qualified_round_of_32": int((group_tables["group_position"] <= 2).sum() + len(best_thirds)),
        "knockout_matches": int(len(bracket)),
        "simulated_champion": str(bracket[bracket["stage"] == "Final"]["winner"].iloc[0]),
    }
    (PROCESSED / "simulation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest, src, master, group_matches, group_tables, best_thirds, bracket, summary


def make_notebook():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

    md("""
# Copa 2026 — ingestão de dados Kaggle, base combinada e simulação

Este notebook recria a pasta `Data/`, carrega os datasets Kaggle solicitados, combina as fontes por seleção e gera:

- base processada com 48 seleções;
- tabelas de fase de grupos simulada;
- 8 melhores terceiros;
- chaveamento resolvido a partir do arquivo de slots da Copa 2026;
- arquivo Excel com guias separadas.

A simulação é probabilística e usa seed fixa (`SEED = 42`) para reprodutibilidade.
""")
    code("""
from pathlib import Path
import json
import pandas as pd

SEED = 42
ROOT = Path.cwd()
DATA = ROOT / "Data"
PROCESSED = DATA / "processed"
print("Projeto:", ROOT)
print("Pasta Data:", DATA)
""")
    md("""## 1. Fontes baixadas do Kaggle""")
    code("""
manifest = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
fontes = pd.DataFrame([
    {"label": item.get("label"), "dataset": item.get("dataset"), "local_path": item.get("local_path"), "arquivos": len(item.get("files", []))}
    for item in manifest
])
display(fontes)
""")
    md("""## 2. Amostras dos datasets brutos""")
    code("""
raw_files = {
    "EA FC 26 jogadores": DATA / "raw/kaggle/justdhia__ea-sports-fc-26-player-ratings/ea_fc26_players.csv",
    "Elo histórico": DATA / "raw/kaggle/afonsofernandescruz__2026-fifa-world-cup-historical-elo-ratings/elo_ratings_wc2026.csv",
    "EA FC 25 stats": DATA / "raw/kaggle/samandarabdujabbar__ea-sports-fc-25-complete-player-stats-and-analysis/ea_sports_fc25_full.csv",
    "Grupos 2026": DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_groups.csv",
    "Probabilidades jogos de grupo": DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/model_v5_group_match_probabilities.csv",
    "Slots do chaveamento": DATA / "raw/kaggle/pranishkessi__fifa-world-cup-2026-prediction-simulator/data/worldcup_2026/worldcup_2026_knockout_slots.csv",
}
for nome, path in raw_files.items():
    df = pd.read_csv(path, nrows=5)
    print()
    print(f"=== {nome} ===")
    print(path)
    display(df)
""" )
    md("""## 3. Base combinada gerada""")
    code("""
base = pd.read_csv(PROCESSED / "copa_2026_master_team_dataset.csv")
print("Dimensão da base combinada:", base.shape)
print("Seleções:", base["team"].nunique())
print("Grupos:", base["group"].nunique())
display(base[[
    "group", "team_slot", "team", "combined_rank", "combined_strength",
    "champion_probability", "elo_rating", "elo_rank",
    "fc26_top23_overall_mean", "fc25_top23_overall_mean", "fc25_total_value_eur_top23"
]].sort_values("combined_rank").head(20))
""")
    md("""## 4. Validação do formato: 48 seleções, 12 grupos, 4 por grupo""")
    code("""
validacao_grupos = base.groupby("group").agg(selecoes=("team", "nunique"), times=("team", lambda s: ", ".join(s))).reset_index()
display(validacao_grupos)
assert base["team"].nunique() == 48
assert base["group"].nunique() == 12
assert validacao_grupos["selecoes"].eq(4).all()
print("Validação OK: 48 seleções, 12 grupos e 4 seleções por grupo.")
""")
    md("""## 5. Resultados simulados da fase de grupos""")
    code("""
group_matches = pd.read_csv(PROCESSED / "copa_2026_group_stage_simulated_matches.csv")
group_tables = pd.read_csv(PROCESSED / "copa_2026_group_stage_tables.csv")
print("Jogos de grupo simulados:", len(group_matches))
display(group_matches[["match_id", "group", "team_a", "team_b", "p_team_a_win", "p_draw", "p_team_b_win", "sim_goals_team_a", "sim_goals_team_b", "sim_winner"]].head(20))

for group in sorted(group_tables["group"].unique()):
    print()
    print(f"Grupo {group}")
    display(group_tables[group_tables["group"] == group][["group_position", "team", "played", "wins", "draws", "losses", "goals_for", "goals_against", "goal_difference", "points"]])
""")
    md("""## 6. Classificados: 2 primeiros + 8 melhores terceiros""")
    code("""
best_thirds = pd.read_csv(PROCESSED / "copa_2026_best_thirds.csv")
direct = group_tables[group_tables["group_position"] <= 2].copy()
qualified = pd.concat([direct, best_thirds], ignore_index=True)
print("Classificados diretos:", len(direct))
print("Melhores terceiros:", len(best_thirds))
print("Total para Round of 32:", len(qualified))
display(best_thirds[["group", "group_position", "team", "points", "goal_difference", "goals_for", "combined_strength"]])
assert len(qualified) == 32
""")
    md("""## 7. Chaveamento 2026 resolvido""")
    code("""
bracket = pd.read_csv(PROCESSED / "copa_2026_resolved_bracket_simulation.csv")
display(bracket[["match_id", "stage", "slot_home", "slot_away", "team_home", "team_away", "p_home_win", "winner"]])
campeao = bracket.loc[bracket["stage"].eq("Final"), "winner"].iloc[0]
print("Campeão simulado nesta seed:", campeao)
""")
    md("""## 8. Arquivos salvos""")
    code("""
arquivos = [
    PROCESSED / "copa_2026_master_team_dataset.csv",
    PROCESSED / "copa_2026_group_stage_simulated_matches.csv",
    PROCESSED / "copa_2026_group_stage_tables.csv",
    PROCESSED / "copa_2026_best_thirds.csv",
    PROCESSED / "copa_2026_resolved_bracket_simulation.csv",
    PROCESSED / "copa_2026_base_combinada.xlsx",
    PROCESSED / "simulation_summary.json",
]
for path in arquivos:
    print(path.resolve(), "->", path.stat().st_size, "bytes")
""")

    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    })
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    manifest, src, master, group_matches, group_tables, best_thirds, bracket, summary = build_outputs(seed=42)
    make_notebook()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Notebook:", NOTEBOOK.resolve())
    print("Excel:", (PROCESSED / "copa_2026_base_combinada.xlsx").resolve())
