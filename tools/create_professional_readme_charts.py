"""Generate publication-quality charts for the Copa 2026 README.

The charts intentionally avoid notebook-only state and are built only from
versioned CSV outputs, making them reproducible in CI or on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PROB_PATH = OUTPUT_DIR / "updated_2026_probabilities.csv"
BRACKET_PATH = OUTPUT_DIR / "updated_round_of_32_bracket.csv"
SUMMARY_PATH = OUTPUT_DIR / "updated_2026_simulation_summary.md"

# Paleta visual solicitada para os gráficos do README.
# Alto valor -> bege/off-white; baixo valor -> azul quase preto.
BG = "#06111A"
PANEL = "#06111A"
PANEL_2 = "#0F2D44"
TEXT = "#F3E5CD"
MUTED = "#D4A84B"
GRID = "#0F2D44"
HIGH = "#F3E5CD"
GOLD = "#D4A84B"
OLIVE = "#8B7E35"
MOSS = "#315C3A"
PETROL = "#0F2D44"
LOW = "#06111A"
CHART_CMAP = LinearSegmentedColormap.from_list(
    "copa_requested",
    [LOW, PETROL, MOSS, OLIVE, GOLD, HIGH],
)

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
    }
)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def save(fig: Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(path.relative_to(ROOT))


def add_footer(fig: Figure) -> None:
    fig.text(
        0.99,
        0.015,
        "Fonte: datasets Kaggle + simulação Monte Carlo | Seed 42 | Copa 2026",
        ha="right",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )


def title_bar(ax: Axes, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0,
        1.13,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=TEXT,
        fontsize=19,
        fontweight="bold",
    )
    if subtitle:
        ax.text(0, 1.065, subtitle, transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=11)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    probs = pd.read_csv(PROB_PATH).sort_values("champion_probability", ascending=False).reset_index(drop=True)
    bracket = pd.read_csv(BRACKET_PATH)
    return probs, bracket


def make_title_ranking(probs: pd.DataFrame) -> None:
    top = probs.head(14).sort_values("champion_probability", ascending=True)
    values = top["champion_probability"] * 100
    norm = plt.Normalize(values.min(), values.max())
    colors = CHART_CMAP(norm(values))

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top["team"], values, color=colors, edgecolor="none", height=0.68)

    for bar, value in zip(bars, top["champion_probability"] * 100):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", ha="left", fontsize=11, color=TEXT, fontweight="bold")

    title_bar(
        ax,
        "Probabilidade de título — Top 14 seleções",
        "Percentual de campeonatos vencidos no conjunto de 100.000 simulações.",
    )
    ax.set_xlabel("Probabilidade de ser campeão (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(top["champion_probability"] * 100) * 1.22)
    ax.grid(axis="x", linestyle="--")
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_footer(fig)
    save(fig, "professional_champion_ranking.png")


def make_phase_heatmap(probs: pd.DataFrame) -> None:
    phase_cols = [
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    labels = ["R32", "Oitavas", "Quartas", "Semis", "Final", "Título"]
    top = probs.head(18).copy()
    matrix = top.set_index("team")[phase_cols] * 100
    matrix.columns = labels

    cmap = CHART_CMAP

    fig, ax = plt.subplots(figsize=(12.5, 9))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color=BG, linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = BG if values[row, col] >= 55 else TEXT
            ax.text(col, row, f"{values[row, col]:.0f}", ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(image, ax=ax, shrink=0.78)
    cbar.set_label("Probabilidade (%)", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=MUTED)

    title_bar(
        ax,
        "Mapa de progressão por fase",
        "Probabilidade percentual de cada seleção alcançar as fases decisivas.",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    add_footer(fig)
    save(fig, "professional_phase_heatmap.png")


def make_funnel(probs: pd.DataFrame) -> None:
    teams = probs.head(7)["team"].tolist()
    phase_cols = [
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    labels = ["R32", "Oitavas", "Quartas", "Semis", "Final", "Título"]
    palette = [HIGH, GOLD, OLIVE, MOSS, PETROL, "#A98F45", "#6E7938"]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    x = np.arange(len(labels))
    for idx, team in enumerate(teams):
        row = probs.loc[probs["team"] == team, phase_cols].iloc[0].to_numpy(dtype=float) * 100
        ax.plot(x, row, marker="o", linewidth=3.0, markersize=7, color=palette[idx], label=team)
        ax.scatter(x[-1], row[-1], s=95, color=palette[idx], edgecolors=BG, linewidth=1.5, zorder=5)

    title_bar(
        ax,
        "Funil de sobrevivência dos favoritos",
        "Queda de probabilidade conforme o torneio avança para fases mais difíceis.",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Probabilidade de alcançar a fase (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_footer(fig)
    save(fig, "professional_favorites_funnel.png")


def make_bracket_strength(bracket: pd.DataFrame) -> None:
    data = bracket.copy().head(16)
    data["favorite_probability"] = data[["p_home_win", "p_away_win"]].max(axis=1) * 100
    data["underdog_probability"] = 100 - data["favorite_probability"]
    data["label"] = data.apply(
        lambda r: f"{int(r['match_id'])}: {r['team_home']} × {r['team_away']}", axis=1
    )
    data = data.sort_values("favorite_probability", ascending=True)

    fig, ax = plt.subplots(figsize=(13.5, 9))
    y = np.arange(len(data))
    norm = plt.Normalize(50, 100)
    fav_colors = CHART_CMAP(norm(data["favorite_probability"]))
    ax.barh(y, data["favorite_probability"], color=fav_colors, edgecolor="none", height=0.62, label="Favorito")
    ax.barh(y, data["underdog_probability"], left=data["favorite_probability"], color=PANEL_2, edgecolor="none", height=0.62, label="Adversário")

    for yi, fav, winner in zip(y, data["favorite_probability"], data["most_likely_winner"]):
        label_color = BG if fav >= 68 else TEXT
        ax.text(fav - 2, float(yi), f"{winner} {fav:.0f}%", ha="right", va="center", color=label_color, fontsize=9.5, fontweight="bold")

    ax.set_yticks(y, [textwrap.shorten(v, width=44, placeholder="…") for v in data["label"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilidade estimada no confronto (%)")
    ax.set_ylabel("")
    title_bar(
        ax,
        "Round of 32 — equilíbrio dos confrontos",
        "Quanto maior a faixa verde, maior a vantagem probabilística do favorito.",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--")
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_footer(fig)
    save(fig, "professional_round32_balance.png")


def main() -> None:
    probs, bracket = load_data()
    make_title_ranking(probs)
    make_phase_heatmap(probs)
    make_funnel(probs)
    make_bracket_strength(bracket)


if __name__ == "__main__":
    main()
