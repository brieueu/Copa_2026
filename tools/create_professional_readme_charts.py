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
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PROB_PATH = OUTPUT_DIR / "updated_2026_probabilities.csv"
BRACKET_PATH = OUTPUT_DIR / "updated_round_of_32_bracket.csv"
SUMMARY_PATH = OUTPUT_DIR / "updated_2026_simulation_summary.md"

BG = "#0B1020"
PANEL = "#111827"
PANEL_2 = "#172033"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
GRID = "#263143"
GOLD = "#F8C14A"
BLUE = "#38BDF8"
GREEN = "#34D399"
RED = "#FB7185"
PURPLE = "#A78BFA"
ORANGE = "#FDBA74"

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
    colors = [GOLD if t in {"Spain", "France", "Germany"} else BLUE for t in top["team"]]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top["team"], top["champion_probability"] * 100, color=colors, edgecolor="none", height=0.68)

    for bar, value in zip(bars, top["champion_probability"] * 100):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", ha="left", fontsize=11, color=TEXT, fontweight="bold")

    title_bar(
        ax,
        "Probabilidade de título — Top 14 seleções",
        "Percentual de campeonatos vencidos no conjunto de 10.000 simulações.",
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

    cmap = LinearSegmentedColormap.from_list("copa", ["#111827", "#1E3A8A", "#0891B2", "#34D399", "#F8C14A"])

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
    palette = [GOLD, BLUE, GREEN, RED, PURPLE, ORANGE, "#22D3EE"]

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
    ax.barh(y, data["favorite_probability"], color=GREEN, edgecolor="none", height=0.62, label="Favorito")
    ax.barh(y, data["underdog_probability"], left=data["favorite_probability"], color=PANEL_2, edgecolor="none", height=0.62, label="Adversário")

    for yi, fav, winner in zip(y, data["favorite_probability"], data["most_likely_winner"]):
        ax.text(fav - 2, float(yi), f"{winner} {fav:.0f}%", ha="right", va="center", color=BG, fontsize=9.5, fontweight="bold")

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


def make_executive_dashboard(probs: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.9, 2.6], width_ratios=[1, 1, 1, 1], hspace=0.34, wspace=0.22)

    cards = [
        ("10.000", "simulações", BLUE),
        ("48", "seleções", GREEN),
        ("22,34%", "maior chance de título", GOLD),
        ("12", "grupos", PURPLE),
    ]
    for i, (value, label, color) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()
        box = FancyBboxPatch(
            (0.02, 0.12),
            0.96,
            0.76,
            boxstyle="round,pad=0.025,rounding_size=0.06",
            facecolor=PANEL,
            edgecolor=GRID,
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(0.08, 0.58, value, transform=ax.transAxes, fontsize=27, fontweight="bold", color=color, ha="left", va="center")
        ax.text(0.08, 0.34, label.upper(), transform=ax.transAxes, fontsize=10, color=MUTED, ha="left", va="center", fontweight="bold")

    ax1 = fig.add_subplot(gs[1, :2])
    top = probs.head(8).sort_values("champion_probability")
    ax1.barh(top["team"], top["champion_probability"] * 100, color=[GOLD if x in {"Spain", "France"} else BLUE for x in top["team"]], height=0.65)
    for y, v in enumerate(top["champion_probability"] * 100):
        ax1.text(v + 0.35, y, f"{v:.1f}%", va="center", fontsize=10, color=TEXT, fontweight="bold")
    title_bar(ax1, "Ranking de título", "Top 8 por probabilidade de campeão")
    ax1.set_xlabel("Probabilidade (%)")
    ax1.set_ylabel("")
    ax1.set_xlim(0, 25)
    ax1.grid(axis="x", linestyle="--")
    ax1.grid(axis="y", visible=False)
    for s in ax1.spines.values():
        s.set_visible(False)

    ax2 = fig.add_subplot(gs[1, 2:])
    phases = ["round_of_16_probability", "quarter_final_probability", "semi_final_probability", "final_probability", "champion_probability"]
    labels = ["Oitavas", "Quartas", "Semis", "Final", "Título"]
    for color, team in zip([GOLD, BLUE, GREEN, RED], probs.head(4)["team"]):
        values = probs.loc[probs["team"] == team, phases].iloc[0].to_numpy(dtype=float) * 100
        ax2.plot(labels, values, marker="o", linewidth=3, markersize=7, label=team, color=color)
    title_bar(ax2, "Trajetória dos favoritos", "Probabilidade acumulada por fase")
    ax2.set_ylabel("Probabilidade (%)")
    ax2.set_ylim(0, 90)
    ax2.grid(axis="y", linestyle="--")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right")
    for s in ax2.spines.values():
        s.set_visible(False)

    fig.suptitle("Copa do Mundo 2026 — Simulação probabilística", x=0.03, y=0.99, ha="left", fontsize=24, fontweight="bold", color=TEXT)
    fig.text(0.03, 0.94, "Visão executiva dos resultados do modelo Monte Carlo", ha="left", fontsize=12, color=MUTED)
    add_footer(fig)
    save(fig, "professional_executive_dashboard.png")


def main() -> None:
    probs, bracket = load_data()
    make_executive_dashboard(probs)
    make_title_ranking(probs)
    make_phase_heatmap(probs)
    make_funnel(probs)
    make_bracket_strength(bracket)


if __name__ == "__main__":
    main()
