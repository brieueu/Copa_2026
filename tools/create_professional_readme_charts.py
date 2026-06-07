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
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PROB_PATH = OUTPUT_DIR / "updated_2026_probabilities.csv"
BRACKET_PATH = OUTPUT_DIR / "updated_round_of_32_bracket.csv"

# Paleta visual solicitada para os gráficos do README.
# Alto valor -> bege/off-white; baixo valor -> azul quase preto.
BG = "#06111A"
PANEL = "#0A1824"
PANEL_SOFT = "#102536"
TEXT = "#F3E5CD"
MUTED = "#D4A84B"
GRID = "#18384D"
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
        "axes.facecolor": "none",
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
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
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(path.relative_to(ROOT))


def add_background(fig: Figure) -> None:
    """Add a subtle vertical gradient and a rounded chart card."""
    bg_ax = fig.add_axes((0, 0, 1, 1), zorder=-2)
    grad = np.linspace(0, 1, 500).reshape(-1, 1)
    bg_ax.imshow(
        grad,
        cmap=LinearSegmentedColormap.from_list("bg", [BG, "#092235", BG]),
        aspect="auto",
        extent=(0, 1, 0, 1),
        origin="lower",
    )
    bg_ax.axis("off")
    card = FancyBboxPatch(
        (0.018, 0.032),
        0.964,
        0.925,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        transform=fig.transFigure,
        facecolor=to_rgba(PANEL, 0.86),
        edgecolor=to_rgba(GOLD, 0.28),
        linewidth=1.2,
        zorder=-1,
    )
    fig.patches.append(card)


def add_footer(fig: Figure) -> None:
    fig.text(
        0.982,
        0.02,
        "Fonte: datasets Kaggle + simulação Monte Carlo | Seed 42 | Copa 2026",
        ha="right",
        va="bottom",
        fontsize=9,
        color=to_rgba(MUTED, 0.88),
    )


def title_bar(ax: Axes, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0,
        1.16,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=TEXT,
        fontsize=21,
        fontweight="bold",
    )
    if subtitle:
        ax.text(
            0,
            1.09,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=11.5,
        )


def style_axes(ax: Axes, xgrid: bool = True, ygrid: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xgrid:
        ax.grid(axis="x", linestyle="--", linewidth=0.8)
    else:
        ax.grid(axis="x", visible=False)
    if ygrid:
        ax.grid(axis="y", linestyle="--", linewidth=0.8)
    else:
        ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    probs = pd.read_csv(PROB_PATH).sort_values("champion_probability", ascending=False).reset_index(drop=True)
    bracket = pd.read_csv(BRACKET_PATH)
    return probs, bracket


def make_title_ranking(probs: pd.DataFrame) -> None:
    top = probs.head(14).sort_values("champion_probability", ascending=True)
    values = top["champion_probability"] * 100
    norm = Normalize(0, max(values.max(), 1))
    colors = CHART_CMAP(norm(values))

    fig, ax = plt.subplots(figsize=(13.2, 8.3))
    add_background(fig)
    fig.subplots_adjust(left=0.22, right=0.91, top=0.78, bottom=0.16)

    bars = ax.barh(top["team"], values, color=colors, edgecolor=to_rgba(HIGH, 0.12), height=0.68)
    ax.barh(top["team"], values, color="none", edgecolor=to_rgba(HIGH, 0.24), height=0.68, linewidth=0.8)

    top_three = list(probs.head(3)["team"])
    medals = {top_three[0]: "1º", top_three[1]: "2º", top_three[2]: "3º"}
    for bar, team, value in zip(bars, top["team"], values):
        label = f"{value:.1f}%".replace(".", ",")
        ax.text(value + 0.38, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=11, color=TEXT, fontweight="bold")
        if team in medals:
            ax.text(0.25, bar.get_y() + bar.get_height() / 2, medals[team], va="center", ha="left", fontsize=9.5, color=BG, fontweight="bold", bbox=dict(boxstyle="round,pad=0.28", facecolor=GOLD, edgecolor="none"))

    title_bar(
        ax,
        "Probabilidade de título — Top 14 seleções",
        "Ranking dos campeões mais frequentes em 100.000 simulações; barras mais claras indicam maior chance.",
    )
    ax.set_xlabel("Probabilidade de ser campeão (%)", labelpad=10)
    ax.set_ylabel("")
    ax.set_xlim(0, values.max() * 1.25)
    style_axes(ax, xgrid=True)
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

    fig, ax = plt.subplots(figsize=(13.2, 9.6))
    add_background(fig)
    fig.subplots_adjust(left=0.19, right=0.86, top=0.80, bottom=0.12)
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=CHART_CMAP, vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color=BG, linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = BG if values[row, col] >= 58 else TEXT
            ax.text(col, row, f"{values[row, col]:.0f}%", ha="center", va="center", fontsize=9.2, fontweight="bold", color=color)

    cbar = fig.colorbar(image, ax=ax, shrink=0.76, pad=0.035)
    cbar.set_label("Probabilidade (%)", color=TEXT, labelpad=10)
    cbar.outline.set_edgecolor(GRID)  # type: ignore[attr-defined]
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=MUTED)

    title_bar(
        ax,
        "Mapa de progressão por fase",
        "Leitura horizontal: chance de cada seleção sobreviver da primeira fase até o título.",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, pad=10, labelsize=10.5)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
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
    palette = [HIGH, GOLD, OLIVE, MOSS, "#6E7938", PETROL, "#A98F45"]

    fig, ax = plt.subplots(figsize=(13.6, 7.8))
    add_background(fig)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.77, bottom=0.24)
    x = np.arange(len(labels))
    for idx, team in enumerate(teams):
        row = probs.loc[probs["team"] == team, phase_cols].iloc[0].to_numpy(dtype=float) * 100
        alpha = 1.0 if idx < 3 else 0.82
        ax.plot(x, row, marker="o", linewidth=3.2 if idx < 3 else 2.45, markersize=7.5, color=palette[idx], alpha=alpha, label=team)
        ax.scatter(x[-1], row[-1], s=130 if idx < 3 else 90, color=palette[idx], edgecolors=BG, linewidth=1.6, zorder=5)
        ax.text(x[-1] + 0.09, row[-1], f"{row[-1]:.1f}%".replace(".", ","), va="center", ha="left", fontsize=9.5, color=palette[idx], fontweight="bold")

    title_bar(
        ax,
        "Funil de sobrevivência dos favoritos",
        "Comparação direta do risco acumulado: favoritos fortes mantêm linhas mais altas até o título.",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Probabilidade de alcançar a fase (%)", labelpad=10)
    ax.set_xlim(-0.12, len(labels) - 0.35)
    ax.set_ylim(0, 105)
    style_axes(ax, xgrid=False, ygrid=True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=True, fontsize=10)
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

    fig, ax = plt.subplots(figsize=(14.2, 9.2))
    add_background(fig)
    fig.subplots_adjust(left=0.29, right=0.94, top=0.80, bottom=0.14)
    y = np.arange(len(data))
    norm = Normalize(50, 100)
    fav_colors = CHART_CMAP(norm(data["favorite_probability"]))
    ax.barh(y, data["favorite_probability"], color=fav_colors, edgecolor=to_rgba(HIGH, 0.18), height=0.64, label="Favorito")
    ax.barh(y, data["underdog_probability"], left=data["favorite_probability"], color=to_rgba(PETROL, 0.82), edgecolor=to_rgba(HIGH, 0.08), height=0.64, label="Adversário")
    ax.axvline(50, color=to_rgba(GOLD, 0.55), linewidth=1.3, linestyle="--")

    for yi, fav, winner in zip(y, data["favorite_probability"], data["most_likely_winner"]):
        label_color = BG if fav >= 68 else TEXT
        ax.text(fav - 2.0, float(yi), f"{winner} {fav:.0f}%", ha="right", va="center", color=label_color, fontsize=9.4, fontweight="bold")
        ax.text(98.0, float(yi), f"{100 - fav:.0f}%", ha="right", va="center", color=to_rgba(TEXT, 0.75), fontsize=8.8)

    ax.set_yticks(y, [textwrap.shorten(v, width=46, placeholder="…") for v in data["label"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilidade estimada no confronto (%)", labelpad=10)
    ax.set_ylabel("")
    title_bar(
        ax,
        "Round of 32 — equilíbrio dos confrontos",
        "Linha tracejada marca 50%; barras mais claras indicam favoritismo mais forte.",
    )
    ax.legend(loc="lower right", frameon=True)
    style_axes(ax, xgrid=True)
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
