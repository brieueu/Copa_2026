from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PROB_PATH = OUTPUTS / "updated_2026_probabilities.csv"
GIF_PATH = OUTPUTS / "professional_champion_probability_animation.gif"

BG = "#0B1020"
PANEL = "#111827"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
GRID = "#263143"
GOLD = "#F8C14A"
BLUE = "#38BDF8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "font.family": "DejaVu Sans",
    "grid.color": GRID,
})


def make_frame(data: pd.DataFrame, progress: float, frame_index: int, total_frames: int) -> Image.Image:
    top = data.head(12).sort_values("champion_probability", ascending=True).copy()
    values = top["champion_probability"].to_numpy() * 100 * progress
    colors = [GOLD if team in {"Spain", "France", "Germany"} else BLUE for team in top["team"]]

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    bars = ax.barh(top["team"], values, color=colors, edgecolor="none", height=0.66)

    final_values = top["champion_probability"].to_numpy() * 100
    for bar, shown, final in zip(bars, values, final_values):
        if shown >= 0.35:
            ax.text(
                shown + 0.28,
                bar.get_y() + bar.get_height() / 2,
                f"{final:.2f}%",
                va="center",
                ha="left",
                fontsize=10.5,
                color=TEXT,
                fontweight="bold",
            )

    ax.text(0, 1.13, "Copa 2026 — corrida das probabilidades de título", transform=ax.transAxes, ha="left", va="bottom", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(0, 1.065, "Resultado consolidado da simulação Monte Carlo com 100.000 torneios", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=MUTED)
    ax.text(0.99, 1.065, f"frame {frame_index + 1}/{total_frames}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=MUTED)

    ax.set_xlabel("Probabilidade de título (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(final_values) * 1.24)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.99, 0.015, "Fonte: datasets Kaggle + simulação Monte Carlo | Seed 42", ha="right", va="bottom", fontsize=9, color=MUTED)
    fig.tight_layout(pad=2.0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", pad_inches=0.25, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE)


def main() -> None:
    data = pd.read_csv(PROB_PATH).sort_values("champion_probability", ascending=False).reset_index(drop=True)
    frames = []
    total = 42
    # ease-out animation: fast start, smooth finish
    for i in range(total):
        t = i / (total - 1)
        progress = 1 - (1 - t) ** 3
        frames.append(make_frame(data, progress, i, total))
    # hold final frame
    frames.extend([frames[-1]] * 10)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
    print(GIF_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
