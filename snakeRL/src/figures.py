"""Figures of the README from results/eval.json (written by src.evaluate).

uv run python -m src.figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
MAX_SCORE = 222 # 15 x 15 = 225 cells, the snake starts with 3

# order of the rows: baselines first, then the agents, each one followed by its learned counterpart
AGENTS = (
    ("greedy", "Heuristique gloutonne", "baseline"),
    ("mlp-8", "DQN MLP · 8 features", "dqn"),
    ("flood-greedy", "Heuristique flood-fill", "baseline"),
    ("mlp-25", "DQN MLP · 25 features", "dqn"),
    ("cnn-film", "DQN CNN + FiLM", "dqn"),
)
COLORS = {"baseline": "#8a8984", "dqn": "#2a78d6"}
CAUSES = (
    ("collision évitable", ("walls-avoidable", "self-avoidable"), "#2a78d6"),
    ("collision piégée", ("walls-trapped", "self-trapped"), "#eb6834"),
    ("famine (225 steps sans fruit)", ("starving",), "#1baf7a"),
    ("limite de 5 000 steps", ("timeout",), "#eda100"),
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scores_figure(results):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for row, (key, label, kind) in enumerate(AGENTS):
        s = results[key]["summary"]
        scores = np.array([g["score"] for g in results[key]["games"]])
        y = row + rng.uniform(-0.18, 0.18, len(scores))
        ax.scatter(scores, y, s=14, color=COLORS[kind], alpha=0.45, linewidths=0)
        ax.plot(s["ci95"], [row, row], color=INK, linewidth=2, solid_capstyle="round")
        ax.scatter([s["mean"]], [row], s=60, color="white", edgecolors=INK, linewidths=2, zorder=3)
        ax.text(MAX_SCORE + 6, row, f"{s['mean']:.0f}  (méd. {s['median']:.0f})", va="center", color=INK, fontsize=9)
    ax.axvline(MAX_SCORE, color=MUTED, linewidth=1, linestyle="--")
    ax.text(MAX_SCORE + 3, -0.4, "plateau plein (222)", ha="left", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(AGENTS)), [label for _, label, _ in AGENTS], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-5, MAX_SCORE + 55)
    ax.set_xticks(range(0, 225, 50))
    ax.set_xlabel("fruits mangés par partie", color=MUTED)
    ax.set_title("Score sur 100 parties (ε = 0) : points = parties, ○ = moyenne, trait = IC 95 % bootstrap",
                 loc="left", fontsize=10, color=INK, pad=16)
    style(ax)
    ax.scatter([], [], s=14, color=COLORS["baseline"], label="sans apprentissage")
    ax.scatter([], [], s=14, color=COLORS["dqn"], label="DQN")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scores.png", dpi=150, facecolor="white")


def deaths_figure(results):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for row, (key, label, _) in enumerate(AGENTS):
        causes = results[key]["summary"]["causes"]
        left = 0.0
        for name, keys, color in CAUSES:
            share = sum(causes.get(k, 0.0) for k in keys)
            if share == 0:
                continue
            # 2px surface gap between segments
            ax.barh(row, share, left=left, height=0.6, color=color, edgecolor="white", linewidth=2)
            if share >= 0.06:
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center", va="center", color="white", fontsize=8)
            left += share
    ax.set_yticks(range(len(AGENTS)), [label for _, label, _ in AGENTS], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Comment se terminent les 100 parties", loc="left", fontsize=10, color=INK)
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CAUSES]
    ax.legend(handles, [n for n, _, _ in CAUSES], loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=4,
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "deaths.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = json.loads(Path("results/eval.json").read_text())
    scores_figure(results)
    deaths_figure(results)
