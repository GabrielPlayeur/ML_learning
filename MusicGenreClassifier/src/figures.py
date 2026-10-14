"""Figures du README à partir de results/benchmark.json (écrit par src.benchmark).

uv run python -m src.figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
GENRES = ["hiphop", "classical", "blues", "metal", "jazz", "country", "pop", "rock", "disco", "reggae"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

ARCHITECTURES = (
    ("logreg", "Régression logistique\n(moy. + écart-type MFCC)"),
    ("mlp", "MLP"),
    ("lstm", "LSTM"),
    ("cnn", "CNN"),
)
# (variante, référence, libellé) : différence appariée par seed, même split
ABLATIONS = (
    ("mlp+deltas", "mlp", "MLP + deltas"),
    ("cnn+deltas", "cnn", "CNN + deltas"),
    ("lstm+deltas", "lstm", "LSTM + deltas"),
    ("cnn-noaug", "cnn", "CNN sans augmentation"),
    ("lstm-mean", "lstm", "LSTM : moyenne des états\nau lieu du dernier"),
)


def style(ax, grid_axis="x"):
    for side in ("top", "right", "left" if grid_axis == "x" else "bottom"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def values(results, name, metric):
    return np.array([run[metric] for run in results[name].values()]) * 100


def architectures_figure(results):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for row, (name, label) in enumerate(ARCHITECTURES):
        for metric, color, offset in (("acc", BLUE, -0.12), ("track_acc", ORANGE, 0.12)):
            v = values(results, name, metric)
            ax.scatter(v, np.full(len(v), row + offset), s=22, color=color, alpha=0.5, linewidths=0)
            ax.plot([v.min(), v.max()], [row + offset] * 2, color=color, linewidth=2, solid_capstyle="round")
            ax.scatter([v.mean()], [row + offset], s=70, color="white", edgecolors=color, linewidths=2, zorder=3)
            ax.text(v.max() + 1.2, row + offset, f"{v.mean():.1f} %", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(ARCHITECTURES)), [label for _, label in ARCHITECTURES], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(45, 100)
    ax.set_xlabel("accuracy sur le test (%)", color=MUTED)
    ax.set_title("Accuracy test, 3 seeds (○ = moyenne, points = seeds) · MFCC seuls, même split par seed",
                 loc="left", fontsize=10, color=INK)
    ax.scatter([], [], s=40, color=BLUE, label="par segment de 3 s")
    ax.scatter([], [], s=40, color=ORANGE, label="par morceau (vote sur 10 segments)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "architectures.png", dpi=150, facecolor="white")


def ablations_figure(results):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for row, (variant, reference, label) in enumerate(ABLATIONS):
        delta = values(results, variant, "acc") - values(results, reference, "acc")
        color = BLUE
        ax.scatter(delta, np.full(len(delta), row), s=22, color=color, alpha=0.5, linewidths=0)
        ax.plot([delta.min(), delta.max()], [row, row], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([delta.mean()], [row], s=70, color="white", edgecolors=color, linewidths=2, zorder=3)
        ax.text(max(delta.max(), 0) + 0.4, row, f"{delta.mean():+.1f} pts", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(ABLATIONS)), [label for *_, label in ABLATIONS], color=INK, fontsize=9)
    ax.invert_yaxis()
    lim = max(abs(np.array(ax.get_xlim()))) + 2
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("écart d'accuracy par segment avec la référence, même seed (points de %)", color=MUTED)
    ax.set_title("Effet de chaque modification, toutes choses égales par ailleurs", loc="left", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png", dpi=150, facecolor="white")


def curves_figure(results, seed="seed0"):
    names = (("mlp", "MLP"), ("lstm", "LSTM"), ("cnn", "CNN"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (name, label) in zip(axes, names):
        history = results[name][seed]["history"]
        ax.plot(history["train_loss"], color=BLUE, linewidth=2, label="train")
        ax.plot(history["val_loss"], color=ORANGE, linewidth=2, label="validation")
        best = results[name][seed]["best_epoch"]
        ax.axvline(best, color=MUTED, linewidth=1, linestyle="--")
        ax.text(best, 2.25, " meilleur epoch", fontsize=7.5, color=MUTED, va="top")
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("epoch", color=MUTED)
        style(ax, "y")
    axes[0].set_ylabel("cross-entropy", color=MUTED)
    axes[0].set_ylim(0, 2.3)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Courbes d'apprentissage (seed 0) : l'écart train / validation mesure le sur-apprentissage",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=150, facecolor="white")


def confusion_figure(results, name="cnn"):
    cm = sum(np.array(run["confusion"]) for run in results[name].values())
    cm = cm / cm.sum(axis=1, keepdims=True)
    order = np.argsort(-np.diag(cm))
    cm = cm[order][:, order]
    labels = [GENRES[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#3987e5", "#104281"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.03:
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xticks(range(10), labels, rotation=45, ha="right", color=INK, fontsize=9)
    ax.set_yticks(range(10), labels, color=INK, fontsize=9)
    ax.set_xlabel("genre prédit", color=MUTED)
    ax.set_ylabel("vrai genre", color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("CNN : confusion par segment (% de chaque ligne, 3 seeds)", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_cnn.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = json.loads(Path("results/benchmark.json").read_text())
    architectures_figure(results)
    ablations_figure(results)
    curves_figure(results)
    confusion_figure(results)
