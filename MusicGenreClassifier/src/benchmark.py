"""Expériences contrôlées : même split (par seed), même budget, même protocole pour chaque variante.

uv run python -m src.benchmark                 # toutes les expériences, 3 seeds -> results/benchmark.json
uv run python -m src.benchmark -e cnn mlp      # seulement certaines variantes

Un run déjà présent dans le JSON n'est pas relancé : on peut interrompre et reprendre.
Pas de W&B ici (CSVLogger local) : les courbes d'apprentissage sont relues depuis metrics.csv.
"""
import argparse
import csv
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
from hydra import compose, initialize
from lightning.pytorch.loggers import CSVLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import GTZANDataModule
from .train import fit

SEEDS = (0, 1, 2)
RESULTS = Path("results/benchmark.json")

# nom -> overrides Hydra ; chaque variante ne change qu'un facteur par rapport à sa référence
EXPERIMENTS = {
    # architectures (MFCC seuls)
    "mlp": ["model=mlp"],
    "cnn": ["model=cnn"],
    "lstm": ["model=lstm"],
    # + deltas et delta-deltas (39 coefficients au lieu de 13)
    "mlp+deltas": ["model=mlp", "data.use_deltas=true"],
    "cnn+deltas": ["model=cnn", "data.use_deltas=true"],
    "lstm+deltas": ["model=lstm", "data.use_deltas=true"],
    # ablation : sans augmentation (bruit + masques temps/fréquence)
    "cnn-noaug": ["model=cnn", "data.aug_probability=0.0"],
    # LSTM : moyenne des états cachés au lieu du dernier
    "lstm-mean": ["model=lstm", "model.pooling=mean"],
    # contre-exemple : split par segment, les segments d'un même morceau sont en train ET en test
    "cnn-leaky": ["model=cnn", "data.split_by_track=false"],
}


def track_vote(probs, targets, groups):
    """Prédiction par morceau : moyenne des probabilités de ses segments, puis argmax."""
    track_ids = np.unique(groups)
    track_probs = np.stack([probs[groups == t].mean(axis=0) for t in track_ids])
    track_targets = np.array([targets[groups == t][0] for t in track_ids])
    return track_probs.argmax(axis=1), track_targets


def scores(probs, targets, groups):
    preds = probs.argmax(axis=1)
    track_preds, track_targets = track_vote(probs, targets, groups)
    return {
        "acc": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, average="macro"),
        "track_acc": accuracy_score(track_targets, track_preds),
        "track_f1": f1_score(track_targets, track_preds, average="macro"),
        "confusion": confusion_matrix(targets, preds, labels=range(10)).tolist(),
    }


def run_network(name, seed):
    overrides = EXPERIMENTS[name] + [f"seed={seed}", f"data.random_seed={seed}"]
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=overrides)

    logger = CSVLogger("results/logs", name=name, version=f"seed{seed}")
    start = time.time()
    model, datamodule, trainer = fit(cfg, logger, progress_bar=False)

    # metrics.csv : une ligne par log, train et val de la même epoch sur des lignes séparées
    history = {"train_loss": {}, "val_loss": {}, "train_acc": {}, "val_acc": {}}
    with open(Path(logger.log_dir) / "metrics.csv") as file:
        for row in csv.DictReader(file):
            for key in history:
                value = row.get(key.replace("_", "/", 1))
                if value and row["epoch"]:
                    history[key][int(row["epoch"])] = float(value)
    history = {key: [values[e] for e in sorted(values)] for key, values in history.items()}
    result = scores(model.test_probs, model.test_targets, datamodule.test_groups)
    result.update({
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": int(np.argmin(history["val_loss"])),
        "epochs": len(history["val_loss"]),
        "seconds": time.time() - start,
        "history": history,
    })
    return result


def run_logreg(seed):
    """Baseline non profonde : moyenne et écart-type de chaque MFCC sur le segment (26 valeurs) -> régression logistique."""
    datamodule = GTZANDataModule("dataset/mfcc.npz", random_seed=seed)
    datamodule.setup()

    def summarize(X):
        return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)

    (X_train, y_train), (X_test, y_test) = datamodule.train_data, datamodule.test_data
    start = time.time()
    clf = LogisticRegression(max_iter=2000).fit(summarize(X_train), y_train)
    result = scores(clf.predict_proba(summarize(X_test)), y_test, datamodule.test_groups)
    result.update({"params": int(clf.coef_.size + clf.intercept_.size), "seconds": time.time() - start})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments", nargs="+", choices=("logreg", *EXPERIMENTS), default=("logreg", *EXPERIMENTS))
    parser.add_argument("-s", "--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

    for name in args.experiments:
        for seed in args.seeds:
            key = f"seed{seed}"
            if key in results.get(name, {}):
                continue
            result = run_logreg(seed) if name == "logreg" else run_network(name, seed)
            results.setdefault(name, {})[key] = result
            RESULTS.write_text(json.dumps(results, indent=1))
            print(f"{name:12s} {key}: acc {result['acc']:.3f} | f1 {result['f1']:.3f} | "
                  f"track acc {result['track_acc']:.3f} | {result['seconds']:.0f}s", flush=True)
