"""Evaluate trained agents and hand-written baselines on the same games (epsilon 0, fixed seeds).

uv run python -m src.evaluate                    # every run of RUNS, 100 games each -> results/eval.json
uv run python -m src.evaluate -g 20 -r greedy    # quick check of one run
"""
import argparse
import json
import random
from pathlib import Path
from time import time

import numpy as np
import torch

from .snake import Game
from .snake.snake import OPPOSITE
from .models import MLP, MLPScalars, CNN
from .module import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# name -> (model class, checkpoint) for the DQN agents, None for the hand-written baselines
RUNS = {
    "greedy": None,
    "flood-greedy": None,
    "mlp-8": (MLP, "mlp-new"),
    "mlp-25": (MLPScalars, "exp-mlp-scalars"),
    "cnn-film": (CNN, "exp-film3"),
}


class Baseline:
    """A policy without network: reads the Game directly, so the comparison with the DQN is about learning only."""
    state_type = "vector"

    def __init__(self, game_ref, flood: bool):
        self.game_ref = game_ref
        self.flood = flood
        self.epsilon = 0.0

    def action(self, state):
        game = self.game_ref[0]
        snake = game.snake
        head = snake.head.pos()
        moves = [d for d in "udlr" if d != OPPOSITE[snake.direction]]

        def fruit_dist(d):
            dx, dy = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}[d]
            x, y = head[0] + dx, head[1] + dy
            return min(abs(x - fx) + abs(y - fy) for fx, fy in game.fruits.positions)

        if not self.flood:
            # 8-feature rule: closest to the fruit among the moves that do not die right away
            danger = dict(zip("udlr", state[:4]))
            safe = [d for d in moves if not danger[d]] or moves
            return min(safe, key=fruit_dist)

        # same information as the 17 scalars of the networks: prefer a move after which the tail is reachable,
        # then the biggest area, then the closest fruit
        areas, tails = game.action_areas(snake.positions, snake.tail.pos(), head, snake.direction)
        return max(moves, key=lambda d: (tails[d], min(areas[d], snake.size), -fruit_dist(d)))


def play_games(name: str, n_games: int, max_steps: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    spec = RUNS[name]
    game_ref = [None]
    if spec is None:
        agent = Baseline(game_ref, flood=(name == "flood-greedy"))
    else:
        network, checkpoint = spec
        agent = Module(network, device, input_size=getattr(network, "input_size", 8), epsilon=0.0)
        agent.load(f"checkpoints/{checkpoint}.pt")
        agent.network.eval()

    game = Game(agent, log=False)
    game_ref[0] = game
    records = []
    for _ in range(n_games):
        game.reset()
        deaths_before = dict(game.death_log)
        state = game.observe()
        steps = 0
        while not game.stop and steps < max_steps:
            steps += 1
            state, _, _ = game.step(agent.action(state))
        cause = next((k for k, v in game.death_log.items() if v > deaths_before[k]), "timeout")
        d = game.last_death
        records.append({
            "score": game.score,
            "steps": steps,
            "cause": cause,
            "safe_moves": None if d is None else d["safe_moves"],
        })
    return records


def summary(records):
    scores = np.array([r["score"] for r in records])
    # 95% bootstrap interval of the mean score
    rng = np.random.default_rng(0)
    boot = rng.choice(scores, (10_000, len(scores))).mean(axis=1)
    causes = {}
    for r in records:
        label = r["cause"]
        if label in ("walls", "self"):
            label += "-trapped" if r["safe_moves"] == 0 else "-avoidable"
        causes[label] = causes.get(label, 0) + 1
    return {
        "mean": float(scores.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "median": float(np.median(scores)), "max": int(scores.max()), "min": int(scores.min()),
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "causes": {k: v / len(records) for k, v in sorted(causes.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--games", type=int, default=100)
    parser.add_argument("-s", "--max-steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-r", "--runs", nargs="+", choices=tuple(RUNS), default=tuple(RUNS))
    parser.add_argument("-o", "--output", default="results/eval.json")
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in args.runs:
        start = time()
        records = play_games(name, args.games, args.max_steps, args.seed)
        results[name] = {"summary": summary(records), "games": records}
        s = results[name]["summary"]
        print(f"{name:13s} mean {s['mean']:6.1f} [{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}] median {s['median']:5.0f} "
              f"max {s['max']:3d} | {s['causes']} | {time() - start:.0f}s")
        path.write_text(json.dumps(results, indent=1))
