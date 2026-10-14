
from time import time_ns
import math

from .snake import Game

class Trainer:
    def __init__(self, module, total_steps: int, max_step_per_epoch: int=100):
        self.module = module
        self.total_steps = total_steps
        self.steps_done = 0
        self.max_step_per_epoch = max_step_per_epoch
        self.n_val_epoch = 10

        self.game = Game(module, log=False)
        self.history = {
            "score": [], "reward": [], "loss": [], "steps": [], "epsilon": [],
            "val_score": [], "val_reward": [], "val_steps": [],
        }
        # one record per finished game: (steps, cause, length, safe_moves, best_area, random_action)
        # "train" is polluted by epsilon (random_action), "val" is played at epsilon 0
        self.deaths = {"train": [], "val": []}
        self.q_history = [] # mean Q-value of the batches of each episode
        self.x_steps = [] # cumulative env steps at the end of each episode (x-axis of the plots)
        # walls / self are split: "avoidable" = a safe move existed just before the fatal one, "trapped" = none did
        self.death_causes = ("walls-avoidable", "walls-trapped", "self-avoidable", "self-trapped", "starving", "timeout")
        self.death_colors = ("tab:blue", "lightsteelblue", "tab:orange", "navajowhite", "tab:green", "tab:red")
        self.death_history = {cause: [] for cause in self.death_causes}

    def train(self, log_every_steps: int=10_000):
        start_time = time_ns()
        seg_time, seg_steps = start_time, 0 # start of the current segment (validation excluded) for the speed
        print("START TRAINING")
        e = 0
        next_log = log_every_steps
        while self.steps_done < self.total_steps:
            state = self.game.observe()
            deaths_before = dict(self.game.death_log)
            reward_sum = 0.0
            loss_sum = 0.0 # becomes a tensor on the first update
            q_sum = 0.0
            loss_count = 0

            i = 0
            while not self.game.stop and i < self.max_step_per_epoch:
                i += 1
                self.steps_done += 1
                action = self.module.action(state)
                next_state, reward, done = self.game.step(action)
                loss = self.module.learn(state, action, reward, next_state, done)
                self.module.decay_epsilon()
                reward_sum += reward
                if loss is not None:
                    loss_sum = loss_sum + loss
                    q_sum = q_sum + self.module.last_q
                    loss_count += 1
                state = next_state
                if self.steps_done >= self.total_steps:
                    break

            e += 1
            self.module.end_episode()

            record = self._death_record(deaths_before, self.module.last_action_random)
            self.deaths["train"].append(record)
            label = self.death_label(record)
            for c in self.death_causes:
                self.death_history[c].append(int(c == label))

            self.x_steps.append(self.steps_done)
            self.history["score"].append(self.game.score)
            self.history["reward"].append(reward_sum / i)
            self.history["loss"].append(float(loss_sum) / loss_count if loss_count > 0 else float("nan"))
            self.q_history.append(float(q_sum) / loss_count if loss_count > 0 else float("nan"))
            self.history["steps"].append(i)
            self.history["epsilon"].append(self.module.epsilon)

            do_log = self.steps_done >= next_log
            if do_log:
                next_log += log_every_steps
                speed = (self.steps_done - seg_steps) / ((time_ns() - seg_time) / 1e9)
                results = [self.validation() for _ in range(self.n_val_epoch)]
                val_score, val_reward, val_steps = (sum(v) / len(results) for v in zip(*(r[:3] for r in results)))
                self.deaths["val"].extend(r[3] for r in results)
                seg_time, seg_steps = time_ns(), self.steps_done
            else:
                val_score, val_reward, val_steps = float("nan"), float("nan"), float("nan")
            self.history["val_score"].append(val_score)
            self.history["val_reward"].append(val_reward)
            self.history["val_steps"].append(val_steps)

            if do_log:
                elapsed = (time_ns() - start_time) / 1e9
                minutes, seconds = divmod(int(elapsed), 60)
                print(f"[{minutes:02d}:{seconds:02d}] {(self.steps_done/self.total_steps)*100:.2f}% {self.steps_done}/{self.total_steps} (ep {e}):\n\tstep {val_steps} | score {val_score} | reward {val_reward:.2f} | epsilon {self.history["epsilon"][-1]:.3f} | {speed:.0f} steps/s")

            self.game.reset()

        self.death_report()

    def _death_record(self, deaths_before, random_action):
        """How the game that just finished ended (death_log is cumulative, hence the diff)."""
        cause = next((k for k, v in self.game.death_log.items() if v > deaths_before[k]), "timeout")
        d = self.game.last_death
        if d is None:
            return (self.steps_done, "timeout", self.game.snake.size, None, None, False)
        return (self.steps_done, cause, d["length"], d["safe_moves"], d["best_area"], random_action)

    @staticmethod
    def death_label(record):
        """walls / self are split: "avoidable" = a safe move existed just before the fatal one, "trapped" = none did."""
        cause, safe_moves = record[1], record[3]
        if cause in ("walls", "self"):
            return f"{cause}-{'trapped' if safe_moves == 0 else 'avoidable'}"
        return cause

    def report_lines(self, source: str = "train", last_fraction: float = 0.2) -> list[str]:
        """Were the deaths of the last part of the training unavoidable (trapped) or mistakes?"""
        threshold = self.steps_done * (1 - last_fraction)
        recent = [d for d in self.deaths[source] if d[0] >= threshold]
        title = "TRAIN (epsilon on)" if source == "train" else "VALIDATION (epsilon 0)"
        lines = [
            f"DEATH REPORT {title} - last {last_fraction:.0%} of the steps, {len(recent)} games",
            f"{'cause':9s} {'n':>5s} {'length':>7s} {'trapped':>8s} {'avoidable':>10s} {'safe moves':>11s} {'best area':>10s} {'random action':>14s}",
        ]
        for cause in ("self", "walls", "starving", "timeout"):
            rows = [d for d in recent if d[1] == cause]
            if not rows:
                continue
            n = len(rows)
            line = f"{cause:9s} {n:5d} {sum(d[2] for d in rows)/n:7.1f}"
            if cause != "timeout":
                trapped = sum(d[3] == 0 for d in rows) / n
                line += (
                    f" {trapped:8.0%} {1-trapped:10.0%} {sum(d[3] for d in rows)/n:11.2f}"
                    f" {sum(d[4] for d in rows)/n:10.1f} {sum(d[5] for d in rows)/n:14.0%}"
                )
            lines.append(line)
        return lines

    def death_report(self, last_fraction: float = 0.2):
        for source in ("train", "val"):
            print("\n".join(self.report_lines(source, last_fraction)))

    def validation(self):
        save_epsilon = self.module.epsilon
        self.module.epsilon = 0.0
        self.game.reset()
        deaths_before = dict(self.game.death_log)

        state = self.game.observe()
        reward_sum = 0.0
        i = 0
        while not self.game.stop and i < self.max_step_per_epoch:
            i += 1
            action = self.module.action(state)
            state, reward, _ = self.game.step(action)
            reward_sum += reward

        score = self.game.score
        record = self._death_record(deaths_before, False)

        self.module.epsilon = save_epsilon
        self.game.reset()
        return score, reward_sum / max(i, 1), i, record

    def plot(self, path: str | None = None):
        import matplotlib.pyplot as plt

        base_keys = [key for key in self.history if not key.startswith("val_")]
        window = 100

        fig = plt.figure(figsize=(14, 9.5))
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.75])
        axes = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(6)]
        for ax, key in zip(axes, base_keys):
            ax.plot(self.x_steps, self.history[key], label="train")
            val_key = f"val_{key}"
            if val_key in self.history:
                xs = [x for x, v in zip(self.x_steps, self.history[val_key]) if not math.isnan(v)]
                ys = [v for v in self.history[val_key] if not math.isnan(v)]
                ax.plot(xs, ys, label="val", marker="o", markersize=3, linestyle=":")
                ax.legend()
            if key == "loss":
                ax_q = ax.twinx()
                ax_q.plot(self.x_steps, self.q_history, color="tab:red", alpha=0.6, label="mean Q")
                ax_q.set_ylabel("mean Q", color="tab:red")
            ax.set_title(key)
            ax.set_xlabel("steps")

        ax = axes[len(base_keys)]
        n = len(self.x_steps)
        if n >= window:
            xs = self.x_steps[window - 1:]
            ratios = []
            for cause in self.death_causes:
                h = self.death_history[cause]
                csum = [0]
                for v in h:
                    csum.append(csum[-1] + v)
                ratios.append([(csum[i + 1] - csum[i + 1 - window]) / window for i in range(window - 1, n)])
            ax.stackplot(xs, *ratios, labels=self.death_causes, colors=self.death_colors)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"death causes (rolling {window})")
        ax.set_xlabel("steps")

        for ax in axes[len(base_keys) + 1:]:
            ax.set_visible(False)

        # death causes of the validation games (epsilon 0): one bar per validation, its games split by cause
        ax_val = fig.add_subplot(grid[2, 0])
        by_x = {}
        for record in self.deaths["val"]:
            by_x.setdefault(record[0], []).append(self.death_label(record))
        xs = list(by_x)
        width = (xs[1] - xs[0]) * 0.8 if len(xs) > 1 else 1
        bottoms = [0.0] * len(xs)
        for cause, color in zip(self.death_causes, self.death_colors):
            fractions = [labels.count(cause) / len(labels) for labels in by_x.values()]
            ax_val.bar(xs, fractions, width=width, bottom=bottoms, color=color, label=cause)
            bottoms = [b + f for b, f in zip(bottoms, fractions)]
        ax_val.set_ylim(0, 1)
        ax_val.set_title("death causes - validation (epsilon 0)")
        ax_val.set_xlabel("steps")

        ax_text = fig.add_subplot(grid[2, 1:])
        ax_text.axis("off")
        text = "\n\n".join("\n".join(self.report_lines(source)) for source in ("train", "val"))
        ax_text.text(0, 1, text, family="monospace", fontsize=8, va="top")

        from matplotlib.ticker import FuncFormatter, MaxNLocator
        for ax in axes[:len(base_keys) + 1] + [ax_val]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))

        from matplotlib.ticker import FuncFormatter, MaxNLocator
        for ax in axes[:len(base_keys) + 1] + [ax_val]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        fig.tight_layout()

        if path:
            fig.savefig(path)
        else:
            plt.show()
