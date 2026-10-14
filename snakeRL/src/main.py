import torch
import argparse

from .snake import Game
from .models import Humain, MLP, MLPScalars, CNN, CNNBig, CNNBase

from .train import Trainer
from .module import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

MODELS = {
    "MLP": MLP,
    "MLPScalars": MLPScalars,
    "CNN": CNN,
    "CNNBig": CNNBig,
    "CNNBase": CNNBase,
}

NAME = "cnn"

def train(name: str, total_steps: int, model_name: str):

    module = Module(
        MODELS[model_name],
        device,
        input_size = getattr(MODELS[model_name], "input_size", 8),
        lr = 1e-4,
        epsilon = 1.0,
        epsilon_min = 0.01,
        epsilon_decay_steps = 100_000,
        gamma = 0.99,
        batch_size = 64,
        buffer_capacity = 100_000,
        target_update_freq = 200, # real_freq = train * target
        train_freq = 4,
        n_step = 1,
        )

    trainer = Trainer(
        module=module,
        total_steps=total_steps,
        max_step_per_epoch=5_000
        )
    trainer.train(log_every_steps=10_000)
    trainer.plot(f"charts/{name}.png")
    module.save(f"checkpoints/{name}.pt")

def play(name: str, model_name: str):
    replay_model = Module(MODELS[model_name], device, epsilon=0.0)
    replay_model.load(f"checkpoints/{name}.pt")
    Game(replay_model, log=True, delay=True).play()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", choices=("train", "play"), default="train")
    parser.add_argument("-o", "--output", default=NAME)
    parser.add_argument("-n", "--n-steps", type=int, default=200_000)
    parser.add_argument("-m", "--model", choices=tuple(MODELS), default="CNN")
    args = parser.parse_args()
    if args.type == "play":
        play(args.output, args.model)
    else:
        train(args.output, args.n_steps, args.model)