
import torch
import torch.nn as nn

class MLP(nn.Module):
    state_type = "vector"

    def __init__(self, input_size: int = 8, num_actions: int = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.layers(x)


class MLPScalars(MLP):
    """Same architecture, fed Game.get_state_scalars(): the original 8 features (danger + food direction)
    plus the 17 engineered scalars also used by the CNN (direction, area, space for the body, tail reachable,
    hunger) - no spatial grid at all. Compares against CNN to see how much of its gain is the raster map
    vs. these scalars."""
    state_type = "scalars"
    input_size = 25 # 8 (danger + food direction) + 17 (action_scalars)