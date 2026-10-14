import torch
import torch.nn as nn

# must match Game.get_state_grid: the state is a flat vector = IN_MAT spatial planes (GRID x GRID) followed by IN_SCA scalars
GRID = 17
IN_MAT = 4
IN_SCA = 17
MAT_SIZE = IN_MAT * GRID * GRID
# scalars that describe one action each: one slice of 4 values ("udlr" order) per feature
ACTION_FEATURES = (
    slice(4, 8),   # area after each action / size^2
    slice(8, 12),  # is the area big enough for the body?
    slice(12, 16), # tail reachable after each action
)


def split(x):
    """Flat state (B, MAT_SIZE + IN_SCA) -> spatial planes (B, IN_MAT, GRID, GRID) and scalars (B, IN_SCA)."""
    return x[:, :MAT_SIZE].reshape(-1, IN_MAT, GRID, GRID), x[:, MAT_SIZE:]


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (Perez et al. 2018): the scalars scale and shift every channel of a feature map.
    A constant plane through a conv only adds a per-channel shift (beta), FiLM also scales (gamma), without storing planes.
    Zero init: starts as the identity, the network learns how much to listen to the scalars."""

    def __init__(self, in_features: int, channels: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, sca):
        gamma, beta = self.linear(sca)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + gamma) + beta


class CNN(nn.Module):
    """Spatial planes in the convs, the scalars injected where they matter instead of being concatenated:
    - FiLM: all the scalars modulate each conv layer (global context: direction, hunger, areas)
    - per-action features: the same small MLP turns the features of each action (area, space for the body,
      tail reachable) into a term added straight to the advantage of that action, so they are never drowned
      by the 18_496 conv features."""
    state_type = "grid"

    def __init__(self, input_size: int = 8, num_actions: int = 4, grid_size: int = 15):
        super().__init__()
        self.conv1 = nn.Conv2d(IN_MAT, 32, kernel_size=3, stride=1, padding=1)
        self.film1 = FiLM(IN_SCA, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.film2 = FiLM(IN_SCA, 64)

        self.fc = nn.Sequential(
            nn.Linear(18_496, 256),
            nn.ReLU(),
        )

        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_actions)
        self.action_mlp = nn.Sequential(
            nn.Linear(len(ACTION_FEATURES), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        mat, sca = split(x)
        h = torch.relu(self.film1(self.conv1(mat), sca))
        h = torch.relu(self.film2(self.conv2(h), sca))
        h = self.fc(h.flatten(1))
        feats = torch.stack([sca[:, s] for s in ACTION_FEATURES], dim=2) # (B, actions, features)
        v = self.value(h)
        a = self.advantage(h) + self.action_mlp(feats).squeeze(2)
        return v + a - a.mean(dim=1, keepdim=True)


class CNNBase(nn.Module):
    """The first CNN: each scalar is broadcast to a constant plane and goes through the convs with the spatial planes."""
    state_type = "grid"

    def __init__(self, input_size: int = 8, num_actions: int = 4, grid_size: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_MAT + IN_SCA, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(18_496, 64),
            nn.ReLU(),

            nn.Linear(64, num_actions),
        )

    def flatten_size(self, grid_size: int = GRID, in_channels: int = IN_MAT + IN_SCA) -> int:
        with torch.no_grad():
            return self.conv(torch.zeros(1, in_channels, grid_size, grid_size)).shape[1]

    def forward(self, x):
        mat, sca = split(x)
        planes = sca[:, :, None, None].expand(-1, -1, GRID, GRID) # the buffer stays flat, the planes are built on the fly
        return self.fc(self.conv(torch.cat([mat, planes], dim=1)))


class CNNBig(nn.Module):
    state_type = "grid"

    def __init__(self, input_size: int = 8, num_actions: int = 4, grid_size: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_MAT, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(18_496 + IN_SCA, 256),
            nn.ReLU(),
        )

        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, num_actions)

    def flatten_size(self, grid_size: int = GRID, in_channels: int = IN_MAT) -> int:
        with torch.no_grad():
            return self.conv(torch.zeros(1, in_channels, grid_size, grid_size)).shape[1]

    def forward(self, x):
        mat, sca = split(x)
        h = self.fc(torch.cat([self.conv(mat), sca], dim=1))
        v, a = self.value(h), self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


if __name__ == "__main__":
    print(CNNBase().flatten_size(), CNNBig().flatten_size())
