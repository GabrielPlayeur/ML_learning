import torch
import torch.nn as nn

class MLPGenreClassifier(nn.Module):
    """Baseline sans a priori : le segment (frames, coefficients) est aplati en un seul vecteur."""
    def __init__(
        self,
        input_shape: tuple[int, int] = (130, 13),
        hidden_layers: tuple[int, ...] = (256, 64),
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Flatten()]
        input_features = input_shape[0] * input_shape[1]
        for output_features in hidden_layers:
            layers.extend([
                nn.Linear(input_features, output_features),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_features = output_features
        layers.append(nn.Linear(input_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
