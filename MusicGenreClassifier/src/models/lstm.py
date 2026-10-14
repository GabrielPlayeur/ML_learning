import torch
import torch.nn as nn

class LSTMGenreClassifier(nn.Module):
    """Le segment est lu comme une séquence de frames (batch, frames, coefficients).
    pooling = "last" : seul le dernier état caché sert à classer ; "mean" : moyenne des états sur toutes les frames."""
    def __init__(
        self,
        input_shape: tuple[int, int] = (130, 13),
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling inconnu : {pooling}")
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_outputs, _ = self.lstm(inputs)
        if self.pooling == "last":
            features = sequence_outputs[:, -1, :]
        else:
            features = sequence_outputs.mean(dim=1)
        return self.classifier(features)
