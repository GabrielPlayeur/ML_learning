import torch
import torch.nn as nn

class RNNLSTMGenreClassifier(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int] = (130, 13),
        hidden_layers: tuple[int, int] = (64, 64),
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_features = input_shape[1]

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_layers[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[1], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected inputs with shape (batch, sequence, features), got {tuple(inputs.shape)}"
            )

        sequence_outputs, _ = self.lstm(inputs)
        last_outputs = sequence_outputs[:, -1, :]
        return self.classifier(last_outputs)