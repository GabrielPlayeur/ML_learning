import torch
import torch.nn as nn

class GenreClassifier(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int] = (130, 13),
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(32),

            nn.Flatten(),
            nn.Linear(32 * 16 * 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        return self.network(inputs)