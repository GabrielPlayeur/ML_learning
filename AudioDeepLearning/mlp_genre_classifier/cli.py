import os

import numpy as np
import torch

from .config import TrainingConfig
from .data import create_data_loader, load_data, normalize_data, split_data
from .evaluation import plot_training_history
from .model import GenreClassifier
from .trainer import Trainer


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = TrainingConfig()
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    inputs, targets = load_data(config.dataset_path)
    (
        train_inputs, validation_inputs, test_inputs,
        train_targets, validation_targets, test_targets,
    ) = split_data(inputs, targets, config.random_seed)

    train_inputs, validation_inputs, test_inputs, mean, std = normalize_data(train_inputs, validation_inputs, test_inputs)

    train_loader = create_data_loader(train_inputs, train_targets, config.batch_size, shuffle=True)
    validation_loader = create_data_loader(validation_inputs, validation_targets, config.batch_size	)
    test_loader = create_data_loader(test_inputs, test_targets, config.batch_size)

    model = GenreClassifier(
        input_shape=train_inputs.shape[1:],
        hidden_layers=config.hidden_layers,
        num_classes=len(np.unique(targets)),
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss(), device)

    history, best_validation_accuracy = trainer.fit(train_loader, validation_loader, config.epochs, config.patience, config.min_delta)

    plot_training_history(history)
    test_metrics = trainer.validate(test_loader)

    output_directory = os.path.dirname(config.output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": train_inputs.shape[1:],
            "num_classes": len(np.unique(targets)),
            "mean": mean,
            "std": std,
            "config": config.__dict__,
        },
        config.output_path,
    )
    print(
        f"Best validation accuracy: {best_validation_accuracy:.4f} - "
        f"test accuracy: {test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
