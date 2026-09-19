import os
import argparse
import json

import numpy as np
import torch

from .config import MLPConfig, CNNConfig, RNNLSTMConfig
from .data import (
    MFCCTrainAugmentation,
    create_data_loader,
    load_data,
    normalize_data,
    split_data,
)
from .evaluation import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    save_evaluation_report,
)
from .models import MLPGenreClassifier, CNNGenreClassifier, RNNLSTMGenreClassifier
from .trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("mlp", "cnn", "rnn"),
        required=True,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def create_model(model_name, input_shape, num_classes, config):
    if model_name == "mlp":
        return MLPGenreClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_layers=config.hidden_layers,
        )

    if model_name == "cnn":
        return CNNGenreClassifier(
            input_shape=input_shape,
            channels=config.channels,
            num_classes=num_classes,
            dropout=config.dropout,
        )

    if model_name == "rnn":
        return RNNLSTMGenreClassifier(
            input_shape=input_shape,
            hidden_layers=config.hidden_layers,
            num_classes=num_classes,
            dropout=config.dropout,
        )

    raise ValueError(f"Modèle inconnu : {model_name}")

def get_config(model_name: str):
    if model_name == "mlp":
        return MLPConfig()

    if model_name == "cnn":
        return CNNConfig()

    if model_name == "rnn":
        return RNNLSTMConfig()

    raise ValueError(f"Modèle inconnu : {model_name}")


def load_class_names(dataset_path: str, num_classes: int) -> list[str]:
    with open(dataset_path, "r") as file:
        mapping = json.load(file)["mapping"]
    class_names = []
    for name in mapping:
        clean_name = name.strip()
        if clean_name and clean_name not in class_names:
            class_names.append(clean_name)
    if len(class_names) != num_classes:
        return [f"Classe {index}" for index in range(num_classes)]
    return class_names

def main() -> None:
    args = parse_args()
    config = get_config(args.model)
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    inputs, targets, groups = load_data(config.dataset_path)

    (
        train_inputs,  validation_inputs,  test_inputs, 
        train_targets, validation_targets, test_targets,
    ) = split_data(inputs, targets, groups, config.random_seed)

    train_inputs, validation_inputs, test_inputs, mean, std = normalize_data(train_inputs, validation_inputs, test_inputs)

    train_transform = None
    if config.train_augmentation:
        train_transform = MFCCTrainAugmentation(
            probability=config.augmentation_probability,
            noise_std=config.augmentation_noise_std,
            time_mask_max=config.augmentation_time_mask_max,
            frequency_mask_max=config.augmentation_frequency_mask_max,
        )

    train_loader = create_data_loader(
        train_inputs,
        train_targets,
        config.batch_size,
        shuffle=True,
        transform=train_transform,
        oversample_classes=config.oversample_classes,
        oversample_factor=config.oversample_factor,
    )
    validation_loader = create_data_loader(validation_inputs, validation_targets, config.batch_size)
    test_loader = create_data_loader(test_inputs, test_targets, config.batch_size)

    model = create_model(
        args.model,
        input_shape=train_inputs.shape[1:],
        num_classes=len(np.unique(targets)),
        config=config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    trainer = Trainer(model, optimizer, scheduler, torch.nn.CrossEntropyLoss(), device)

    history, best_validation_accuracy = trainer.fit(train_loader, validation_loader, config.epochs, config.patience, config.min_delta)

    plot_training_history(history, output_path=config.output_curves_path)
    test_metrics = trainer.validate(test_loader)
    class_names = load_class_names(config.dataset_path, len(np.unique(targets)))
    save_evaluation_report(test_metrics, class_names, config.output_report_path)
    plot_confusion_matrix(test_metrics, class_names, config.output_confusion_path)
    plot_roc_curves(test_metrics, class_names, config.output_roc_path)

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
        f"test accuracy: {test_metrics['accuracy']:.4f} - "
        f"test F1 macro: {test_metrics['f1_macro']:.4f} - "
        f"test ROC-AUC: {test_metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
