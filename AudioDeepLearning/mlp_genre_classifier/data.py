import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def split_data(inputs: np.ndarray, targets: np.ndarray, random_seed: int = 42):
    train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(
        inputs,
        targets,
        test_size=0.3,
        random_state=random_seed,
        stratify=targets,
    )
    validation_inputs, test_inputs, validation_targets, test_targets = train_test_split(
        temp_inputs,
        temp_targets,
        test_size=0.5,
        random_state=random_seed,
        stratify=temp_targets,
    )
    return (
        train_inputs,
        validation_inputs,
        test_inputs,
        train_targets,
        validation_targets,
        test_targets,
    )


def normalize_data(train_inputs, validation_inputs, test_inputs):
    mean = train_inputs.mean(axis=(0, 1), keepdims=True)
    std = train_inputs.std(axis=(0, 1), keepdims=True)
    train_inputs = (train_inputs - mean) / (std + 1e-8)
    validation_inputs = (validation_inputs - mean) / (std + 1e-8)
    test_inputs = (test_inputs - mean) / (std + 1e-8)
    return train_inputs, validation_inputs, test_inputs, mean, std


def create_data_loader(inputs, targets, batch_size: int, shuffle: bool = False):
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    tensor_targets = torch.tensor(targets, dtype=torch.long)
    return DataLoader(
        TensorDataset(tensor_inputs, tensor_targets),
        batch_size=batch_size,
        shuffle=shuffle,
    )