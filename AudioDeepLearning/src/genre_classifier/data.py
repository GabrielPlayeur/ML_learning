import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    groups = np.array(data["track_ids"])

    return inputs, targets, groups



def split_data(inputs, targets, groups, random_seed=42):
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_seed,
    )

    train_indices, temporary_indices = next(
        first_split.split(inputs, targets, groups)
    )

    temporary_groups = groups[temporary_indices]

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_seed,
    )

    validation_relative, test_relative = next(
        second_split.split(
            temporary_indices,
            targets[temporary_indices],
            temporary_groups,
        )
    )

    validation_indices = temporary_indices[validation_relative]
    test_indices = temporary_indices[test_relative]

    return (
        inputs[train_indices],
        inputs[validation_indices],
        inputs[test_indices],
        targets[train_indices],
        targets[validation_indices],
        targets[test_indices],
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