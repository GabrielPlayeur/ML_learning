import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class MFCCTrainAugmentation:
    def __init__(
        self,
        probability: float = 0.5,
        noise_std: float = 0.02,
        time_mask_max: int = 8,
        frequency_mask_max: int = 3,
    ):
        self.probability = probability
        self.noise_std = noise_std
        self.time_mask_max = time_mask_max
        self.frequency_mask_max = frequency_mask_max

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.probability:
            return inputs

        augmented = inputs.clone()
        if self.noise_std > 0:
            augmented += torch.randn_like(augmented) * self.noise_std

        time_size, frequency_size = augmented.shape
        time_width = int(torch.randint(0, min(self.time_mask_max, time_size) + 1, (1,)))
        if time_width:
            time_start = int(torch.randint(0, time_size - time_width + 1, (1,)))
            augmented[time_start:time_start + time_width, :] = 0

        frequency_width = int(
            torch.randint(0, min(self.frequency_mask_max, frequency_size) + 1, (1,))
        )
        if frequency_width:
            frequency_start = int(
                torch.randint(0, frequency_size - frequency_width + 1, (1,))
            )
            augmented[:, frequency_start:frequency_start + frequency_width] = 0

        return augmented


class AudioFeatureDataset(Dataset):
    def __init__(self, inputs, targets, transform=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, self.targets[index]


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


def create_data_loader(
    inputs,
    targets,
    batch_size: int,
    shuffle: bool = False,
    transform=None,
    oversample_classes: tuple[int, ...] = (),
    oversample_factor: float = 1.0,
):
    dataset = AudioFeatureDataset(inputs, targets, transform=transform)
    sampler = None
    if oversample_classes and oversample_factor > 1:
        sample_weights = torch.ones(len(targets), dtype=torch.double)
        target_array = np.asarray(targets)
        for class_index in oversample_classes:
            sample_weights[target_array == class_index] = oversample_factor
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )