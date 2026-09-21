import numpy as np
import lightning as L
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import AudioFeatureDataset
from .augmentation import MFCCTrainAugmentation

class GTZANDataModule(L.LightningDataModule):
    def __init__(self,
            dataset_path: str,
            batch_size: int = 32,
            test_size: float = 0.15,
            validation_size: float = 0.15,
            random_seed: int = 42,
            oversample_classes: tuple = (),
            oversample_factor: float = 1.0,
            aug_probability: float = 0.5,
            aug_noise_std: float = 0.02,
            aug_time_mask_max: int = 8,
            aug_frequency_mask_max: int = 3,

        ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        inputs, targets, groups = self.load_data()

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(inputs, targets, groups)

        X_train, X_val, X_test, self.mean, self.std = self.normalize_data(X_train, X_val, X_test)

        # np.save("data/norm_stats.npy", {"mean": self.mean, "std": self.std})

        self.train_data = (X_train, y_train)
        self.val_data   = (X_val,   y_val)
        self.test_data  = (X_test,  y_test)

    def train_dataloader(self):
        X, y = self.train_data
        transform = MFCCTrainAugmentation(
            probability=self.hparams.aug_probability,
            noise_std=self.hparams.aug_noise_std,
            time_mask_max=self.hparams.aug_time_mask_max,
            frequency_mask_max=self.hparams.aug_frequency_mask_max,
        )
        return self.create_data_loader(X, y, shuffle=True, transform=transform)

    def val_dataloader(self):
        X, y = self.val_data
        return self.create_data_loader(X, y, shuffle=False)

    def test_dataloader(self):
        X, y = self.test_data
        return self.create_data_loader(X, y, shuffle=False)

    def load_data(self):
        data = np.load(self.hparams.dataset_path, allow_pickle=False)
        return data["mfcc"], data["labels"], data["track_ids"]

    def split_data(self, inputs, targets, groups):
        val_test_size = self.hparams.test_size+self.hparams.validation_size

        first_split = GroupShuffleSplit(
                n_splits=1,
                test_size=val_test_size,
                random_state=self.hparams.random_seed,
            )

        train_indices, temporary_indices = next(first_split.split(inputs, targets, groups))
        temporary_groups = groups[temporary_indices]

        second_split = GroupShuffleSplit(
            n_splits=1,
            test_size=round(self.hparams.test_size/val_test_size, 2),
            random_state=self.hparams.random_seed,
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

    def normalize_data(self, X_train, X_val, X_test):
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True)
        train_inputs = (X_train - mean) / (std + 1e-8)
        validation_inputs = (X_val - mean) / (std + 1e-8)
        test_inputs = (X_test - mean) / (std + 1e-8)
        return train_inputs, validation_inputs, test_inputs, mean, std

    def create_data_loader(self,
            inputs,
            targets,
            shuffle: bool = False,
            transform=None
        ):
        batch_size = self.hparams.batch_size
        oversample_classes = self.hparams.oversample_classes
        oversample_factor = self.hparams.oversample_factor

        dataset = AudioFeatureDataset(inputs, targets, transform=transform)

        sampler = None
        if oversample_classes and oversample_factor > 1:
            sample_weights = torch.ones(len(targets), dtype=torch.double)
            target_array = np.asarray(targets)
            for class_index in oversample_classes:
                sample_weights[target_array == class_index] = oversample_factor
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )