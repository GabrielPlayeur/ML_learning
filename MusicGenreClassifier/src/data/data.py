import numpy as np
import lightning as L
from sklearn.model_selection import train_test_split
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
            use_deltas: bool = False,
            oversample_classes: tuple = (),
            oversample_factor: float = 1.0,
            aug_probability: float = 0.5,
            aug_noise_std: float = 0.02,
            aug_time_mask_max: int = 8,
            aug_frequency_mask_max: int = 3,
            split_by_track: bool = True,
        ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        inputs, targets, groups = self.load_data()

        X_train, X_val, X_test, y_train, y_val, y_test, self.test_groups = self.split_data(inputs, targets, groups)

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

        inputs = data["mfcc"]
        if not self.hparams.use_deltas:
            inputs = inputs[:, :, :13]

        return inputs, data["labels"], data["track_ids"]

    def split_data(self, inputs, targets, groups):
        """Split par morceau (les 10 segments d'un morceau restent dans le même ensemble, pas de fuite)
        et stratifié par genre (chaque genre a la même part de morceaux dans train / val / test).
        split_by_track=False : contre-exemple volontaire, les segments sont répartis un par un et
        ceux d'un même morceau se retrouvent à la fois en train et en test (fuite)."""
        if self.hparams.split_by_track:
            units, first_index = np.unique(groups, return_index=True)
            unit_of_sample = groups
        else:
            units = first_index = unit_of_sample = np.arange(len(targets))
        unit_labels = targets[first_index]
        val_test_size = self.hparams.test_size + self.hparams.validation_size

        train_units, temporary_units, _, temporary_labels = train_test_split(
            units, unit_labels,
            test_size=val_test_size,
            stratify=unit_labels,
            random_state=self.hparams.random_seed,
        )
        validation_units, test_units = train_test_split(
            temporary_units,
            test_size=self.hparams.test_size / val_test_size,
            stratify=temporary_labels,
            random_state=self.hparams.random_seed,
        )

        train_indices = np.isin(unit_of_sample, train_units)
        validation_indices = np.isin(unit_of_sample, validation_units)
        test_indices = np.isin(unit_of_sample, test_units)

        return (
            inputs[train_indices],
            inputs[validation_indices],
            inputs[test_indices],
            targets[train_indices],
            targets[validation_indices],
            targets[test_indices],
            groups[test_indices],
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
        # shuffle n'est vrai que pour le train : val et test ne doivent jamais être rééchantillonnés
        if shuffle and oversample_classes and oversample_factor > 1:
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