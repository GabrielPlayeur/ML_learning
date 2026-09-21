import torch
from torch.utils.data import Dataset


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
