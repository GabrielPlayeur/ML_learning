import torch

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
