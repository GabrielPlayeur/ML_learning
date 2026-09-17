from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
	dataset_path: str = "data/data.json"
	batch_size: int = 32
	learning_rate: float = 1e-4
	weight_decay: float = 1e-4
	epochs: int = 100
	patience: int = 5
	min_delta: float = 1e-4
	dropout: float = 0.3
	random_seed: int = 42


@dataclass(frozen=True)
class MLPConfig(TrainingConfig):
	hidden_layers: tuple[int, ...] = (256, 64)
	output_path: str = "models/mlp_model.pt"
	output_curves_path: str = "curves/mlp.png"


@dataclass(frozen=True)
class CNNConfig(TrainingConfig):
	output_path: str = "models/cnn_model.pt"
	channels: tuple[int, ...] = (128, 64, 32)
	output_curves_path: str = "curves/cnn.png"
