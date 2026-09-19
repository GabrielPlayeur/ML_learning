from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
	dataset_path: str = "data/data.json"
	batch_size: int = 32
	learning_rate: float = 3e-4
	weight_decay: float = 1e-4
	epochs: int = 100
	patience: int = 7
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
	channels: tuple[int, ...] = (32, 64, 128)
	output_curves_path: str = "curves/cnn_big_pad_poolInt_lr.png"


@dataclass(frozen=True)
class RNNLSTMConfig(TrainingConfig):
	output_path: str = "models/rnn_model.pt"
	hidden_layers: tuple[int, ...] = (64, 64)
	output_curves_path: str = "curves/rnn.png"