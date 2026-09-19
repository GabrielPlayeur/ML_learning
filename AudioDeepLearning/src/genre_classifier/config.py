from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
	dataset_path: str = "data/data.json"
	batch_size: int = 32
	learning_rate: float = 3e-4
	weight_decay: float = 3e-4
	epochs: int = 100
	patience: int = 7
	min_delta: float = 1e-4
	dropout: float = 0.3
	random_seed: int = 42
	train_augmentation: bool = False
	augmentation_probability: float = 0.5
	augmentation_noise_std: float = 0.02
	augmentation_time_mask_max: int = 8
	augmentation_frequency_mask_max: int = 3
	oversample_classes: tuple[int, ...] = ()
	oversample_factor: float = 1.0


@dataclass(frozen=True)
class MLPConfig(TrainingConfig):
	hidden_layers: tuple[int, ...] = (256, 64)
	output_path: str = "models/mlp_model.pt"
	output_curves_path: str = "curves/mlp.png"
	output_report_path: str = "curves/mlp_report.json"
	output_confusion_path: str = "curves/mlp_confusion.png"
	output_roc_path: str = "curves/mlp_roc.png"


@dataclass(frozen=True)
class CNNConfig(TrainingConfig):
	output_path: str = "models/cnn_model.pt"
	channels: tuple[int, ...] = (32, 64, 128)
	output_curves_path: str = "curves/cnn.png"
	output_report_path: str = "curves/cnn_report.json"
	output_confusion_path: str = "curves/cnn_confusion.png"
	output_roc_path: str = "curves/cnn_roc.png"
	train_augmentation: bool = True
	oversample_classes: tuple[int, ...] = (5, 8, 9)
	oversample_factor: float = 1.5


@dataclass(frozen=True)
class RNNLSTMConfig(TrainingConfig):
	output_path: str = "models/rnn_model.pt"
	hidden_layers: tuple[int, ...] = (64, 64)
	output_curves_path: str = "curves/rnn.png"
	output_report_path: str = "curves/rnn_report.json"
	output_confusion_path: str = "curves/rnn_confusion.png"
	output_roc_path: str = "curves/rnn_roc.png"