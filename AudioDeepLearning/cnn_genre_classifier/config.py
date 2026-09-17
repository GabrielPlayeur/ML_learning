from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
	dataset_path: str = "data/data.json"
	batch_size: int = 32
	learning_rate: float = 3e-4
	weight_decay: float = 1e-3
	epochs: int = 100
	patience: int = 5
	min_delta: float = 1e-4
	dropout: float = 0.3
	hidden_layers: tuple[int, ...] = (256,64)
	random_seed: int = 42
	output_path: str = "models/cnn_model.pt"
	output_curves_path: str = "curves/cnn_reduceLR_para_W.png"
