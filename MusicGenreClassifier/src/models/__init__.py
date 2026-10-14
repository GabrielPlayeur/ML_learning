from .cnn import CNNGenreClassifier
from .mlp import MLPGenreClassifier
from .lstm import LSTMGenreClassifier


def build_backbone(model_cfg, input_shape: tuple[int, int]):
    """model_cfg.name choisit l'architecture ; input_shape = (frames, coefficients)."""
    if model_cfg.name == "mlp":
        return MLPGenreClassifier(
            input_shape=input_shape,
            hidden_layers=tuple(model_cfg.hidden_layers),
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
        )
    if model_cfg.name == "cnn":
        return CNNGenreClassifier(
            channels=tuple(model_cfg.channels),
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
        )
    if model_cfg.name == "lstm":
        return LSTMGenreClassifier(
            input_shape=input_shape,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers,
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
            pooling=model_cfg.pooling,
        )
    raise ValueError(f"Modèle inconnu : {model_cfg.name}")
