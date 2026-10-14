import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .data import GTZANDataModule
from .models import build_backbone
from .module import GenreClassifierModule

N_FRAMES = 130 # 3 s de signal à 22 050 Hz avec hop_length = 512

def fit(cfg: DictConfig, logger, progress_bar: bool = True):
    """Entraîne puis teste le modèle décrit par cfg. Renvoie (model, datamodule, trainer)."""
    L.seed_everything(cfg.seed)

    datamodule = GTZANDataModule(
        dataset_path=cfg.data.dataset_path,
        batch_size=cfg.data.batch_size,
        test_size=cfg.data.test_size,
        validation_size=cfg.data.validation_size,
        random_seed=cfg.data.random_seed,
        use_deltas=cfg.data.use_deltas,
        oversample_classes=tuple(cfg.data.oversample_classes),
        oversample_factor=cfg.data.oversample_factor,
        aug_probability=cfg.data.aug_probability,
        aug_noise_std=cfg.data.aug_noise_std,
        aug_time_mask_max=cfg.data.aug_time_mask_max,
        aug_frequency_mask_max=cfg.data.aug_frequency_mask_max,
        split_by_track=cfg.data.split_by_track,
    )

    n_coefficients = 3 * cfg.data.n_mfcc if cfg.data.use_deltas else cfg.data.n_mfcc
    backbone = build_backbone(cfg.model, input_shape=(N_FRAMES, n_coefficients))

    model = GenreClassifierModule(
        model=backbone,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        scheduler_patience=cfg.model.scheduler_patience,
        scheduler_factor=cfg.model.scheduler_factor,
    )

    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            min_delta=cfg.training.early_stopping_min_delta,
        ),
        ModelCheckpoint(
            monitor="val/loss",
            save_top_k=1,
            mode="min",
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_progress_bar=progress_bar,
        enable_model_summary=progress_bar,
    )

    trainer.fit(model, datamodule)
    # test sur les poids du meilleur epoch (val/loss minimale), pas sur ceux du dernier
    trainer.test(model, datamodule, ckpt_path="best")
    return model, datamodule, trainer


@hydra.main(version_base=None,
            config_path="configs",
            config_name="config")
def train(cfg: DictConfig):
    logger = WandbLogger(
        project="MusicGenreClassifier",
        entity="playeur-audio-team",
        name=cfg.get("wandb", {}).get("run_name", None),
        config=dict(cfg)
        )
    fit(cfg, logger)

if __name__ == "__main__":
    train()
