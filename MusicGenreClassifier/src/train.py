import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch

from .data import GTZANDataModule
from .models import CNNGenreClassifier
from .module import GenreClassifierModule

def main():
    datamodule = GTZANDataModule(
        dataset_path="dataset/mfcc.npz",
        batch_size=32,
        test_size=0.15,
        validation_size=0.15,
    )

    backbone = CNNGenreClassifier(num_classes=10, channels=(32,64,128))

    model = GenreClassifierModule(
        model=backbone,
        num_classes=10,
        learning_rate=3e-4,
        weight_decay=3e-4,
    )

    callbacks = [
        EarlyStopping(monitor="val/loss", patience=7, min_delta=1e-4),
        ModelCheckpoint(monitor="val/loss", save_top_k=1, mode="min"),
    ]

    logger = WandbLogger(
        project="MusicGenreClassifier",
        entity="playeur-audio-team",
        )

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()