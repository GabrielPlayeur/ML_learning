import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, AUROC
from torchmetrics.classification import MulticlassConfusionMatrix

class GenreClassifierModule(L.LightningModule):
    def __init__(self,
            model: nn.Module,
            num_classes: int = 10,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            scheduler_patience: int = 4,
            scheduler_factor: float = 0.5,
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss = nn.CrossEntropyLoss()

        metrics_kwargs = {"task": "multiclass", "num_classes": num_classes}
        self.train_acc = Accuracy(**metrics_kwargs)
        self.val_acc   = Accuracy(**metrics_kwargs)
        self.val_f1    = F1Score(**metrics_kwargs, average="macro")
        self.val_auroc = AUROC(**metrics_kwargs)
        self.test_acc  = Accuracy(**metrics_kwargs)
        self.test_f1   = F1Score(**metrics_kwargs, average="macro")
        self.test_auroc = AUROC(**metrics_kwargs)
        self.test_cm   = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)
        self.val_acc(preds, y)
        # self.val_f1(preds, y)
        # self.val_auroc(probs, y)
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc",  self.val_acc,   on_epoch=True)
        # self.log("val/f1",   self.val_f1,    on_epoch=True)
        # self.log("val/auroc", self.val_auroc, on_epoch=True)

    def on_test_epoch_start(self):
        self._test_probs = []
        self._test_targets = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)
        self.test_cm(preds, y)
        self._test_probs.append(probs.detach().cpu())
        self._test_targets.append(y.detach().cpu())
        self.log("test/loss",  loss,            on_epoch=True)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

    def on_test_epoch_end(self):
        # Courbes tracées une seule fois, sur l'ensemble du test set, plutot
        # que recalculees a chaque epoque de validation.
        probs = torch.cat(self._test_probs).numpy()
        targets = torch.cat(self._test_targets).numpy()
        preds = probs.argmax(axis=1).tolist()

        import wandb
        wandb.log({
            "test/roc_curve": wandb.plot.roc_curve(targets.tolist(), probs.tolist()),
            "test/pr_curve": wandb.plot.pr_curve(targets.tolist(), probs.tolist()),
            "test/confusion_matrix": wandb.plot.confusion_matrix(y_true=targets.tolist(), preds=preds),
        })
        self._test_probs.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
