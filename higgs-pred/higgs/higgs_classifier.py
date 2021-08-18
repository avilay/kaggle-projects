import pytorch_lightning as pl
import torch as t
import torchmetrics as tm


class HiggsClassifier(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.fc1 = t.nn.Linear(28, 128)
        self.fc2 = t.nn.Linear(128, 256)
        self.fc3 = t.nn.Linear(256, 128)
        self.fc4 = t.nn.Linear(128, 64)
        self.fc5 = t.nn.Linear(64, 32)
        self.fc6 = t.nn.Linear(32, 1)

        self.loss_fn = t.nn.BCEWithLogitsLoss()

        self.save_hyperparameters(hp)

    def accuracy(self, logits, y):
        probs = t.sigmoid(logits)
        return tm.functional.accuracy(probs, y, threshold=self.hparams.true_cutoff)

    def forward(self, x):
        x = t.nn.ReLU()(self.fc1(x))
        x = t.nn.ReLU()(self.fc2(x))
        x = t.nn.ReLU()(self.fc3(x))
        x = t.nn.ReLU()(self.fc4(x))
        x = t.nn.ReLU()(self.fc5(x))
        logits = self.fc6(x)
        return t.squeeze(logits, dim=1)

    def _step(self, batch):
        X, y = batch
        logits = self(X)
        loss = self.loss_fn(logits, y.to(t.float32))
        acc = self.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train_loss_step", loss)
        self.log_dict(
            {"train_acc": acc, "train_loss": loss}, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_optimizers(self):
        optim = t.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optim
