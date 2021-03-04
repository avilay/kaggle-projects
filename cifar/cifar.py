import os.path as path
from dataclasses import dataclass

import torch as t
import torch.utils.data as td
import torchvision as tv
from haikunator import Haikunator
from sklearn.metrics import accuracy_score

from torchutils import Hyperparams, Trainer, TrainerArgs, Tuner, HyperparamsSpec
from torchutils.ml_loggers.csv_logger import CsvMLExperiment

from .model_factory import build_cifar_model

import click

DATAROOT = path.join("~", "mldata")


def prep_datasets():
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_val_set = tv.datasets.CIFAR10(DATAROOT, train=True, download=True, transform=xform)
    testset = tv.datasets.CIFAR10(DATAROOT, train=False, download=True, transform=xform)
    train_size = int(len(train_val_set) * 0.9)
    val_size = len(train_val_set) - train_size
    trainset, valset = td.random_split(train_val_set, (train_size, val_size))
    return trainset, valset, testset


@dataclass
class MyHyperparams(Hyperparams):
    batch_size: int
    n_epochs: int
    lr: float
    model_type: str


def accuracy(y_true, y_hat):
    y_pred = t.argmax(y_hat, dim=1)
    return accuracy_score(y_true, y_pred)


def build_trainer(hparams, trainset, valset):
    run_name = Haikunator().haikunate()
    print(f"Starting run {run_name}")

    model, tunable_params = build_cifar_model(hparams.model_type)
    optim = t.optim.Adam(tunable_params, lr=hparams.lr)
    scheduler = t.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.5)
    loss_fn = t.nn.CrossEntropyLoss()

    traindl = td.DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True)
    valdl = td.DataLoader(valset, batch_size=50)

    return TrainerArgs(
        run_name=run_name,
        model=model,
        optimizer=optim,
        lr_scheduler=scheduler,
        loss_fn=loss_fn,
        trainloader=traindl,
        valloader=valdl,
        n_epochs=hparams.n_epochs,
    )


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-b",
    "--batch-size",
    prompt=True,
    default=32,
    type=int,
    help="The mini-batch size used in each epoch.",
)
@click.option(
    "-n",
    "--n-epochs",
    prompt=True,
    default=5,
    type=int,
    help="The number of epochs to run the training for.",
)
@click.option(
    "-a",
    "--learning-rate",
    prompt=True,
    default=0.001,
    type=float,
    help="The learning rate used by the Adam optimizer.",
)
@click.option(
    "-t",
    "--model-type",
    type=click.Choice(["tune-1", "tune-2", "tune-classifier"], case_sensitive=False),
    prompt=True,
    default="tune-1",
    help="The model type to use for training.",
)
def train(batch_size, n_epochs, learning_rate, model_type):
    """
    Trains a VGG11 model on the CIFAR-10 dataset.
    """
    exproot = path.join("~", "temp", "experiments")
    hparams = MyHyperparams(
        n_epochs=n_epochs, lr=learning_rate, batch_size=batch_size, model_type=model_type
    )
    trainset, valset, _ = prep_datasets()
    exp = CsvMLExperiment("cifar-10-1", exproot, stdout=True)
    trainer = Trainer(exp, trainset, valset, [accuracy])
    trainer.metrics_log_frequency = 1  # Log every epoch's metrics
    trainer.params_log_frequency = 5  # Log every 5th epoch's parameter histogram
    trainer.model_log_frequency = 10000  # Don't save the model ever
    trainer.train(hparams, build_trainer)


@main.command()
@click.argument("model_type")
def tune(model_type):
    """
    Tunes the hyperparameter of VGG11 on CIFAR-10.
    """
    if model_type == "tune-1":
        hparams_spec = HyperparamsSpec(
            factory=MyHyperparams,
            spec=[
                {
                    "name": "batch_size",
                    "type": "choice",
                    "value_type": "int",
                    "values": [8, 16, 32, 64, 128],
                },
                {"name": "n_epochs", "type": "range", "value_type": "int", "bounds": [5, 20]},
                {"name": "lr", "type": "range", "bounds": [1e-8, 1e-2], "log_scale": True},
                {"name": "model_type", "type": "fixed", "value_type": "str", "value": "tune-1"},
            ],
        )
    elif model_type == "tune-2":
        hparams_spec = HyperparamsSpec(
            factory=MyHyperparams,
            spec=[
                {
                    "name": "batch_size",
                    "type": "choice",
                    "value_type": "int",
                    "values": [8, 16, 32, 64],
                },
                {"name": "n_epochs", "type": "range", "value_type": "int", "bounds": [5, 20]},
                {"name": "lr", "type": "range", "bounds": [1e-9, 1e-3], "log_scale": True},
                {"name": "model_type", "type": "fixed", "value_type": "str", "value": "tune-2"},
            ],
        )
    elif model_type == "tune-classifier":
        hparams_spec = HyperparamsSpec(
            factory=MyHyperparams,
            spec=[
                {
                    "name": "batch_size",
                    "type": "choice",
                    "value_type": "int",
                    "values": [8, 16, 32, 64],
                },
                {"name": "n_epochs", "type": "range", "value_type": "int", "bounds": [10, 50]},
                {"name": "lr", "type": "range", "bounds": [1e-10, 1e-4], "log_scale": True},
                {
                    "name": "model_type",
                    "type": "fixed",
                    "value_type": "str",
                    "value": "tune-classifier",
                },
            ],
        )

    trainset, valset, _ = prep_datasets()
    exproot = path.join("~", "temp", "experiments")
    exp = CsvMLExperiment("cifar-10-tune-1", exproot, stdout=False)
    tuner = Tuner(exp, trainset, valset, accuracy)
    tuner.metrics_log_frequency = 1
    best_params = tuner.tune(hparams_spec, build_trainer)
    print("\n---------------------")
    print("BEST PARAM")
    print(best_params)


if __name__ == "__main__":
    main()
