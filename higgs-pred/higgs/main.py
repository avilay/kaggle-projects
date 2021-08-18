import warnings
import os
from datetime import datetime

import click
import pandas as pd
import pytorch_lightning as pl
from haikunator import Haikunator
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

from higgs.higgs_classifier import HiggsClassifier
from higgs.higgs_data_module import HiggsDataModule

from .consts import (
    COLNAMES,
    DATAROOT,
    HIGGS_CSV,
    HIGGS_TRAIN_CSV,
    HIGGS_TEST_CSV,
    RUNROOT,
    SAMPLE_HIGGS_CSV,
    PROJECT,
)

warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=".*data_loading.*")


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--infile",
    default=DATAROOT / PROJECT / HIGGS_CSV,
    help="Input file from which to draw a sample.",
    show_default=True,
)
@click.option(
    "--outfile",
    default=DATAROOT / PROJECT / SAMPLE_HIGGS_CSV,
    help="Output file in which sample will be saved.",
    show_default=True,
)
@click.option("--size", default=1000, help="Sample size to create.", show_default=True)
def sample(infile, outfile, size):
    """
    Samples the specified number of instances from the full dataset and saves the sample to another file.
    """
    df = pd.read_csv(infile, header=None, names=COLNAMES)
    df.sample(size).to_csv(outfile, header=None, index=None)
    print(f"Sampled {size:,} examples from {infile} and saved to {outfile}")


@main.command()
@click.option(
    "--infile",
    default=DATAROOT / PROJECT / HIGGS_CSV,
    help="The full data file.",
    show_default=True,
)
@click.option(
    "--trainfile",
    default=DATAROOT / PROJECT / HIGGS_TRAIN_CSV,
    help="Output filename of the training set.",
    show_default=True,
)
@click.option(
    "--testfile",
    default=DATAROOT / PROJECT / HIGGS_TEST_CSV,
    help="Output filename of the test set.",
    show_default=True,
)
@click.option(
    "--frac",
    default=0.9,
    help="Fraction of the full dataset that will be in the training set.",
    show_default=True,
)
def split(infile, trainfile, testfile, frac):
    """
    Splits the full dataset into a training set and the validation set. During training the training set should be
    further split into a training set and a validation set.
    """
    higgs = pd.read_csv(infile, header=None, names=COLNAMES)
    train_sz = int(len(higgs) * frac)
    shuffled = higgs.sample(frac=1.0)
    train_df = shuffled[:train_sz]
    test_df = shuffled[train_sz:]
    train_df.to_csv(trainfile, header=None, index=None)
    test_df.to_csv(testfile, header=None, index=None)
    print(
        f"Split {infile} into {trainfile} containing {len(train_df)} rows and {testfile} containing {len(test_df)} rows."
    )


@main.command()
@click.option(
    "--dataroot",
    default=DATAROOT,
    help="Local directory of data files.",
    show_default=True,
)
@click.option(
    "--project", default=PROJECT, help="Name of this project.", show_default=True
)
@click.option(
    "--trainfile",
    default=HIGGS_TRAIN_CSV,
    help="File containing the training dataset.",
    show_default=True,
)
@click.option(
    "--frac",
    default=0.9,
    help="Fraction of data that will be in the main training set.",
    show_default=True,
)
@click.option(
    "--runroot",
    default=RUNROOT,
    help="Local directory where wandg logs will be saved.",
    show_default=True,
)
@click.option(
    "--name",
    default=Haikunator().haikunate(),
    help="Unique run name.",
    show_default=True,
)
@click.option(
    "--hparams",
    default="higgs.yml",
    help="Hyperparameters yml file.",
    show_default=True,
)
def train(dataroot, project, trainfile, frac, runroot, name, hparams):
    """
    Trains the classifier.
    """
    print(f"Starting run {project}/{name}")
    hparams = OmegaConf.load(hparams)
    model = HiggsClassifier(hp=hparams)
    data = HiggsDataModule(
        dataroot=dataroot,
        project=project,
        trainfile=trainfile,
        trainset_prop=frac,
        hp=hparams,
    )
    data.prepare()
    print(f"Train set size: {data.trainsize}, Validation set size: {data.valsize}")

    os.makedirs(runroot / project, exist_ok=True)

    logger = WandbLogger(
        project=project, name=name, save_dir=runroot / project, log_model="all", id=name
    )
    logger.watch(model, log="all")

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")

    start = datetime.now()
    trainer = pl.Trainer(
        default_root_dir=runroot / project,
        max_epochs=hparams.n_epochs,
        logger=logger,
        callbacks=[checkpoint],
    )
    trainer.fit(model, data)
    end = datetime.now()
    print(f"Took {end - start} to finish training.")


if __name__ == "__main__":
    main()
