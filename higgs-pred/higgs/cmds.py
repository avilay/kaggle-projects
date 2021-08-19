import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from haikunator import Haikunator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from higgs.higgs_data_module import COLNAMES

from .higgs_classifier import HiggsClassifier
from .higgs_data_module import HiggsDataModule

logger = logging.getLogger(__name__)


def train(cfg):
    """
    Trains the classifier.
    """
    if cfg.name == "auto":
        cfg.name = Haikunator().haikunate()
    train_csv = Path(cfg.dataroot) / cfg.train_csv
    logger.info(f"Starting run {cfg.name}")
    model = HiggsClassifier(hp=cfg.hparams)
    data = HiggsDataModule(
        trainfile=train_csv, trainset_prop=cfg.train_val_split_frac, hp=cfg.hparams,
    )
    data.prepare()
    logger.info(
        f"Train set size: {data.trainsize}, Validation set size: {data.valsize}"
    )
    os.makedirs(cfg.runroot, exist_ok=True)
    ml_logger = WandbLogger(
        project="higgs",
        name=cfg.name,
        save_dir=cfg.runroot,
        log_model="all",
        id=cfg.name,
    )
    ml_logger.watch(model, log="all")

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    start = datetime.now()
    trainer = Trainer(
        default_root_dir=cfg.runroot,
        max_epochs=cfg.hparams.n_epochs,
        logger=ml_logger,
        callbacks=[checkpoint],
    )
    trainer.fit(model, data)
    end = datetime.now()
    logger.info(f"Took {end - start} to finish training.")


def split(cfg):
    """
    Splits the specified data file into a training set and the validation set. During training the training set should
    be further split into a training set and a validation set.
    """
    df = pd.read_csv(cfg.infilepath, header=None, names=COLNAMES)
    df = df.sample(frac=1.0)
    train_csv = Path(cfg.path) / cfg.train_csv
    test_csv = Path(cfg.path) / cfg.test_csv
    train_sz = int(len(df) * cfg.train_split_frac)
    train_df = df[:train_sz]
    test_df = df[train_sz:]
    logger.info(
        f"Splitting {cfg.infilepath} into {train_csv} with {len(train_df)} instances and {test_csv} with {len(test_df)} instances."
    )
    train_df.to_csv(train_csv, header=None, index=None)
    test_df.to_csv(test_csv, header=None, index=None)


def sandbox(cfg):
    """
    Samples some instances from the main data file into another specified file.
    Splits the new file into train and test files.
    """
    data_csv = Path(cfg.path) / cfg.data_csv
    logger.info(
        f"Creating new file {data_csv} with {cfg.size} instances drawn from {cfg.infilepath}."
    )
    df = pd.read_csv(cfg.infilepath, header=None, names=COLNAMES)
    sample_df = df.sample(cfg.size)
    os.makedirs(cfg.path, exist_ok=True)
    sample_df.to_csv(data_csv, header=None, index=None)
    cfg.infilepath = str(data_csv)
    split(cfg)


def version(cfg):
    print("v.0.0.1")
