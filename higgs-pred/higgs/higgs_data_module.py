import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as t
import torch.utils.data as td

from .consts import COLNAMES, DATAROOT, HIGGS_TEST_CSV, HIGGS_TRAIN_CSV, PROJECT


class HiggsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataroot=DATAROOT,
        project=PROJECT,
        trainfile=HIGGS_TRAIN_CSV,
        trainset_prop=0.9,
        testfile=HIGGS_TEST_CSV,
        hp=None,
    ):
        super().__init__()

        if trainfile:
            self._trainfile = dataroot / project / trainfile
            if not self._trainfile.exists():
                raise RuntimeError(f"{self._trainfile} does not exist!")
        else:
            self._trainfile = None

        if testfile:
            self._testfile = dataroot / project / testfile
            if not self._testfile.exists():
                raise RuntimeError(f"{self._testfile} does not exist!")
        else:
            self._testfile = None

        self.hparams = hp

        self._trainset_prop = trainset_prop

        self._trainpkl = dataroot / project / "train.pkl"
        self._train_ds = None
        self._trainsize = 0

        self._valpkl = dataroot / project / "val.pkl"
        self._val_ds = None
        self._valsize = 0

        self._testpkl = dataroot / project / "test.pkl"
        self._test_ds = None
        self._testsize = 0

    def _pickle_df(self, df, picklefile):
        X = df[COLNAMES[1:]].values.astype(np.float32)
        y = df["label"].values.astype(int)
        with open(picklefile, "wb") as f:
            pickle.dump({"X": X, "y": y}, f)

    def _create_ds(self, picklefile):
        with open(picklefile, "rb") as f:
            dataset = pickle.load(f)
            return td.TensorDataset(
                t.from_numpy(dataset["X"]).to(t.float32),
                t.from_numpy(dataset["y"]).to(t.int),
            )

    @property
    def trainsize(self):
        return self._trainsize

    @property
    def valsize(self):
        return self._valsize

    @property
    def testsize(self):
        return self._testsize

    def prepare(self):
        if self._trainpkl.exists() and self._valpkl.exists() and self._testpkl.exists():
            with open(self._trainpkl, "rb") as f:
                dataset = pickle.load(f)
                self._trainsize = dataset["X"].shape[0]

            with open(self._valpkl, "rb") as f:
                dataset = pickle.load(f)
                self._valsize = dataset["X"].shape[0]

            with open(self._testpkl, "rb") as f:
                dataset = pickle.load(f)
                self._testsize = dataset["X"].shape[0]
            return

        train_val_df = pd.read_csv(self._trainfile, header=None, names=COLNAMES)
        self._trainsize = int(len(train_val_df) * self._trainset_prop)

        train_df = train_val_df[: self._trainsize]
        self._pickle_df(train_df, self._trainpkl)

        val_df = train_val_df[self._trainsize :]
        self._valsize = len(val_df)
        self._pickle_df(val_df, self._valpkl)

        test_df = pd.read_csv(self._testfile, header=None, names=COLNAMES)
        self._tstsize = len(test_df)
        self._pickle_df(test_df, self._testpkl)

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self._train_ds = self._create_ds(self._trainpkl)
            self._val_ds = self._create_ds(self._valpkl)
        elif stage == "test":
            self._test_ds = self._create_ds(self.testpkl)

    def train_dataloader(self):
        return td.DataLoader(
            self._train_ds, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return td.DataLoader(self._val_ds, batch_size=5000)

    def test_dataloader(self):
        return td.DataLoader(self._test_ds, batch_size=5000)
