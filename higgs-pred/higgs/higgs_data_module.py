import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as t
import torch.utils.data as td

logger = logging.getLogger(__name__)

COLNAMES = [
    "label",
    "lepton_pt",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet_1_pt",
    "jet_1_eta",
    "jet_1_phi",
    "jet_1_b-tag",
    "jet_2_pt",
    "jet_2_eta",
    "jet_2_phi",
    "jet_2_b-tag",
    "jet_3_pt",
    "jet_3_eta",
    "jet_3_phi",
    "jet_3_b-tag",
    "jet_4_pt",
    "jet_4_eta",
    "jet_4_phi",
    "jet_4_b-tag",
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]


class HiggsDataModule(pl.LightningDataModule):
    def __init__(
        self, trainfile=None, trainset_prop=0.9, testfile=None, hp=None,
    ):
        super().__init__()

        if trainfile and Path(trainfile).exists():
            self._trainfile = trainfile
        else:
            self._trainfile = None

        if testfile and Path(testfile).exists():
            self._testfile = testfile
        else:
            self._testfile = None

        if (not trainfile) and (not testfile):
            raise RuntimeError(
                "At least one valid trainfile or testfile must be specified."
            )

        # self.hparams = hp
        self.save_hyperparameters(hp)

        self._trainset_prop = trainset_prop

        self._trainpkl = Path("train.pkl")
        self._train_ds = None
        self._trainsize = 0

        self._valpkl = Path("val.pkl")
        self._val_ds = None
        self._valsize = 0

        self._testpkl = Path("test.pkl")
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
        cookie_pkl = Path("cookie.pkl")
        if cookie_pkl.exists():
            cookie = None
            with open(cookie_pkl, "rb") as f:
                cookie = pickle.load(f)
            if (
                cookie["trainfile"]
                and self._trainfile
                and cookie["trainfile"] == self._trainfile
            ):
                self._trainsize = cookie["trainsize"]
                self._valsize = cookie["valsize"]
                logger.info("Found pickled training data. Nothing to prepare.")
                return
            if (
                cookie["testfile"]
                and self._testfile
                and cookie["testfile"] == self._testfile
            ):
                logger.info("Found pickled test data. Nothing to prepare.")
                self.testsize = cookie["testsize"]
                return

        if self._trainfile:
            train_val_df = pd.read_csv(self._trainfile, header=None, names=COLNAMES)
            self._trainsize = int(len(train_val_df) * self._trainset_prop)

            train_df = train_val_df[: self._trainsize]
            self._pickle_df(train_df, self._trainpkl)

            val_df = train_val_df[self._trainsize :]
            self._valsize = len(val_df)
            self._pickle_df(val_df, self._valpkl)

        if self._testfile:
            test_df = pd.read_csv(self._testfile, header=None, names=COLNAMES)
            self._tstsize = len(test_df)
            self._pickle_df(test_df, self._testpkl)

        # Write the cookie
        cookie = {
            "trainfile": self._trainfile,
            "testfile": self._testfile,
            "trainsize": self._trainsize,
            "valsize": self._valsize,
            "testsize": self._testsize,
        }
        with open("cookie.pkl", "wb") as f:
            pickle.dump(cookie, f)
        logger.info("Pickled datasets for future use.")

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
