import os.path as path

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import json
import os

import consts


def download_from_gcp_storage():
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set!")
    storage_client = storage.Client(project="avilabs")


class PlantPathologyDataset(Dataset):
    def __init__(self, imgroot, transformer=None):
        self._xform = transformer
        self._imgroot = imgroot

        train_df = pd.read_csv(path.join(consts.DATAROOT, "train.csv"))
        self._image_ids = []
        self._labels = []

        for row in train_df.itertuples():
            self._image_ids.append(row.image_id)
            if row.rust == 1:
                self._labels.append(consts.RUST)
            elif row.scab == 1:
                self._labels.append(consts.SCAB)
            elif row.healthy == 1:
                self._labels.append(consts.HEALTHY)
            elif row.multiple_diseases:
                self._labels.append(consts.MULTIPLE_DISEASES)
            else:
                raise RuntimeError(f"Unknown label!! {row}")

    def __getitem__(self, idx):
        img_file = f"{self._image_ids[idx]}.jpg"
        img_path = path.join(self._imgroot, img_file)
        img = Image.open(img_path)
        if self._xform is not None:
            img = self._xform(img)
        return img, self._labels[idx]

    def __len__(self):
        return len(self._labels)


class PlantPathologyTestDataset(Dataset):
    def __init__(self, imgroot, transformer=None):
        self._xform = transformer
        self._imgroot = imgroot

        test_df = pd.read_csv(path.join(consts.DATAROOT, "test.csv"))
        self.image_ids = []
        for row in test_df.itertuples():
            self.image_ids.append(row.image_id)

        self.image_ids = self.image_ids[:10]

    def __getitem__(self, idx):
        img_file = f"{self.image_ids[idx]}.jpg"
        img_path = path.join(self._imgroot, img_file)
        img = Image.open(img_path)
        if self._xform is not None:
            img = self._xform(img)
        return img, 0

    def __len__(self):
        return len(self.image_ids)
