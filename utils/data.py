from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import constants
from . import file_utils


class OptionDataset(Dataset):
    """
    Abstract class for the option data

    Arguments:
        path: str
            Path to the pickle dataset.

        preprocess: callable or None, default: None
            Preprocessing to be applied to the dataframe.

        transform: callable or None, default: None
            Preprocessing to be applied online when querying
            the dataloader.
    """
    def __init__(self, path, categorical_features, numerical_features, output_feature, preprocess=None, transform=None):
        self.data = self._load_data(path, preprocess)
        self.y = self.data[output_feature].astype(np.float32).values  # .reshape(-1, 1)
        self.X_num = self.data[numerical_features].astype(np.float32).values
        self.X_cat = self.data[categorical_features].astype(np.int64).values
        self.transform = transform

    def _load_data(self, path, preprocess=None):
        data = file_utils.pickle2dataframe(path)
        if preprocess:
            data = preprocess(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], np.log(self.y[idx])

    @property
    def num_features_(self):
        return len(self[0][0])
