from torch.utils.data import Dataset
import pandas as pd

import constants
from . import file_utils


class OptionDataset(Dataset):
    """
    Abstract class for the option

    Args
        path: (string) path to the pickle dataset
    """
    def __init__(self, path, preprocess=None, transform=None):
        self.data = self._load_data(path)
        self.transform = transform

    def _load_data(self, path, preprocess=None):
        data = file_utils.pickle2dataframe(path)
        if preprocess:
            data = preprocess(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Input processing
        input = self.data.drop(constants.OUTPUT_FEATURE, axis=1)[idx]
        input = input if not self.transform else self.transform(input)

        # Target processing
        target = self.data[constants.OUTPUT_FEATURE][idx]

        return input, target
