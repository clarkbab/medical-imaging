import inspect
import numpy as np
import os
import pandas as pd
from typing import *

from mymi import cache
from mymi import config
from mymi import types

from ..types import types as ds_types

FILENAME_NUM_DIGITS = 5

class ProcessedDataset:
    def __init__(
        self,
        name: str):
        """
        args:
            name: the name of the dataset.
        """
        self._name = name
        self._path = os.path.join(config.directories.datasets, name, 'processed')

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{name}' not found.")

    def description(self) -> str:
        """
        returns: a short descriptive string.
        """
        # Create description.
        desc = f"Name: {self._name}, Type: PROCESSED"
        return desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> int:
        return ds_types.PROCESSED

    def manifest(
        self,
        folder: types.ProcessedFolder) -> Sequence[str]:
        """
        returns: a sequence of patient IDs for that folder.
        args:
            folder: read the manifest from this folder.
        """
        # Read manifest file.
        filepath = os.path.join(self._path, 'manifests', f"{folder}.csv")
        pats_df = pd.read_csv(filepath)
        pats = pats_df['patient-id'].tolist()

        return pats

    def input(
        self,
        folder: types.ProcessedFolder,
        sample_idx: int) -> np.ndarray:
        """
        returns: the input data for sample i.
        args:
            folder: 'train', 'test' or 'validation'.
            sample_idx: the sample index to load.
        """
        # Load the input data.
        filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-input"
        input_path = os.path.join(self._path, folder, filename)
        f = open(input_path, 'rb')
        input = np.load(f)
        return input

    def label(
        self,
        folder: types.ProcessedFolder,
        sample_idx: int) -> np.ndarray:
        """
        returns: the label data for sample i.
        args:
            folder: 'train', 'test' or 'validation'.
            sample_idx: the sample index to load.
        """
        # Load the label data.
        filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-label"
        label_path = os.path.join(self._path, folder, filename)
        f = open(label_path, 'rb')
        label = np.load(f)
        return label

    def sample(
        self,
        folder: types.ProcessedFolder,
        sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns: the (input, label) pair for the given sample index.
        args:
            folder: 'train', 'test' or 'validation'.
            sample_idx: the sample index to load.
        """
        # Get sample (input, label) pair.
        pair = self.input(folder, sample_idx), self.label(folder, sample_idx)
        return pair 

    # TODO: Move to subclass if necessary.
    @classmethod
    def class_frequencies(cls, folder):
        """
        returns: the frequencies for each class.
        args:
            folder: 'train', 'test', or 'validation'.
        """
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'folder': folder
            }
        }
        result = cache.read(params, 'array')
        if result is not None:
            return tuple(result)

        # Get number of samples.
        folder_path = os.path.join(cls.data_dir(), folder)
        num_samples = int(len(os.listdir(folder_path)) / 2)

        # Count background and foreground frequencies.
        num_back, num_fore = 0, 0
        for i in range(num_samples):
            # Load label.
            label_data = cls.label(folder, i)

            # Add values.
            shape = label_data.shape
            sum = label_data.sum()
            num_back += shape[0] * shape[1] * shape[2] - sum
            num_fore += sum

        # Create frequencies tuple.
        freqs = (num_back, num_fore)

        # Write data to cache.
        cache.write(params, freqs, 'array')

        return freqs
