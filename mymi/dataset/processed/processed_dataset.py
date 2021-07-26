import gzip
import inspect
import nibabel as nib
import numpy as np
import os
import pandas as pd
from typing import *

from mymi import cache
from mymi import config
from mymi import types

from ..dataset import Dataset, DatasetType

FILENAME_NUM_DIGITS = 5

class ProcessedDataset(Dataset):
    def __init__(
        self,
        name: str):
        """
        args:
            name: the name of the dataset.
        """
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'processed', name)
        self._folders = ['train', 'validation', 'test']

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self.description}' not found.")

    @property
    def description(self) -> str:
        return f"PROCESSED: {self._name}"

    @property
    def folders(self) -> List[str]:
        return self._folders

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def type(self) -> DatasetType:
        return DatasetType.PROCESSED

    def manifest(
        self,
        folder: types.ProcessedFolder) -> List[str]:
        """
        returns: a sequence of patient IDs for that folder.
        args:
            folder: read the manifest from this folder.
        """
        # Read manifest file.
        filepath = os.path.join(self._path, 'manifests', f"{folder}.csv")
        df = pd.read_csv(filepath)
        return df

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

    def create_input(
        self,
        id: str,
        data: np.ndarray,
        folder: str) -> int:
        """
        effect: creates an input sample.
        returns: the index of the sample.
        args:
            id: the object ID.
            data: the data to save.
            folder: the folder to save to.
        """
        # Get next index.
        input_path = os.path.join(self._path, 'data', folder, 'inputs')
        if os.path.exists(input_path):
            inputs = os.listdir(input_path)
            if len(inputs) == 0:
                index = -1
            else:
                index = int(list(sorted(inputs))[-1].replace('.npz', '')) + 1
        else:
            os.makedirs(input_path)
            index = 0

        # Save the input data.
        filename = f"{index:0{FILENAME_NUM_DIGITS}}.npz"
        filepath = os.path.join(input_path, filename)
        np.savez_compressed(filepath, data=data)

        # Update the manifest.
        self._append_to_manifest(folder, index, id)

        return index

    def create_label(
        self,
        index: int,
        region: str,
        label: np.ndarray,
        folder: str) -> None:
        """
        effect: creates an input sample.
        args:
            pat_id: the patient ID to add to the manifest.
            region: the region name.
            label: the label data.
            folder: the folder to save to.
        """
        # Create region folder if it doesn't exist.
        region_path = os.path.join(self._path, 'data', folder, 'labels', region)
        if not os.path.exists(region_path):
            os.makedirs(region_path)

        # Save label.
        filename = f"{index:0{FILENAME_NUM_DIGITS}}.npz"
        filepath = os.path.join(region_path, filename)
        f = open(filepath, 'wb')
        np.savez_compressed(f, label=label)

    def _append_to_manifest(
        self,
        folder: str,
        index: int,
        id: str) -> None:
        """
        effect: adds a line to the manifest.
        """
        # Create manifest if not present.
        manifest_path = os.path.join(self._path, 'manifest.csv')
        if not os.path.exists(manifest_path):
            with open(manifest_path, 'w') as f:
                f.write('folder,id,index\n')

        # Append line to manifest. 
        with open(manifest_path, 'a') as f:
            f.write(f"{folder},{id},{index}\n")
