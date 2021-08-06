import numpy as np
import os
import pandas as pd
from typing import List

from mymi import cache
from mymi import types

from .partition_sample import PartitionSample

FILENAME_NUM_DIGITS = 5

class ProcessedPartition:
    def __init__(
        self,
        dataset: 'ProcessedDataset',
        name: types.ProcessedPartition):
        """
        args:
            dataset: the dataset name.
            name: the partition name.
        """
        self._dataset = dataset
        self._name = name
        self._path = os.path.join(dataset.path, 'data', name)

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Partition '{name}' not found for dataset '{dataset.name}'.")

    def list_samples(self) -> List[int]:
        """
        returns: the sample indices.
        """
        path = os.path.join(self._path, 'inputs')
        if os.path.exists(path):
            indices = [int(f.replace('.npz', '')) for f in os.listdir(path)]
        else:
            indices = []
        return indices

    def list_regions(self) -> List[str]:
        """
        returns: the region names. 
        """
        path = os.path.join(self._path, 'labels')
        return os.listdir(path)

    def sample(
        self,
        index: int) -> PartitionSample:
        """
        returns: the partition sample.
        """
        return PartitionSample(self, index)

    def create_input(
        self,
        id: str,
        data: np.ndarray) -> int:
        """
        effect: creates an input sample.
        returns: the index of the sample.
        args:
            id: the object ID.
            data: the data to save.
        """
        # Get next index.
        path = os.path.join(self._path, 'inputs')
        if os.path.exists(path):
            inputs = os.listdir(path)
            if len(inputs) == 0:
                index = -1
            else:
                index = int(list(sorted(inputs))[-1].replace('.npz', '')) + 1
        else:
            os.makedirs(path)
            index = 0

        # Save the input data.
        filename = f"{index:0{FILENAME_NUM_DIGITS}}.npz"
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath, data=data)

        # Update the manifest.
        self._dataset.append_to_manifest(self._name, index, id)

        return index

    def create_label(
        self,
        index: int,
        region: str,
        data: np.ndarray) -> None:
        """
        effect: creates an input sample.
        args:
            pat_id: the patient ID to add to the manifest.
            region: the region name.
            data: the label data.
        """
        # Create region partition if it doesn't exist.
        region_path = os.path.join(self._path, 'labels', region)
        if not os.path.exists(region_path):
            os.makedirs(region_path)

        # Save label.
        filename = f"{index:0{FILENAME_NUM_DIGITS}}.npz"
        filepath = os.path.join(region_path, filename)
        f = open(filepath, 'wb')
        np.savez_compressed(f, data=data)

    def input_summary(self) -> pd.DataFrame:
        cols = {
            'size-x': int,
            'size-y': int,
            'size-z': int
        }
        df = pd.DataFrame(columns=cols.keys())

        for sample in self.list_samples():
            row = self.sample(sample).input_summary()
            data = {
                'size-x': row['size-x'],
                'size-y': row['size-y'],
                'size-z': row['size-z']
            }
            df = df.append(data, ignore_index=True)

        df = df.astype(cols)
        return df

    def label_summary(self) -> pd.DataFrame:
        cols = {
            'sample': int,
            'region': str,
            'size-x': int,
            'size-y': int,
            'size-z': int
        }
        df = pd.DataFrame(columns=cols.keys())

        for sample in self.list_samples():
            row = self.sample(sample).input_summary()
            data = {
                'sample': sample,
                'region': row['region'],
                'size-x': row['size-x'],
                'size-y': row['size-y'],
                'size-z': row['size-z']
            }
            df = df.append(data, ignore_index=True)

        df = df.astype(cols)
        return df
