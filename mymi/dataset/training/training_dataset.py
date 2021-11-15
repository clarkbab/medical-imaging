import os
import pandas as pd
from typing import List

from mymi import config
from mymi import types

from ..dataset import Dataset, DatasetType
from .training_partition import TrainingPartition

class TrainingDataset(Dataset):
    def __init__(
        self,
        name: str):
        """
        args:
            name: the name of the dataset.
        """
        self._global_id = f"TRAINING: {name}"
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'training', name)
        self._partitions = ['train', 'validation', 'test']

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self.description}' not found.")

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    def params(self) -> pd.DataFrame:
        filepath = os.path.join(self._path, 'params.csv')
        df = pd.read_csv(filepath)
        return df

    def manifest(self) -> pd.DataFrame:
        filepath = os.path.join(self._path, 'manifest.csv')
        df = pd.read_csv(filepath)
        return df

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def list_partitions(self) -> List[str]:
        path = os.path.join(self._path, 'data')
        if os.path.exists(path):
            return os.listdir(path)
        else:
            return []

    def list_regions(self) -> pd.DataFrame:
        p_data = []
        for p in self.list_partitions():
            region_df = self.partition(p).list_regions()
            region_df.insert(0, 'partition', p)
            p_data.append(region_df)
        region_df = pd.concat(p_data)
        return region_df

    def create_partition(
        self,
        name: types.TrainingPartition) -> TrainingPartition:
        """
        effect: creates partition folder.
        args:
            name: the partition name.
        """
        path = os.path.join(self._path, 'data', name)
        os.makedirs(path)
        return self.partition(name)

    def partition(
        self,
        name: types.TrainingPartition) -> TrainingPartition:
        return TrainingPartition(self, name)
