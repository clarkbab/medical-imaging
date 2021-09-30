import os
import pandas as pd
from typing import List

from mymi import config
from mymi import types

from ..dataset import Dataset, DatasetType
from .processed_partition import ProcessedPartition

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
        self._partitions = ['train', 'validation', 'test']

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self.description}' not found.")

    @property
    def description(self) -> str:
        return f"PROCESSED: {self._name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def type(self) -> DatasetType:
        return DatasetType.PROCESSED

    def list_partitions(self) -> List[str]:
        path = os.path.join(self._path, 'data')
        if os.path.exists(path):
            return os.listdir(path)
        else:
            return []

    def list_regions(
        self,
        clear_cache: bool = False) -> pd.DataFrame:
        p_data = []
        for p in self.list_partitions():
            region_df = self.partition(p).list_regions()
            region_df.insert(0, 'partition', p)
            p_data.append(region_df)
        region_df = pd.concat(p_data)
        return region_df

    def create_partition(
        self,
        name: types.ProcessedPartition) -> ProcessedPartition:
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
        name: types.ProcessedPartition) -> ProcessedPartition:
        return ProcessedPartition(self, name)

    def manifest(self) -> pd.DataFrame:
        """
        returns: the manifest table.
        """
        filepath = os.path.join(self._path, 'manifest.csv')
        df = pd.read_csv(filepath)
        return df

    def params(self) -> pd.DataFrame:
        """
        returns: the params table.
        """
        filepath = os.path.join(self._path, 'params.csv')
        df = pd.read_csv(filepath)
        return df

    def append_to_manifest(
        self,
        partition: str,
        index: int,
        id: str) -> None:
        """
        effect: adds a line to the manifest.
        """
        # Create manifest if not present.
        manifest_path = os.path.join(self._path, 'manifest.csv')
        if not os.path.exists(manifest_path):
            with open(manifest_path, 'w') as f:
                f.write('partition,patient-id,index\n')

        # Append line to manifest. 
        with open(manifest_path, 'a') as f:
            f.write(f"{partition},{id},{index}\n")
