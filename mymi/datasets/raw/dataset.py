import os

from mymi import config

from ..dataset import Dataset, DatasetType

class RawDataset(Dataset):
    def __init__(
        self,
        name: str):
        # Create 'global ID'.
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'raw', self.__name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset 'RAW: {self.__name}' not found. Filepath: {self.__path}")
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def path(self) -> str:
        return self.__path

    @property
    def type(self) -> DatasetType:
        return DatasetType.RAW

    def __str__(self) -> str:
        return self.__global_id
    