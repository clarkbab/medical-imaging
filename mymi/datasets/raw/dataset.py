import os

from mymi import config
from mymi.typing import *

from ..dataset import Dataset, DatasetType

class RawDataset(Dataset):
    def __init__(
        self,
        id: DatasetID) -> None:
        self._path = os.path.join(config.directories.datasets, 'nifti', str(id))
        if not os.path.exists(self._path):
            raise ValueError(f"No nifti dataset '{id}' found at path: {self._path}")
        super().__init__(id)
    