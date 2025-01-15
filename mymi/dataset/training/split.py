import os
import pandas as pd
from typing import *

from .sample import TrainingSample

class TrainingSplit:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        name: str) -> None:
        self.__dataset = dataset
        self.__global_id = f"{self.__dataset}:{name}"
        self.__name = name
        self.__path = os.path.join(self.__dataset.path, 'data', name)
        self.__index = None

    @property
    def dataset(self) -> 'TrainingDataset':
        return self.__dataset

    @property
    def index(self) -> pd.DataFrame:
        if self.__index is None:
            ds_index = self.dataset.index
            self.__index = ds_index[ds_index['split'] == self.__name].copy()
        return self.__index

    @property
    def path(self) -> str:
        return self.__path

    def list_samples(self) -> List[int]:
        sample_ids = self.index['sample-id'].to_list()
        return sample_ids

    def sample(
        self,
        sample_id: int) -> TrainingSample:
        return TrainingSample(self, sample_id)

    def __str__(self) -> str:
        return self.__global_id
