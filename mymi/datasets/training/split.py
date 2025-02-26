import os
import pandas as pd
from typing import *

from mymi.utils import *

from .sample import TrainingSample

class TrainingSplit:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        id: str) -> None:
        self.__dataset = dataset
        self.__id = id
        self.__global_id = f"{self.__dataset}:{self.__id}"
        self.__path = os.path.join(self.__dataset.path, 'data', self.__id)
        self.__index = None

    @property
    def dataset(self) -> 'TrainingDataset':
        return self.__dataset

    @property
    def index(self) -> pd.DataFrame:
        if self.__index is None:
            ds_index = self.dataset.index
            self.__index = ds_index[ds_index['split'] == self.__id].copy()
        return self.__index

    @property
    def path(self) -> str:
        return self.__path

    def list_samples(
        self,
        regions: Optional[PatientRegions] = None) -> List[int]:
        regions = arg_to_list(regions, PatientRegion)
        sample_ids = self.index['sample-id'].to_list()
        if regions is not None:
            sample_ids = [s for s in sample_ids if self.sample(s).has_regions(regions)]
        return sample_ids

    def sample(
        self,
        sample_id: int) -> TrainingSample:
        return TrainingSample(self, sample_id)

    def __str__(self) -> str:
        return self.__global_id
