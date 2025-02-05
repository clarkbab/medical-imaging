import numpy as np
import os
import pandas as pd
from typing import *

from mymi import config
from mymi.typing import *

from ..dataset import Dataset, DatasetType
from .split import TrainingSplit

class TrainingDataset(Dataset):
    def __init__(
        self,
        name: str,
        **kwargs):
        self.__index = None
        self.__name = name
        self.__global_id = f"TRAINING:{self.__name}"
        self.__params = None
        self.__path = os.path.join(config.directories.datasets, 'training', self.__name)

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def index(self) -> str:
        if self.__index is None:
            filepath = os.path.join(self.__path, 'index.csv')
            self.__index = pd.read_csv(filepath).astype({ 'sample-id': int, 'origin-patient-id': str , 'origin-study-id': str })
        return self.__index

    @property
    def name(self) -> str:
        return self.__name

    @property
    def params(self) -> pd.DataFrame:
        if self.__params is None:
            filepath = os.path.join(self.path, 'params.csv')
            self.__params = pd.read_csv(filepath)
        return self.__params

    @property
    def path(self) -> str:
        return self.__path
    
    @property
    def spacing(self) -> ImageSpacing3D:
        return eval(self.params[self.params['param'] == 'spacing']['value'].iloc[0])

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def list_regions(self) -> List[str]:
        return eval(self.params[self.params['param'] == 'regions']['value'].iloc[0])

    def list_splits(self) -> List[str]:
        return list(self.index['split'].unique())

    def split(
        self,
        name: str) -> TrainingSplit:
        return TrainingSplit(self, name)

    def __str__(self) -> str:
        return self.__global_id
