import numpy as np
import os
import pandas as pd
from typing import *

from mymi import config
from mymi.typing import *
from mymi.utils import *

from ..dataset import Dataset, DatasetType
from .split import TrainingSample
from .split import HoldoutSplit

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
        if not os.path.exists(self.__path):
            raise ValueError(f"Training dataset '{self.__global_id}' does not exist.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def index(self) -> str:
        if self.__index is None:
            filepath = os.path.join(self.__path, 'index.csv')
            self.__index = pd.read_csv(filepath)
            str_types = [
                'origin-study-id',
                'origin-fixed-study-id',
                'origin-moving-study-id'
            ]
            for t in str_types:
                if t in self.__index.columns:
                    self.__index[t] = self.__index[t].astype(str)
        return self.__index

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def label_types(self) -> List[str]:
        return self.params['label-types'] if 'label-types' in self.params else []

    def list_samples(
        self,
        splits: Optional[Splits] = 'all') -> List[SampleID]:
        splits = arg_to_list(splits, Split, literals={ 'all': self.list_splits })
        samples = []
        for s in splits:
            split = self.split(s)
            samples += split.list_samples()
        samples = list(sorted(samples))
        return samples

    def list_splits(self) -> List[HoldoutSplit]:
        return list(self.index['split'].unique())
    
    @property
    def landmarks(self) -> List[LandmarkID]:
        return self.params['landmarks'] if 'landmarks' in self.params else []
    
    @property
    def n_input_channels(self) -> int:
        def_split = self.split(self.list_splits()[0])
        def_sample = def_split.sample(def_split.list_samples()[0])
        input = def_sample.input
        n_channels = input.shape[0]
        return n_channels

    @property
    def params(self) -> pd.DataFrame:
        if self.__params is None:
            filepath = os.path.join(self.path, 'params.csv')
            self.__params = pd.read_csv(filepath)
            self.__params = dict((p, v) for p, v in zip(self.__params['param'], self.__params['value']))
            eval_params = ['spacing', 'regions', 'landmarks', 'label-types']
            for e in eval_params:
                if e in self.__params:
                    self.__params[e] = eval(self.__params[e])
        return self.__params

    @property
    def path(self) -> str:
        return self.__path
    
    @property
    def regions(self) -> List[Region]:
        return self.params['regions'] if 'regions' in self.params else []

    def sample(self,
        id: SampleID) -> TrainingSample:
        splits = self.list_splits()
        for s in splits:
            split = self.split(s)
            samples = split.list_samples()
            if id in samples:
                return split.sample(id)
        raise ValueError(f"Unknown sample ID '{id}'.")
    
    @property
    def spacing(self) -> Spacing3D:
        return self.params['spacing']

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def split(
        self,
        name: str) -> HoldoutSplit:
        return HoldoutSplit(self, name)

    def __str__(self) -> str:
        return self.__global_id
