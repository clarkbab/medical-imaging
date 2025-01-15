import numpy as np
import os
from typing import *

from mymi.types import *

class TrainingSample:
    def __init__(
        self,
        split: 'TrainingSplit',
        id: int) -> None:
        self.__split = split
        self.__id = id
        self.__index = None
        self.__global_id = f'{self.__split} - {self.__id}'

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> str:
        if self.__index is None:
            s_index = self.split.index
            self.__index = s_index[s_index['sample-id'] == self.__id].iloc[0].copy()
        return self.__index

    @property
    def input(self) -> np.ndarray:
        filepath = os.path.join(self.split.path, 'inputs', f"{self.__id:03}.npz")
        input = np.load(filepath)['data']
        return input

    @property
    def label(self) -> np.ndarray:
        filepath = os.path.join(self.split.path, 'labels', f"{self.__id:03}.npz")
        input = np.load(filepath)['data']
        return input

    @property
    def origin(self) -> Tuple[str, str, str]:
        return self.__index['origin-dataset'], self.__index['origin-patient-id'], self.__index['origin-study-id']

    @property
    def pair(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.input, self.label

    @property
    def size(self) -> ImageSize3D:
        return self.input.shape

    @property
    def spacing(self) -> ImageSpacing3D:
        return self.__split.dataset.spacing

    @property
    def split(self) -> 'TrainingSplit':
        return self.__split

    def __str__(self) -> str:
        return self.__global_id
