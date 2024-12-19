import numpy as np
import os

from mymi.types import SeriesID
from mymi.utils import load_nrrd

from .data import NrrdData

class CtData(NrrdData):
    def __init__(
        self,
        study: 'NrrdStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__path = os.path.join(study.path, 'ct', f'{id}.nrrd')
        self.__load_data()

    @property
    def data(self) -> np.ndarray: return self.__data

    @property
    def offset(self) -> np.ndarray: return self.__offset

    @property
    def path(self) -> str: return self.__path

    @property
    def size(self) -> np.ndarray: return self.__data.shape

    @property
    def spacing(self) -> np.ndarray: return self.__spacing

    def __load_data(self) -> None:
        data, spacing, offset = load_nrrd(self.__path)
        self.__data = data
        self.__spacing = spacing
        self.__offset = offset
