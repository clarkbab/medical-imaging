import numpy as np
import os

from mymi.typing import *
from mymi.utils import *

from .data import NiftiData

class MrNiftiData(NiftiData):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__path = os.path.join(study.path, 'mr', f'{id}.nii.gz')

    @property
    def data(self) -> MrImage:
        data, _, _ = load_nifti(self.__path)
        return data

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def offset(self) -> np.ndarray:
        _, _, offset = load_nifti(self.__path)
        return offset

    @property
    def path(self) -> str:
        return self.__path

    @property
    def size(self) -> np.ndarray:
        data, _, _ = load_nifti(self.__path)
        return data.shape

    @property
    def spacing(self) -> np.ndarray:
        _, spacing, _ = load_nifti(self.__path)
        return spacing

