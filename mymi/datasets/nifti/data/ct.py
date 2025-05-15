import numpy as np
import os

from mymi.typing import *
from mymi.utils import *

from .data import NiftiData

class CtNiftiData(NiftiData):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__path = os.path.join(study.path, 'ct', f'{id}.nii.gz')

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__data, self.__spacing, self.__offset = load_nifti(self.__path)
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> CtImage:
        return self.__data

    @property
    @ensure_loaded
    def extrema(self) -> Point3D:
        extrema = tuple(np.array(self.fov) + self.__offset)
        return extrema

    @property
    @ensure_loaded
    def fov(self) -> ImageFOV3D:
        fov = tuple((np.array(self.__data.shape) - 1) * self.__spacing)
        return fov

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    @ensure_loaded
    def offset(self) -> Point3D:
        return self.__offset

    @property
    def path(self) -> str:
        return self.__path

    @property
    @ensure_loaded
    def size(self) -> np.ndarray:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> np.ndarray:
        return self.__spacing

