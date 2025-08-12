import numpy as np
import os

from mymi.geometry import fov
from mymi.typing import *
from mymi.utils import *

from .images import NiftiImageSeries

class CtImageSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: NiftiSeriesID,
        filepath: FilePath) -> None:
        self.__dataset_id = dataset_id
        self.__filepath = filepath
        self._global_id = f'NIFTI:{dataset_id}:{pat_id}:{study_id}:{id}'
        self.__id = id
        self.__pat_id = pat_id
        self.__study_id = study_id

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                if self.__filepath.endswith('.nii') or self.__filepath.endswith('.nii.gz'):
                    self.__data, self.__spacing, self.__offset = load_nifti(self.__filepath)
                elif self.__filepath.endswith('.nrrd'):
                    self.__data, self.__spacing, self.__offset = load_nrrd(self.__filepath)
                else:
                    raise ValueError(f'Unsupported file format: {self.__filepath}')
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> CtData:
        return self.__data

    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Fov3D:
        return fov(self.__data, spacing=self.__spacing, offset=self.__offset, **kwargs)

    @property
    @ensure_loaded
    def offset(self) -> Point3D:
        return self.__offset

    @property
    @ensure_loaded
    def size(self) -> np.ndarray:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> np.ndarray:
        return self.__spacing

# Add properties.
props = ['filepath', 'id']
for p in props:
    setattr(CtImageSeries, p, property(lambda self, p=p: getattr(self, f'_{CtImageSeries.__name__}__{p}')))
