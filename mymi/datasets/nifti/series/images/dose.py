import numpy as np
import os

from mymi.geometry import fov
from mymi.typing import *
from mymi.utils import *

from .image import NiftiImageSeries

class NiftiDoseSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID) -> None:
        extensions = ['.nii', '.nii.gz', '.nrrd']
        basepath = os.path.join(config.directories.datasets, 'nifti', str(dataset_id), 'data', 'patients', str(pat_id), str(study_id), 'dose', str(id))
        filepath = None
        for e in extensions:
            fpath = f"{basepath}{e}"
            if os.path.exists(fpath):
                filepath = fpath
        if filepath is None:
            raise ValueError(f"No NiftiDoseSeries found for study '{study_id}'. Filepath: {basepath}, with extensions {extensions}.")
        self.__filepath = filepath
        super().__init__(dataset_id, pat_id, study_id, id)

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                if self.__filepath.endswith('.nii') or self.__filepath.endswith('.nii.gz'):
                    self.__data, self.__spacing, self.__origin = load_nifti(self.__filepath)
                elif self.__filepath.endswith('.nrrd'):
                    self.__data, self.__spacing, self.__origin = load_nrrd(self.__filepath)
                else:
                    raise ValueError(f'Unsupported file format: {self.__filepath}')
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> DoseImageArray:
        return self.__data

    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Fov3D:
        return fov(self.__data, spacing=self.__spacing, origin=self.__origin, **kwargs)

    @property
    @ensure_loaded
    def origin(self) -> Point3D:
        return self.__origin

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

# Add properties.
props = ['filepath', 'id']
for p in props:
    setattr(NiftiDoseSeries, p, property(lambda self, p=p: getattr(self, f'_{NiftiDoseSeries.__name__}__{p}')))
