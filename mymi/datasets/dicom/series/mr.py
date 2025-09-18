import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import *

from mymi import config
from mymi.constants import DICOM_DATE_FORMAT, DICOM_TIME_FORMAT
from mymi.geometry import fov
from mymi.typing import *
from mymi.utils import *

from .series import DicomSeries

class MrSeries(DicomSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any]) -> None:
        datasetpath = os.path.join(config.directories.datasets, 'dicom', dataset_id, 'data', 'patients')
        relpaths = list(index['filepath'])
        abspaths = [os.path.join(datasetpath, p) for p in relpaths]
        self._dataset_id = dataset_id
        self.__filepaths = abspaths
        self._id = id
        self.__index = index
        self.__index_policy = index_policy
        self.__modality = 'mr'
        self._pat_id = pat_id
        self._study_id = study_id

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper
    
    @property
    def dicoms(self) -> List[dcm.FileDataset]:
        # Sort MRs by z position, smallest first.
        mr_dicoms = [dcm.dcmread(f, force=False) for f in self.__filepaths]
        mr_dicoms = list(sorted(mr_dicoms, key=lambda m: m.ImagePositionPatient[2]))
        return mr_dicoms

    @property
    @ensure_loaded
    def data(self) -> MrImageArray:
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
    def size(self) -> Spacing3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

    def __load_data(self) -> None:
        mr_dicoms = self.dicoms

        # Store origin.
        # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
        origin = mr_dicoms[0].ImagePositionPatient    
        self.__origin = tuple(float(round(o)) for o in origin)

        # Store size.
        # Indexing checked that MR slices had consisent x/y spacing in series.
        self.__size = (
            mr_dicoms[0].pixel_array.shape[1],
            mr_dicoms[0].pixel_array.shape[0],
            len(mr_dicoms)
        )

        # Store spacing.
        # Indexing checked that MR slices were equally spaced in z-dimension.
        self.__spacing = (
            float(mr_dicoms[0].PixelSpacing[0]),
            float(mr_dicoms[0].PixelSpacing[1]),
            float(np.abs(mr_dicoms[1].ImagePositionPatient[2] - mr_dicoms[0].ImagePositionPatient[2]))
        )

        # Store MR data.
        data = np.zeros(shape=self.__size)
        for m in mr_dicoms:
            mr_data = np.transpose(m.pixel_array)      # 'pixel_array' contains row-first image data.

            # Get z index.
            z_origin =  m.ImagePositionPatient[2] - self.__origin[2]
            z_idx = int(round(z_origin / self.__spacing[2]))

            # Add data.
            data[:, :, z_idx] = mr_data

        self.__data = data

# Add properties.
props = ['dataset_id', 'filepaths', 'id', 'index', 'index_policy', 'modality', 'pat_id', 'study_id']
for p in props:
    setattr(MrSeries, p, property(lambda self, p=p: getattr(self, f'_{MrSeries.__name__}__{p}')))
