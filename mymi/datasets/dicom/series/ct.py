from datetime import datetime
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

class CtSeries(DicomSeries):
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
        self.__dataset_id = dataset_id
        self.__filepaths = abspaths
        self.__id = id
        self.__index = index
        self.__index_policy = index_policy
        self.__modality = 'ct'
        self.__pat_id = pat_id
        self.__study_id = study_id

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper
    
    # Could return 'CTFile' objects - this would align with other series, but would create a lot of objects in memory.
    @property
    def dicoms(self) -> List[CtDicom]:
        # Sort CTs by z position, smallest first.
        ct_dicoms = [dcm.read_file(f, force=False) for f in self.__filepaths]
        ct_dicoms = list(sorted(ct_dicoms, key=lambda c: c.ImagePositionPatient[2]))
        return ct_dicoms

    @property
    @ensure_loaded
    def data(self) -> np.ndarray:
        return self.__data

    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Fov3D:
        return fov(self.__data, spacing=self.__spacing, offset=self.__offset, **kwargs)

    @property
    def filepaths(self) -> List[str]:
        return self.__filepaths

    @property
    @ensure_loaded
    def offset(self) -> Point3D:
        return self.__offset

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

    def __load_data(self) -> None:
        ct_dicoms = self.dicoms

        # Store offset.
        # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
        offset = ct_dicoms[0].ImagePositionPatient    
        self.__offset = tuple(float(o) for o in offset)

        # Store size.
        # Indexing checked that CT slices had consisent x/y spacing in series.
        self.__size = (
            ct_dicoms[0].pixel_array.shape[1],
            ct_dicoms[0].pixel_array.shape[0],
            len(ct_dicoms)
        )

        # Store spacing.
        # Indexing checked that CT slices were equally spaced in z-dimension.
        self.__spacing = (
            float(ct_dicoms[0].PixelSpacing[0]),
            float(ct_dicoms[0].PixelSpacing[1]),
            float(np.abs(ct_dicoms[1].ImagePositionPatient[2] - ct_dicoms[0].ImagePositionPatient[2]))
        )

        # Store CT data.
        data = np.zeros(shape=self.__size)
        for c in ct_dicoms:
            # Convert values to HU.
            ct_data = np.transpose(c.pixel_array)      # 'pixel_array' contains row-first image data.
            ct_data = c.RescaleSlope * ct_data + c.RescaleIntercept

            # Get z index.
            z_offset =  c.ImagePositionPatient[2] - self.__offset[2]
            z_idx = int(round(z_offset / self.__spacing[2]))

            # Add data.
            data[:, :, z_idx] = ct_data
        self.__data = data

# Add properties.
props = ['dataset_id', 'filepaths', 'id', 'index', 'index_policy', 'modality', 'pat_id', 'study_id']
for p in props:
    setattr(CtSeries, p, property(lambda self, p=p: getattr(self, f'_{CtSeries.__name__}__{p}')))
