from datetime import datetime
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import *

from mymi.constants import DICOM_DATE_FORMAT, DICOM_TIME_FORMAT
from mymi.geometry import get_extent
from mymi.typing import *
from mymi.utils import *

from .series import DicomSeries

class CtSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesID,
        force_dicom_read: bool = False) -> None:
        self.__force_dicom_read = force_dicom_read
        self.__global_id = f"{study}:{id}"
        self.__study = study
        self.__id = id

        # Load index.
        index = self.__study.index
        index = index[(index.modality == 'ct') & (index['series-id'] == id)].copy()
        if len(index) == 0:
            raise ValueError(f"No CT series with ID '{id}' found in study '{study}'.")
        self.__index = index

        # Save paths.
        relpaths = list(self.__index['filepath'])
        abspaths = [os.path.join(self.__study.patient.dataset.path, p) for p in relpaths]
        self.__filepaths = abspaths

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
        ct_dicoms = [dcm.read_file(f, force=self.__force_dicom_read) for f in self.__filepaths]
        ct_dicoms = list(sorted(ct_dicoms, key=lambda c: c.ImagePositionPatient[2]))
        return ct_dicoms

    @property
    @ensure_loaded
    def data(self) -> np.ndarray:
        return self.__data

    @ensure_loaded
    def extent(
        self,
        use_patient_coords: bool = True) -> Union[Point3D, Voxel]:
        return get_extent(self.__data, spacing=self.__spacing, offset=self.__offset, use_patient_coords=use_patient_coords)

    @property
    def filepaths(self) -> List[str]:
        return self.__filepaths

    @property
    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Union[FOV3D, Size3D]:
        ext_min, ext_max = self.extent(**kwargs)
        fov = tuple(np.array(ext_max) - ext_min)
        return fov

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def modality(self) -> DicomModality:
        return 'ct'

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

    @property
    def study(self) -> str:
        return self.__study

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
props = ['filepaths', 'global_id', 'id', 'index', 'index_policy', 'study']
for p in props:
    setattr(CtSeries, p, property(lambda self, p=p: getattr(self, f'_{CtSeries.__name__}__{p}')))
