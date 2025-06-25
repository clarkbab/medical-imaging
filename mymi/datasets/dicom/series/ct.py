from datetime import datetime
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import *

from mymi.geometry import get_extent
from mymi.typing import *
from mymi.utils import *

from ..dicom import DATE_FORMAT, TIME_FORMAT
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
        index = index[(index.modality == 'CT') & (index['series-id'] == id)].copy()
        if len(index) == 0:
            raise ValueError(f"No CT series with ID '{id}' found in study '{study}'.")
        self.__index = index

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper
    
    # Should really return 'CT' file object IDs.
    @property
    def ct_files(self) -> List[dcm.FileDataset]:
        # Sort CTs by z position, smallest first.
        rel_ct_paths = list(self.__index['filepath'])
        abs_ct_paths = [os.path.join(self.__study.patient.dataset.path, p) for p in rel_ct_paths]
        cts = [dcm.read_file(f, force=self.__force_dicom_read) for f in abs_ct_paths]
        cts = list(sorted(cts, key=lambda c: c.ImagePositionPatient[2]))
        return cts

    @property
    @ensure_loaded
    def data(self) -> np.ndarray:
        return self.__data

    @property
    def description(self) -> str:
        return self.__global_id

    @ensure_loaded
    def extent(
        self,
        use_patient_coords: bool = True) -> Union[Point3D, Voxel]:
        return get_extent(self.__data, spacing=self.__spacing, offset=self.__offset, use_patient_coords=use_patient_coords)

    @property
    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Union[ImageSizeMM3D, Size3D]:
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
        return 'CT'

    @property
    @ensure_loaded
    def offset(self) -> Point3D:
        return self.__offset

    @property
    def paths(self) -> str:
        return self.__paths

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

    @property
    def study_datetime(self) -> datetime:
        ct = self.ct_files[0]
        datetime_str = f"{ct.StudyDate}:{ct.StudyTime}"
        datetime_fmt = f"{DATE_FORMAT}:{TIME_FORMAT}"
        return datetime.strptime(datetime_str, datetime_fmt)

    @property
    def first_ct(self) -> dcm.FileDataset:
        return self.ct_files[0]

    def __load_data(self) -> None:
        cts = self.ct_files

        # Store offset.
        # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
        offset = cts[0].ImagePositionPatient    
        self.__offset = tuple(float(o) for o in offset)

        # Store size.
        # Indexing checked that CT slices had consisent x/y spacing in series.
        self.__size = (
            cts[0].pixel_array.shape[1],
            cts[0].pixel_array.shape[0],
            len(cts)
        )

        # Store spacing.
        # Indexing checked that CT slices were equally spaced in z-dimension.
        self.__spacing = (
            float(cts[0].PixelSpacing[0]),
            float(cts[0].PixelSpacing[1]),
            float(np.abs(cts[1].ImagePositionPatient[2] - cts[0].ImagePositionPatient[2]))
        )

        # Store CT data.
        data = np.zeros(shape=self.__size)
        for ct in cts:
            # Convert values to HU.
            ct_data = np.transpose(ct.pixel_array)      # 'pixel_array' contains row-first image data.
            ct_data = ct.RescaleSlope * ct_data + ct.RescaleIntercept

            # Get z index.
            z_offset =  ct.ImagePositionPatient[2] - self.__offset[2]
            z_idx = int(round(z_offset / self.__spacing[2]))

            # Add data.
            data[:, :, z_idx] = ct_data
        self.__data = data

    def __str__(self) -> str:
        return self.__global_id
