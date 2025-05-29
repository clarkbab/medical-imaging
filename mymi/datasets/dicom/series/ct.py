from datetime import datetime
import numpy as np
import pandas as pd
import pydicom as dcm
from typing import *

from mymi.typing import *

from ..dicom import DATE_FORMAT, TIME_FORMAT
from .series import Modality, DicomSeries

CLOSENESS_ABS_TOL = 1e-10;

class CtSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesID,
        force_dicom_read: bool = False) -> None:
        self.__data = None          # Lazy-loaded.
        self.__force_dicom_read = force_dicom_read
        self.__fov = None           # Lazy-loaded.
        self.__global_id = f"{study}:{id}"
        self.__offset = None        # Lazy-loaded.
        self.__size = None          # Lazy-loaded.
        self.__spacing = None       # Lazy-loaded.
        self.__study = study
        self.__id = id

        # Load index.
        index = self.__study.index
        index = index[(index.modality == 'CT') & (index['series-id'] == id)]
        self.__index = index
        self.__verify_index()
    
    # Should really return 'CT' file object IDs.
    @property
    def ct_files(self) -> List[dcm.FileDataset]:
        # Sort CTs by z position, smallest first.
        ct_paths = list(self.__index['filepath'])
        cts = [dcm.read_file(f, force=self.__force_dicom_read) for f in ct_paths]
        cts = list(sorted(cts, key=lambda c: c.ImagePositionPatient[2]))
        return cts

    @property
    def data(self) -> np.ndarray:
        if self.__data is None:
            self.__load_data()
        return self.__data

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def fov(self) -> ImageSizeMM3D:
        if self.__fov is None:
            self.__load_data()
        return self.__fov

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def modality(self) -> Modality:
        return 'CT'

    @property
    def offset(self) -> Point3D:
        if self.__offset is None:
            self.__load_data()
        return self.__offset

    @property
    def paths(self) -> str:
        return self.__paths

    @property
    def size(self) -> Spacing3D:
        if self.__size is None:
            self.__load_data()
        return self.__size

    @property
    def spacing(self) -> Spacing3D:
        if self.__spacing is None:
            self.__load_data()
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

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"CtSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_data(self) -> None:
        cts = self.ct_files

        # Store offset.
        # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
        offset = cts[0].ImagePositionPatient    
        self.__offset = tuple(int(round(o)) for o in offset)

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
            np.abs(cts[1].ImagePositionPatient[2] - cts[0].ImagePositionPatient[2])
        )

        # Store field-of-view.
        self.__fov = tuple(np.array(self.__spacing) * self.__size)

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
