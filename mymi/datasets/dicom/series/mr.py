from datetime import datetime
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import *

from mymi.typing import *

from ..dicom import DATE_FORMAT, TIME_FORMAT
from .series import DicomSeries

class MrSeries(DicomSeries):
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
        index = index[(index.modality == 'MR') & (index['series-id'] == id)]
        self.__index = index
        self.__verify_index()
    
    # Should really return 'MR' file object IDs.
    @property
    def mr_files(self) -> List[dcm.FileDataset]:
        # Sort MRs by z position, smallest first.
        rel_filepaths = list(self.__index['filepath'])
        abs_filepaths = [os.path.join(self.__study.patient.dataset.path, p) for p in rel_filepaths]
        mrs = [dcm.read_file(f, force=self.__force_dicom_read) for f in abs_filepaths]
        mrs = list(sorted(mrs, key=lambda m: m.ImagePositionPatient[2]))
        return mrs

    @property
    def data(self) -> MrImage:
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
    def modality(self) -> DicomModality:
        return 'MR'

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
        mr = self.mrs[0]
        datetime_str = f"{mr.StudyDate}:{mr.StudyTime}"
        datetime_fmt = f"{DATE_FORMAT}:{TIME_FORMAT}"
        return datetime.strptime(datetime_str, datetime_fmt)

    @property
    def first_mr(self) -> dcm.FileDataset:
        return self.mrs[0]

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"MrSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_data(self) -> None:
        mrs = self.mr_files

        # Store offset.
        # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
        offset = mrs[0].ImagePositionPatient    
        self.__offset = tuple(int(round(o)) for o in offset)

        # Store size.
        # Indexing checked that MR slices had consisent x/y spacing in series.
        self.__size = (
            mrs[0].pixel_array.shape[1],
            mrs[0].pixel_array.shape[0],
            len(mrs)
        )

        # Store spacing.
        # Indexing checked that MR slices were equally spaced in z-dimension.
        self.__spacing = (
            float(mrs[0].PixelSpacing[0]),
            float(mrs[0].PixelSpacing[1]),
            np.abs(mrs[1].ImagePositionPatient[2] - mrs[0].ImagePositionPatient[2])
        )

        # Store field-of-view.
        self.__fov = tuple(np.array(self.__spacing) * self.__size)

        # Store MR data.
        data = np.zeros(shape=self.__size)
        for mr in mrs:
            mr_data = np.transpose(mr.pixel_array)      # 'pixel_array' contains row-first image data.

            # Get z index.
            z_offset =  mr.ImagePositionPatient[2] - self.__offset[2]
            z_idx = int(round(z_offset / self.__spacing[2]))

            # Add data.
            data[:, :, z_idx] = mr_data
        self.__data = data

    def __str__(self) -> str:
        return self.__global_id
