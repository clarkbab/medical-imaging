from datetime import datetime
import numpy as np
import pandas as pd
from pydicom import read_file
from pydicom.dataset import FileDataset
from typing import List

from mymi import types

from .dicom_series import DICOMModality, DICOMSeries, SeriesInstanceUID

CLOSENESS_ABS_TOL = 1e-10;
STUDY_DATE_FMT = '%Y%m%d'

class CTSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: SeriesInstanceUID):
        self.__data = None          # Lazy-loaded.
        self.__global_id = f"{study} - {id}"
        self.__offset = None        # Lazy-loaded.
        self.__size = None   # Lazy-loaded.
        self.__spacing = None   # Lazy-loaded.
        self.__study = study
        self.__id = id

        # Load index.
        index = self.__study.index
        index = index[(index.modality == 'CT') & (index['series-id'] == id)]
        self.__index = index
        self.__verify_index()

    @property
    def data(self) -> np.ndarray:
        if self.__data is None:
            self.__load_ct_data()
        return self.__data

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def id(self) -> SeriesInstanceUID:
        return self.__id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.CT

    @property
    def offset(self) -> types.PointMM3D:
        if self.__offset is None:
            self.__load_ct_data()
        return self.__offset

    @property
    def paths(self) -> str:
        return self.__paths

    @property
    def size(self) -> types.ImageSpacing3D:
        if self.__size is None:
            self.__load_ct_data()
        return self.__size

    @property
    def spacing(self) -> types.ImageSpacing3D:
        if self.__spacing is None:
            self.__load_ct_data()
        return self.__spacing

    @property
    def study(self) -> str:
        return self.__study

    @property
    def study_date(self) -> datetime:
        dt_str = self.get_cts()[0].StudyDate
        return datetime.strptime(dt_str, STUDY_DATE_FMT)
    
    def get_cts(self) -> List[FileDataset]:
        ct_paths = list(self.__index['filepath'])
        cts = [read_file(f) for f in ct_paths]
        cts = list(sorted(cts, key=lambda c: c.ImagePositionPatient[2]))
        return cts

    @property
    def first_ct(self) -> FileDataset:
        filepath = list(self.__index['filepath'])[0]
        ct = read_file(filepath)
        return ct

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"CTSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_ct_data(self) -> None:
        cts = self.get_cts()

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
