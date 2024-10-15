import numpy as np
import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset

from mymi import logging
from mymi.transforms import resample_3D
from mymi.types import ImageSize3D, ImageSpacing3D, PointMM3D

from .dicom_file import DICOMFile, SOPInstanceUID

class RTDOSE(DICOMFile):
    def __init__(
        self,
        series: 'RTDOSESeries',
        id: SOPInstanceUID):
        self.__loaded_data = False
        self.__data = None      # Lazy-loaded.
        self.__global_id = f"{series} - {id}"
        self.__offset = None    # Lazy-loaded.
        self.__id = id
        self.__series = series
        self.__spacing = None   # Lazy-loaded.

        # Get index.
        index = self.__series.index
        self.__index = index.loc[[self.__id]]       # Double brackets ensure result is DataFrame not Series.
        self.__verify_index()
        self.__path = self.__index.iloc[0]['filepath']

    @property
    def data(self) -> np.ndarray:
        if not self.__loaded_data:
            self.__load_rtdose_data()
            self.__loaded_data = True
        return self.__data

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def offset(self) -> PointMM3D:
        if not self.__loaded_data:
            self.__load_rtdose_data()
            self.__loaded_data = True
        return self.__offset

    @property
    def path(self) -> str:
        return self.__path

    @property
    def series(self) -> str:
        return self.__series

    @property
    def size(self) -> ImageSize3D:
        return self.data.shape

    @property
    def spacing(self) -> ImageSpacing3D:
        if not self.__loaded_data:
            self.__load_rtdose_data()
            self.__loaded_data = True
        return self.__spacing

    def get_rtdose(self) -> FileDataset:
        return dcm.read_file(self.__path)

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTPLAN '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RTPLANs found in index with SOPInstanceUID '{self.__id}' for series '{self.__series}'.")

    def __load_rtdose_data(self) -> None:
        rtdose = self.get_rtdose()

        # Store offset.
        offset = rtdose.ImagePositionPatient
        self.__offset = tuple(int(s) for s in offset)

        # Store spacing.
        spacing_x_y = rtdose.PixelSpacing 
        z_diffs = np.unique(np.diff(rtdose.GridFrameOffsetVector))
        if len(z_diffs) != 1:
            logging.warning(f"Slice z spacings for RTDOSE {self} not equal, setting RTDOSE data to 'None'.")
            self.__offset = None
            self.__spacing = None
            self.__data = None
            return

        spacing_z = z_diffs[0]
        self.__spacing = tuple(np.append(spacing_x_y, spacing_z))

        # Store dose data.
        pat = self.__series.study.patient
        data = np.transpose(rtdose.pixel_array)
        data = rtdose.DoseGridScaling * data
        self.__data = resample_3D(data, origin=self.__offset, spacing=self.__spacing, output_origin=pat.ct_offset, output_size=pat.ct_size, output_spacing=pat.ct_spacing) 

    def __str__(self) -> str:
        return self.__global_id
