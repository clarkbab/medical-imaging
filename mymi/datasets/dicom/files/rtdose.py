import numpy as np
import os
import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset
from typing import *

from mymi import logging
from mymi.constants import TOLERANCE_MM
from mymi.transforms import resample
from mymi.typing import *
from mymi.utils import *

from .files import DicomFile

class RtDoseFile(DicomFile):
    def __init__(
        self,
        series: 'RtDoseSeries',
        id: DicomSOPInstanceUID):
        self.__global_id = f"{series}:{id}"
        self.__id = id
        self.__series = series

        # Get index.
        index = self.__series.index
        self.__index = index.loc[[self.__id]]       # Double brackets ensure result is DataFrame not Series.
        self.__verify_index()
        self.__path = os.path.join(self.__series.study.patient.dataset.path, self.__index.iloc[0]['filepath'])

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> DoseImage:
        return self.__data

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> DicomSOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    @ensure_loaded
    def offset(self) -> Point3D:
        return self.__offset

    @property
    def path(self) -> str:
        return self.__path

    @property
    def series(self) -> str:
        return self.__series

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

    @property
    def dicom(self) -> FileDataset:
        return dcm.read_file(self.__path)

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTPLAN '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RTPLANs found in index with DicomSOPInstanceUID '{self.__id}' for series '{self.__series}'.")

    def __load_data(self) -> None:
        rtdose = self.dicom

        # Store offset.
        self.__offset = tuple(float(o) for o in rtdose.ImagePositionPatient)

        # Store spacing.
        spacing_x_y = rtdose.PixelSpacing 
        z_diffs = np.diff(rtdose.GridFrameOffsetVector)
        z_diffs = np.unique(round(z_diffs, TOLERANCE_MM))
        if len(z_diffs) != 1:
            logging.warning(f"Slice z spacings for RtDoseFile {self} not equal, setting RtDoseFile data to 'None'. Got spacings: {z_diffs}.")
            self.__offset = None
            self.__spacing = None
            self.__data = None
            return

        spacing_z = z_diffs[0]
        self.__spacing = tuple((float(s) for s in np.append(spacing_x_y, spacing_z)))

        # Load data.
        # Resample dose data to CT space.
        pat = self.__series.study.patient
        data = np.transpose(rtdose.pixel_array)
        data = rtdose.DoseGridScaling * data
        kwargs = dict(
            offset=self.__offset,
            output_offset=pat.ct_offset,
            output_size=pat.ct_size,
            output_spacing=pat.ct_spacing,
            spacing=self.__spacing,
        )
        self.__data = resample(data, **kwargs)

    def __str__(self) -> str:
        return self.__global_id
