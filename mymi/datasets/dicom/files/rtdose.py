import numpy as np
import os
import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset
from typing import *

from mymi.constants import *
from mymi import logging
from mymi.constants import *
from mymi.transforms import resample
from mymi.typing import *
from mymi.utils import *

from .files import DicomFile
from .rtplan import RtPlanFile

class RtDoseFile(DicomFile):
    def __init__(
        self,
        series: 'RtDoseSeries',
        id: DicomSOPInstanceUID):
        self.__global_id = f"{series}:{id}"
        self.__id = id
        self.__series = series

        # Get index.
        self.__index = self.__series.index.loc[self.__id].copy()
        self.__filepath = os.path.join(self.__series.study.patient.dataset.path, self.__index['filepath'])

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
    def dicom(self) -> FileDataset:
        return dcm.read_file(self.__filepath)

    @property
    def filepath(self) -> str:
        return self.__filepath

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

    def __load_data(self) -> None:
        rtdose_dicom = self.dicom

        # Store offset.
        self.__offset = tuple(float(o) for o in rtdose_dicom.ImagePositionPatient)

        # Store spacing.
        spacing_x_y = rtdose_dicom.PixelSpacing 
        z_diffs = np.diff(rtdose_dicom.GridFrameOffsetVector)
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
        data = np.transpose(rtdose_dicom.pixel_array)
        data = rtdose_dicom.DoseGridScaling * data
        kwargs = dict(
            offset=self.__offset,
            output_offset=pat.ct_offset,
            output_size=pat.ct_size,
            output_spacing=pat.ct_spacing,
            spacing=self.__spacing,
        )
        self.__data = resample(data, **kwargs)

        # Load referenced RTPLAN.
        if not self.__series.index_policy['no-ref-rtplan']['allow']:
            # Get referenced RTPLAN series from index.
            rtplan_file_id = self.__index['mod-spec'][DICOM_RTDOSE_REF_RTPLAN_KEY]
            self.__ref_rtplan = RtPlanFile(self.__series.study, rtplan_file_id)
        else:
            # Choose study default RTPLAN as "ref".
            self.__ref_rtplan = self.__series.study.default_rtplan.default_file

    @property
    @ensure_loaded
    def ref_rtplan(self) -> 'RtPlanFile':
        return self.__ref_rtplan

    def __str__(self) -> str:
        return self.__global_id
