import os
import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset
from typing import *

from mymi.constants import *
from mymi.typing import *
from mymi.utils import *

from .files import DicomFile
from .rtstruct import RtStructFile

class RtPlanFile(DicomFile):
    def __init__(
        self,
        series: 'RtPlanSeries',
        id: DicomSOPInstanceUID):
        self.__global_id = f"{series}:{id}"
        self.__id = id
        self.__series = series

        # Get index.
        self.__index = self.__series.index.loc[self.__id].copy()
        self.__filepath = os.path.join(self.__series.study.patient.dataset.path, self.__index['filepath'])

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__ref_rtstruct'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

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

    def __load_data(self) -> None:
        if not self.__series.index_policy['no-ref-rtstruct']['allow']:
            # Get referenced RTSTRUCT series from index.
            rtstruct_file_id = self.__index['mod-spec'][DICOM_RTPLAN_REF_RTSTRUCT_KEY]
            self.__ref_rtstruct = RtStructFile(self.__series.study, rtstruct_file_id)
        else:
            # Choose study default RTSTRUCT as "ref".
            self.__ref_rtstruct = self.__series.study.default_rtstruct.default_file

    @property
    @ensure_loaded
    def ref_rtstruct(self) -> 'RtStructFile':
        return self.__ref_rtstruct

    @property
    def series(self) -> str:
        return self.__series

    def __str__(self) -> str:
        return self.__global_id
