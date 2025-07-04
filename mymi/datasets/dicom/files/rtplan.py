import os
import pandas as pd
import pydicom as dcm
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
    def dicom(self) -> RtPlanDicom:
        return dcm.read_file(self.__filepath)

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

# Add properties.
props = ['filepath', 'global_id', 'id', 'index', 'series']
for p in props:
    setattr(RtPlanFile, p, property(lambda self, p=p: getattr(self, f'_{RtPlanFile.__name__}__{p}')))
