import pandas as pd
from typing import *

from mymi.typing import *

from ..files import RtPlanFile, RtStructFile
from .series import DicomSeries

class RtPlanSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesID) -> None:
        self.__global_id = f"{study}:{id}"
        self.__id = id
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == 'rtplan') & (index['series-id'] == self.__id)].copy()
        if len(index) == 0:
            raise ValueError(f"No RTPLAN series with ID '{id}' found in study '{study}'.")
        self.__index = index

        # Get policy.
        self.__index_policy = self.__study.index_policy['rtplan']

    @property
    def default_file(self) -> RtPlanFile:
        # Choose most recent RTPLAN.
        rtplan_ids = self.list_files()
        return self.file(rtplan_ids[-1])

    def file(
        self,
        id: DicomSOPInstanceUID) -> RtPlanFile:
        return RtPlanFile(self, id)

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    @property
    def modality(self) -> DicomModality:
        return 'rtplan'

    @property
    def ref_rtstruct(self) -> RtStructFile:
        return self.default_file.ref_rtstruct

# Add properties.
props = ['global_id', 'id', 'index', 'index_policy', 'study']
for p in props:
    setattr(RtPlanSeries, p, property(lambda self, p=p: getattr(self, f'_{RtPlanSeries.__name__}__{p}')))


