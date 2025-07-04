import pandas as pd
from typing import *

from mymi.typing import *

from ..files import RtDoseFile, RtPlanFile
from .series import DicomSeries

class RtDoseSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesID) -> None:
        self.__global_id = f"{study}:{id}"
        self.__id = id
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == 'rtdose') & (index['series-id'] == self.__id)].copy()
        if len(index) == 0:
            raise ValueError(f"No RTDOSE series with ID '{id}' found in study '{study}'.")
        self.__index = index

        # Get policy.
        self.__index_policy = self.__study.index_policy['rtdose']

    @property
    def default_file(self) -> RtDoseFile:
        # Choose most recent RTDOSE.
        rtdose_ids = self.list_files()
        return self.file(rtdose_ids[-1])

    def file(
        self,
        id: DicomSOPInstanceUID) -> RtDoseFile:
        return RtDoseFile(self, id)

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    @property
    def modality(self) -> DicomModality:
        return 'rtdose'

# Add properties.
props = ['global_id', 'id', 'index', 'index_policy', 'study']
for p in props:
    setattr(RtDoseSeries, p, property(lambda self, p=p: getattr(self, f'_{RtDoseSeries.__name__}__{p}')))

# Add property shortcuts from 'default_file'.
props = ['data', 'filepath', 'fov', 'offset', 'ref_rtplan', 'size', 'spacing']
for p in props:
    setattr(RtDoseSeries, p, property(lambda self, p=p: getattr(self.default_file, p) if self.default_file is not None else None))
