import pandas as pd
from typing import *

from mymi.typing import *

from ..files import RtPlanFile
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
        index = index[(index.modality == 'RTPLAN') & (index['series-id'] == self.__id)].copy()
        if len(index) == 0:
            raise ValueError(f"No RTPLAN series with ID '{id}' found in study '{study}'.")
        self.__index = index

    @property
    def default_file(self) -> Optional[RtPlanFile]:
        # Choose most recent RTPLAN.
        rtplan_ids = self.list_files()
        if len(rtplan_ids) == 0:
            return None
        else:
            return self.rtplan(rtplan_ids[-1])

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def modality(self) -> DicomModality:
        return 'RTPLAN'

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    def rtplan(
        self,
        id: DicomSOPInstanceUID) -> RtPlanFile:
        return RtPlanFile(self, id)

    def __str__(self) -> str:
        return self.__global_id
