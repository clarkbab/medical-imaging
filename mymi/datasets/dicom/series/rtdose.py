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
        index = index[(index.modality == 'RTDOSE') & (index['series-id'] == self.__id)].copy()
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

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def data(self) -> DoseImage:
        return self.default_file.data

    def file(
        self,
        id: DicomSOPInstanceUID) -> RtDoseFile:
        return RtDoseFile(self, id)

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    @property
    def modality(self) -> DicomModality:
        return 'RTDOSE'

    @property
    def offset(self) -> Point3D:
        return self.default_file.offset

    @property
    def ref_rtplan(self) -> RtPlanFile:
        return self.default_file.ref_rtplan

    @property
    def size(self) -> Size3D:
        return self.default_file.size

    @property
    def spacing(self) -> Spacing3D:
        return self.default_file.spacing

    @property
    def study(self) -> str:
        return self.__study

    def __str__(self) -> str:
        return self.__global_id
