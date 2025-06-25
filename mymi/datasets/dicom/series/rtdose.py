import pandas as pd
from typing import *

from mymi.typing import *

from ..files import RtDoseFile
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

    @property
    def default_file(self) -> str:
        # Choose most recent RTDOSE.
        rtdose_ids = self.list_files()
        if len(rtdose_ids) == 0:
            return None
        else:
            return self.rtdose(rtdose_ids[-1])

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def data(self) -> DoseImage:
        return self.default_file.data

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def modality(self) -> DicomModality:
        return 'RTDOSE'

    @property
    def offset(self) -> Point3D:
        return self.default_file.offset

    @property
    def size(self) -> Size3D:
        return self.default_file.size

    @property
    def spacing(self) -> Spacing3D:
        return self.default_file.spacing

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    def rtdose(
        self,
        id: DicomSOPInstanceUID) -> RtDoseFile:
        return RtDoseFile(self, id)

    def __str__(self) -> str:
        return self.__global_id
