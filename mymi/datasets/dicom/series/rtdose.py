import pandas as pd
from typing import List, Optional

from ..files import RTDOSE, SOPInstanceUID
from .series import Modality, DicomSeries, SeriesInstanceUID

class RtdoseSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesInstanceUID) -> None:
        self.__global_id = f"{study}:{id}"
        self.__id = id
        self.__study = study

        # Get index.
        index = self.__study.index
        self.__index = index[(index.modality == Modality.RTDOSE) & (index['series-id'] == self.__id)]
        self.__verify_index()

    @property
    def default_rtdose(self) -> str:
        if self.__default_rtdose is None:
            self.__load_default_rtdose()
        return self.__default_rtdose

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def modality(self) -> Modality:
        return Modality.RTDOSE

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_rtdoses(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index.index))

    def rtdose(
        self,
        id: SOPInstanceUID) -> RTDOSE:
        return RTDOSE(self, id)

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RtdoseSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_default_rtdose(self) -> None:
        # Preference most recent RTDOSE.
        def_rtdose_id = self.list_rtdoses()[-1]
        self.__default_rtdose = self.rtdose(def_rtdose_id)

    def __str__(self) -> str:
        return self.__global_id
