import pandas as pd
from typing import List, Optional

from ..files import RTPLAN, SOPInstanceUID
from .series import Modality, DicomSeries, SeriesInstanceUID

class RtplanSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesInstanceUID) -> None:
        self.__global_id = f"{study} - {id}"
        self.__id = id
        self.__study = study

        # Get index.
        index = self.__study.index
        self.__index = index[(index.modality == Modality.RTPLAN) & (index['series-id'] == self.__id)]
        self.__verify_index()

    @property
    def default_rtplan(self) -> str:
        if self.__default_rtplan is None:
            self.__load_default_rtplan()
        return self.__default_rtplan

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def modality(self) -> Modality:
        return Modality.RTPLAN

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_rtplans(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index.index))

    def rtplan(
        self,
        id: SOPInstanceUID) -> RTPLAN:
        return RTPLAN(self, id)

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RtplanSeries '{self}' not found in index for study '{self.__study}'.")

    def __load_default_rtplan(self) -> None:
        # Preference most recent RTPLAN.
        def_rtplan_id = self.list_rtplans()[-1]
        self.__default_rtplan = self.rtplan(def_rtplan_id)

    def __str__(self) -> str:
        return self.__global_id
