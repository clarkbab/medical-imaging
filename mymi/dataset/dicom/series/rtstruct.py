import pandas as pd
from typing import List, Optional

from mymi.types.types import PatientRegion

from ..files import RegionMap, RTSTRUCT, SOPInstanceUID
from .ct import CtSeries
from .series import Modality, DicomSeries, SeriesInstanceUID

class RtstructSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesInstanceUID,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__global_id = f"{study} - {id}"
        self.__id = id
        self.__region_dups = region_dups
        self.__region_map = region_map
        self.__study = study

        # Get index.
        index = self.__study.index
        self.__index = index[(index.modality == Modality.RTSTRUCT) & (index['series-id'] == self.__id)]
        self.__verify_index()

        # Get policies.
        self.__index_policy = self.__study.index_policy['rtstruct']
        self.__region_policy = self.__study.region_policy

    @property
    def default_rtstruct(self) -> Optional[RTSTRUCT]:
        # Choose most recent RTSTRUCT.
        rtstruct_ids = self.list_rtstructs()
        if len(rtstruct_ids) == 0:
            return None
        else:
            return self.rtstruct(rtstruct_ids[-1])

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    @property
    def modality(self) -> Modality:
        return Modality.RTSTRUCT

    @property
    def ref_ct(self) -> CtSeries:
        return self.default_rtstruct.ref_ct

    @property
    def region_policy(self) -> pd.DataFrame:
        return self.__region_policy

    @property
    def study(self) -> str:
        return self.__study

    def has_landmark(self, *args, **kwargs):
        return self.default_rtstruct.has_landmark(*args, **kwargs)

    def has_region(self, *args, **kwargs):
        return self.default_rtstruct.has_region(*args, **kwargs)

    def landmark_data(self, *args, **kwargs):
        return self.default_rtstruct.landmark_data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs):
        return self.default_rtstruct.list_landmarks(*args, **kwargs)

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def list_rtstructs(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index.index))

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)

    def rtstruct(
        self,
        id: SOPInstanceUID) -> RTSTRUCT:
        return RTSTRUCT(self, id, region_dups=self.__region_dups, region_map=self.__region_map)

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RtstructSeries '{self}' not found in index for study '{self.__study}'.")

    def __str__(self) -> str:
        return self.__global_id
