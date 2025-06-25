import pandas as pd
from typing import *

from mymi.typing import *

from ..files import RegionMap, RtStructFile
from .ct import CtSeries
from .series import DicomSeries

class RtStructSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesID,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__global_id = f"{study}:{id}"
        self.__id = id
        self.__region_dups = region_dups
        self.__region_map = region_map
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == 'RTSTRUCT') & (index['series-id'] == self.__id)].copy()
        if len(index) == 0:
            raise ValueError(f"No RTSTRUCT series with ID '{id}' found in study '{study}'.")
        self.__index = index

        # Get policies.
        self.__index_policy = self.__study.index_policy['rtstruct']
        self.__region_policy = self.__study.region_policy

    @property
    def default_file(self) -> Optional[RtStructFile]:
        # Choose most recent RTSTRUCT.
        rtstruct_ids = self.list_files()
        if len(rtstruct_ids) == 0:
            return None
        else:
            return self.rtstruct(rtstruct_ids[-1])

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    @property
    def modality(self) -> DicomModality:
        return 'RTSTRUCT'

    @property
    def ref_ct(self) -> CtSeries:
        return self.default_file.ref_ct

    @property
    def region_policy(self) -> pd.DataFrame:
        return self.__region_policy

    @property
    def study(self) -> str:
        return self.__study

    def has_landmark(self, *args, **kwargs):
        return self.default_file.has_landmark(*args, **kwargs)

    def has_regions(self, *args, **kwargs):
        return self.default_file.has_regions(*args, **kwargs)

    def landmark_data(self, *args, **kwargs):
        return self.default_file.landmark_data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs):
        return self.default_file.list_landmarks(*args, **kwargs)

    def list_regions(self, *args, **kwargs):
        return self.default_file.list_regions(*args, **kwargs)

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    def region_data(self, *args, **kwargs):
        return self.default_file.region_data(*args, **kwargs)

    def rtstruct(
        self,
        id: DicomSOPInstanceUID) -> RtStructFile:
        return RtStructFile(self, id, region_dups=self.__region_dups, region_map=self.__region_map)

    def __str__(self) -> str:
        return self.__global_id
