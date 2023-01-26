import pandas as pd
from typing import List, Optional

from .dicom_file import SOPInstanceUID
from .dicom_series import DICOMModality, DICOMSeries, SeriesInstanceUID
from .region_map import RegionMap
from .rtstruct import RTSTRUCT

class RTSTRUCTSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: SeriesInstanceUID,
        region_map: Optional[RegionMap] = None) -> None:
        self.__global_id = f"{study} - {id}"
        self.__id = id
        self.__region_map = region_map
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == 'RTSTRUCT') & (index['series-id'] == id)]
        self.__index = index
        self.__check_index()

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTSTRUCT

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_rtstructs(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index['sop-id']))

    def rtstruct(
        self,
        id: SOPInstanceUID) -> RTSTRUCT:
        return RTSTRUCT(self, id, self.__region_map)

    def __str__(self) -> str:
        return self.__global_id

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTSTRUCTSeries '{self}' not found in index for study '{self.__study}'.")
