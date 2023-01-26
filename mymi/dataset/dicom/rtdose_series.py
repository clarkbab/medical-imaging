import pandas as pd
from typing import List, Optional

from .dicom_file import SOPInstanceUID
from .dicom_series import DICOMModality, DICOMSeries, SeriesInstanceUID
from .region_map import RegionMap
from .rtdose import RTDOSE

class RTDOSESeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: SeriesInstanceUID) -> None:
        self.__global_id = f"{study} - {id}"
        self.__id = id
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == 'RTDOSE') & (index['series-id'] == id)]
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
        return DICOMModality.RTDOSE

    @property
    def study(self) -> str:
        return self.__study

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    def list_rtdoses(self) -> List[SOPInstanceUID]:
        return list(sorted(self.__index['sop-id']))

    def rtdose(
        self,
        id: SOPInstanceUID) -> RTDOSE:
        return RTDOSE(self, id)

    def __str__(self) -> str:
        return self.__global_id

    def __check_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTDOSESeries '{self}' not found in index for study '{self.__study}'.")
