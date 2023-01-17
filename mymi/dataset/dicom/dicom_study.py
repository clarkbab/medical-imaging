import os
import pandas as pd
from typing import Dict, List, Optional

from .ct_series import CTSeries
from .dicom_series import DICOMSeries
from .region_map import RegionMap
from .rtdose_series import RTDOSESeries
from .rtplan_series import RTPLANSeries
from .rtstruct_series import RTSTRUCTSeries

class DICOMStudy:
    def __init__(
        self,
        patient: 'DICOMPatient',
        id: str,
        region_map: Optional[RegionMap] = None):
        self.__patient = patient
        self.__id = id
        self.__region_map = region_map
        self.__global_id = f"{patient} - {id}"

        # Get study index.
        index = self.__patient.index
        index = index[index['study-id'] == id]
        self.__index = index 
    
        # Check that study ID exists.
        if len(index) == 0:
            raise ValueError(f"Study '{self}' not found in index for patient '{patient}'.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> str:
        return self.__id

    def __str__(self) -> str:
        return self.__global_id

    @property
    def patient(self) -> str:
        return self.__patient

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def default_rtstruct(self) -> RTSTRUCTSeries:
        rt_series_id = self.list_series('RTSTRUCT')[0]
        rt_series = self.series(rt_series_id, 'RTSTRUCT')
        return rt_series

    def list_series(
        self,
        modality: str) -> List[str]:
        index = self.__index
        index = index[index.modality == modality]
        series = list(sorted(index['series-id'].unique()))
        return series

    def series(
        self,
        id: str,
        modality: str,
        **kwargs: Dict) -> DICOMSeries:
        if modality == 'CT':
            return CTSeries(self, id, **kwargs)
        elif modality == 'RTSTRUCT':
            return RTSTRUCTSeries(self, id, region_map=self.__region_map, **kwargs)
        elif modality == 'RTPLAN':
            return RTPLANSeries(self, id, **kwargs)
        elif modality == 'RTDOSE':
            return RTDOSESeries(self, id, **kwargs)
        else:
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")
