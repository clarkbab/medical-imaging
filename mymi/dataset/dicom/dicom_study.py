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
        if self.__default_rtstruct is None:
            self.__load_default_rtstruct
        return self.__default_rtstruct

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

    def __load_default_rtstruct(self) -> None:
        rtstruct_series_id = self.list_series('RTSTRUCT')[0]
        rtstruct_series = self.series(rtstruct_series_id, 'RTSTRUCT')
        self.__default_rtstruct = rtstruct_series

    @property
    def ct_data(self):
        return self.default_rtstruct.ref_ct.data

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)
