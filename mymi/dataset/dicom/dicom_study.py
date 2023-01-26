import os
import pandas as pd
from typing import Dict, List, Optional

from .ct_series import CTSeries
from .dicom_series import DICOMModality, DICOMSeries, SeriesInstanceUID
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
        self.__default_rtdose = None        # Lazy-loaded.  
        self.__default_rtplan = None        # Lazy-loaded. 
        self.__default_rtstruct = None      # Lazy-loaded. 
        self.__id = id
        self.__patient = patient
        self.__global_id = f"{patient} - {id}"
        self.__region_map = region_map

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
    def default_rtdose(self) -> str:
        if self.__default_rtdose is None:
            self.__load_default_rtplan_and_rtdose()
        return self.__default_rtdose

    @property
    def default_rtplan(self) -> RTPLANSeries:
        if self.__default_rtplan is None:
            self.__load_default_rtdose_and_rtplan()
        return self.__default_rtplan
    
    @property
    def default_rtstruct(self) -> RTSTRUCTSeries:
        if self.__default_rtstruct is None:
            self.__load_default_rtstruct()
        return self.__default_rtstruct


    def list_series(
        self,
        modality: str) -> List[SeriesInstanceUID]:
        index = self.__index
        index = index[index.modality == modality]
        series = list(sorted(index['series-id'].unique()))
        return series

    def series(
        self,
        id: SeriesInstanceUID,
        modality: DICOMModality,
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

    def __load_default_rtplan_and_rtdose(self) -> None:
        # Get RTPLAN/RTDOSE linked to RTSTRUCT. No guarantees in 'index' building that
        # these RTPLAN/RTDOSE files are present.
        def_rtstruct = self.default_rtstruct
        def_rt_sop_id = def_rtstruct.get_rtstruct().SOPInstanceUID

        # Find RTPLANs that link to default RTSTRUCT.
        rtplan_series_ids = self.list_series('RTPLAN')
        linked_rtplan_sop_ids = []
        linked_rtplan_series_ids = []
        for rtplan_series_id in rtplan_series_ids:
            rtplan = self.series(rtplan_series_id, 'RTPLAN')
            rtplan_ref_rtstruct_sop_id = rtplan.get_rtplan().ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            if rtplan_ref_rtstruct_sop_id == def_rt_sop_id:
                linked_rtplan_series_ids.append(rtplan_series_id)
                rtplan_sop_id = rtplan.get_rtplan().SOPInstanceUID
                linked_rtplan_sop_ids.append(rtplan_sop_id)

        if len(linked_rtplan_sop_ids) == 0:
            # If no linked RTPLAN, then no RTDOSE either.
            self.__default_rtplan = None
            self.__default_rtdose = None
            return

        # Set default RTPLAN.
        self.__default_rtplan = RTPLANSeries(linked_rtplan_sop_ids[0])

        # Get RTDOSEs linked to first RTPLAN.
        rtdose_series_ids = self.list_series('RTDOSE')
        linked_rtdose_series_ids = []
        for rtdose_series_id in rtdose_series_ids:
            rtdose = self.series(rtdose_series_id, 'RTDOSE')
            rtdose_ref_rtplan_sop_id = rtdose.get_rtdose().ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
            if rtdose_ref_rtplan_sop_id == self.__default_rtplan.id:
                linked_rtdose_series_ids.append(rtdose_series_id)

        if len(linked_rtdose_series_ids) == 0:
            self.__default_rtdose = None
        else:
            self.__default_rtdose = RTDOSESeries(linked_rtdose_series_ids[0])

    # Proxy calls to default series.

    @property
    def ct_data(self):
        return self.default_rtstruct.ref_ct.data

    @property
    def ct_offset(self):
        return self.default_rtstruct.ref_ct.offset

    @property
    def ct_size(self):
        return self.default_rtstruct.ref_ct.size

    @property
    def ct_spacing(self):
        return self.default_rtstruct.ref_ct.spacing

    @property
    def dose_data(self):
        if self.__default_rtdose is not None:
            return self.default_rtdose.data
        return None

    @property
    def dose_offset(self):
        if self.__default_rtdose is not None:
            return self.default_rtdose.offset
        return None

    @property
    def dose_size(self):
        if self.__default_rtdose is not None:
            return self.default_rtdose.size
        return None

    @property
    def dose_spacing(self):
        if self.__default_rtdose is not None:
            return self.default_rtdose.spacing
        return None

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)
