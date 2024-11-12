from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional

from mymi.dataset.dicom.rtstruct_series import RTSTRUCTSeries

from .ct_series import CTSeries
from .dicom_series import Modality, DICOMSeries, SeriesInstanceUID
from .region_map import RegionMap
from .rtdose import RTDOSE
from .rtdose_series import RTDOSESeries
from .rtplan import RTPLAN
from .rtplan_series import RTPLANSeries
from .rtstruct import RTSTRUCT
from .rtstruct_series import RTSTRUCTSeries

class DICOMStudy:
    def __init__(
        self,
        patient: 'DICOMPatient',
        id: str,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None):
        self.__default_rtdose = None        # Lazy-loaded.  
        self.__default_rtplan = None        # Lazy-loaded. 
        self.__default_rtstruct = None      # Lazy-loaded. 
        self.__loaded_default_rtstruct = False
        self.__id = id
        self.__patient = patient
        self.__global_id = f"{patient} - {id}"
        self.__region_dups = region_dups
        self.__region_map = region_map

        # Get index.
        index = self.__patient.index
        index = index[index['study-id'] == id]
        self.__index = index 
        self.__verify_index()

        # Get policies.
        self.__index_policy = self.__patient.index_policy
        self.__region_policy = self.__patient.region_policy

    @property
    def ct_data(self):
        return self.default_ct_series.data

    @property
    def ct_offset(self):
        return self.default_ct_series.offset

    @property
    def ct_size(self):
        return self.default_ct_series.size

    @property
    def ct_spacing(self):
        return self.default_ct_series.spacing

    @property
    def date(self) -> datetime:
        return self.default_ct_series.study_date

    @property
    def default_ct_series(self) -> CTSeries:
        if self.default_rtstruct is not None:
            return self.default_rtstruct.ref_ct.spacing

        ct_series = self.list_series(Modality.CT)
        if len(ct_series) == 1:
            return self.series(ct_series[0], Modality.CT)
        else:
            raise ValueError(f"Cannot determine default CT series for study '{self}', got {len(ct_series)} series.")

    @property
    def default_rtdose(self) -> RTDOSE:
        if self.__default_rtdose is None:
            self.__load_default_rtplan_and_rtdose()
        return self.__default_rtdose

    @property
    def default_rtplan(self) -> RTPLAN:
        if self.__default_rtplan is None:
            self.__load_default_rtplan_and_rtdose()
        return self.__default_rtplan
    
    @property
    def default_rtstruct(self) -> RTSTRUCT:
        if not self.__loaded_default_rtstruct:
            self.__load_default_rtstruct()
            self.__loaded_default_rtstruct = True
        return self.__default_rtstruct

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dose_data(self):
        if self.default_rtdose is None:
            return None
        return self.default_rtdose.data

    @property
    def dose_offset(self):
        if self.default_rtdose is None:
            return None
        return self.default_rtdose.offset

    @property
    def dose_size(self):
        if self.default_rtdose is None:
            return None
        return self.default_rtdose.size

    @property
    def dose_spacing(self):
        if self.default_rtdose is None:
            return None
        return self.default_rtdose.spacing

    @property
    def first_ct(self):
        return self.default_rtstruct.ref_ct.first_ct

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    @property
    def patient(self) -> str:
        return self.__patient

    @property
    def region_policy(self) -> pd.DataFrame:
        return self.__region_policy

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def list_series(
        self,
        modality: str) -> List[SeriesInstanceUID]:
        index = self.__index
        index = index[index.modality == modality]
        series = list(sorted(index['series-id'].unique()))
        return series

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)

    def series(
        self,
        id: SeriesInstanceUID,
        modality: Modality,
        **kwargs: Dict) -> DICOMSeries:
        if modality == Modality.CT:
            return CTSeries(self, id, **kwargs)
        elif modality == Modality.RTDOSE:
            return RTDOSESeries(self, id, **kwargs)
        elif modality == Modality.RTPLAN:
            return RTPLANSeries(self, id, **kwargs)
        elif modality == Modality.RTSTRUCT:
            return RTSTRUCTSeries(self, id, region_dups=self.__region_dups, region_map=self.__region_map, **kwargs)
        else:
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")

    def __load_default_rtplan_and_rtdose(self) -> None:
        if not self.__index_policy['rtplan']['no-ref-rtstruct']['allow']:
            # Get RTPLAN/RTDOSE linked to RTSTRUCT. No guarantees in 'index' building that
            # these RTPLAN/RTDOSE files are present.
            def_rtstruct = self.default_rtstruct
            def_rt_sop_id = def_rtstruct.get_rtstruct().SOPInstanceUID

            # Find RTPLANs that link to default RTSTRUCT.
            linked_rtplan_sop_ids = []
            linked_rtplan_series_ids = []
            rtplan_series_ids = self.list_series(Modality.RTPLAN)
            for rtplan_series_id in rtplan_series_ids:
                rtplan_series = self.series(rtplan_series_id, Modality.RTPLAN)
                rtplan_sop_ids = rtplan_series.list_rtplans()
                for rtplan_sop_id in rtplan_sop_ids:
                    rtplan = rtplan_series.rtplan(rtplan_sop_id)
                    rtplan_ref_rtstruct_sop_id = rtplan.get_rtplan().ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                    if rtplan_ref_rtstruct_sop_id == def_rt_sop_id:
                        linked_rtplan_series_ids.append(rtplan_series_id)
                        linked_rtplan_sop_ids.append(rtplan_sop_id)

            if len(linked_rtplan_sop_ids) == 0:
                # If no linked RTPLAN, then no RTDOSE either.
                self.__default_rtplan = None
                self.__default_rtdose = None
                return

            # Preference most recent RTPLAN as default.
            def_rtplan_series_id = linked_rtplan_series_ids[-1]
            def_rtplan_sop_id = linked_rtplan_sop_ids[-1]
            def_rtplan_series = self.series(def_rtplan_series_id, Modality.RTPLAN)
            self.__default_rtplan = def_rtplan_series.rtplan(def_rtplan_sop_id)

        elif self.__index_policy['rtplan']['no-ref-rtstruct']['only'] == 'at-least-one-rtstruct':
            # Choose first RTPLAN from study.
            rtplan_series_ids = self.list_series(Modality.RTPLAN)
            if len(rtplan_series_ids) > 0:
                def_rtplan_series_id = rtplan_series_ids[-1]
                def_rtplan_series = self.series(def_rtplan_series_id, Modality.RTPLAN)
                def_rtplan_sop_id = def_rtplan_series.list_rtplans()[-1]
                self.__default_rtplan = def_rtplan_series.rtplan(def_rtplan_sop_id) 
            else:
                self.__default_rtplan = None

        if not self.__index_policy['rtdose']['no-ref-rtplan']['allow']:
            # Get RTDOSEs linked to first RTPLAN.
            linked_rtdose_series_ids = []
            linked_rtdose_sop_ids = []
            rtdose_series_ids = self.list_series(Modality.RTDOSE)
            for rtdose_series_id in rtdose_series_ids:
                rtdose_series = self.series(rtdose_series_id, Modality.RTDOSE)
                rtdose_sop_ids = rtdose_series.list_rtdoses()
                for rtdose_sop_id in rtdose_sop_ids:
                    rtdose = rtdose_series.rtdose(rtdose_sop_id)
                    rtdose_ref_rtplan_sop_id = rtdose.get_rtdose().ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                    if rtdose_ref_rtplan_sop_id == self.__default_rtplan.id:
                        linked_rtdose_series_ids.append(rtdose_series_id)
                        linked_rtdose_sop_ids.append(rtdose_sop_id)

            if len(linked_rtdose_sop_ids) == 0:
                self.__default_rtdose = None
                return

            # Preference most recent RTDOSE as default.
            def_rtdose_series_id = linked_rtdose_series_ids[-1]
            def_rtdose_sop_id = linked_rtdose_sop_ids[-1]
            def_rtdose_series = self.series(def_rtdose_series_id, Modality.RTDOSE)
            self.__default_rtdose = def_rtdose_series.rtdose(def_rtdose_sop_id)

        elif self.__index_policy['rtdose']['no-ref-rtplan']['only'] == 'at-least-one-rtplan':
            # Choose first RTDOSE from study.
            rtdose_series_ids = self.list_series(Modality.RTDOSE)
            if len(rtdose_series_ids) > 0:
                def_rtdose_series_id = rtdose_series_ids[-1]
                def_rtdose_series = self.series(def_rtdose_series_id, Modality.RTDOSE)
                def_rtdose_sop_id = def_rtdose_series.list_rtdoses()[-1]
                self.__default_rtdose = def_rtdose_series.rtdose(def_rtdose_sop_id) 
            else:
                self.__default_rtdose = None

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"DICOMStudy '{self}' not found in index for patient '{self.__patient}'.")

    def __load_default_rtstruct(self) -> None:
        # Preference most recent RTSTRUCT series.
        series = self.list_series(Modality.RTSTRUCT)
        if len(series) == 0:
            self.__default_rtstruct = None
            return

        # Set first RTSTRUCT series as default.
        def_rtstruct_series_id = self.list_series(Modality.RTSTRUCT)[-1]
        def_rtstruct_series = self.series(def_rtstruct_series_id, Modality.RTSTRUCT)
        self.__default_rtstruct = def_rtstruct_series.default_rtstruct

    def __str__(self) -> str:
        return self.__global_id
