from datetime import datetime
import pandas as pd
import pydicom as dcm
from typing import Dict, List, Optional

from .files import RegionMap, RTDOSE, RTPLAN, RTSTRUCT
from .series import CtSeries, DicomSeries, Modality, RtdoseSeries, RtplanSeries, RtstructSeries, SeriesInstanceUID

StudyInstanceUID = str

class DicomStudy:
    def __init__(
        self,
        patient: 'DicomPatient',
        id: StudyInstanceUID,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None):
        self.__default_rtdose = None        # Lazy-loaded.  
        self.__default_rtplan = None        # Lazy-loaded. 
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
    def cts(self) -> List[dcm.FileDataset]:
        return self.default_ct.cts

    @property
    def ct_data(self):
        return self.default_ct.data

    @property
    def ct_offset(self):
        return self.default_ct.offset

    @property
    def ct_size(self):
        return self.default_ct.size

    @property
    def ct_spacing(self):
        return self.default_ct.spacing

    @property
    def datetime(self) -> datetime:
        return self.default_ct.study_datetime

    @property
    def default_ct(self) -> Optional[CtSeries]:
        series_ids = self.list_series(Modality.CT)
        if len(series_ids) == 0:
            return None
        else:
            return self.series(series_ids[-1], Modality.CT)

    @property
    def default_rtdose(self) -> RtdoseSeries:
        if self.__default_rtdose is None:
            self.__load_default_rtplan_and_rtdose()
        return self.__default_rtdose

    @property
    def default_rtplan(self) -> RtplanSeries:
        if self.__default_rtplan is None:
            self.__load_default_rtplan_and_rtdose()
        return self.__default_rtplan
    
    @property
    def default_rtstruct(self) -> Optional[RtstructSeries]:
        # Choose most recent RTSTRUCT series.
        series_ids = self.list_series(Modality.RTSTRUCT)
        if len(series_ids) == 0:
            return None
        else:
            return self.series(series_ids[-1], Modality.RTSTRUCT)

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
        return self.default_ct.first_ct

    @property
    def has_ct(self):
        if len(self.list_series(Modality.CT)) > 0:
            return True
        else:
            return False

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

    def list_series(
        self,
        modality: Modality) -> List[SeriesInstanceUID]:
        # Sort series by date/time - oldest first.
        index = self.__index[self.__index.modality == modality]
        series_ids = list(index.sort_values(['series-date', 'series-time'])['series-id'].unique())
        return series_ids

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)

    def series(
        self,
        id: SeriesInstanceUID,
        modality: Modality,
        **kwargs: Dict) -> DicomSeries:
        if modality == Modality.CT:
            return CtSeries(self, id, **kwargs)
        elif modality == Modality.RTDOSE:
            return RtdoseSeries(self, id, **kwargs)
        elif modality == Modality.RTPLAN:
            return RtplanSeries(self, id, **kwargs)
        elif modality == Modality.RTSTRUCT:
            return RtstructSeries(self, id, region_dups=self.__region_dups, region_map=self.__region_map, **kwargs)
        else:
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")

    def __load_default_rtplan_and_rtdose(self) -> None:
        if not self.__index_policy['rtplan']['no-ref-rtstruct']['allow']:
            # Get RTPLAN/RTDOSE linked to RTSTRUCT. No guarantees in 'index' building that
            # these RTPLAN/RTDOSE files are present.
            def_rtstruct = self.default_rtstruct.default_rtstruct
            def_rt_sop_id = def_rtstruct.rtstruct.SOPInstanceUID

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
            raise ValueError(f"DicomStudy '{self}' not found in index for patient '{self.__patient}'.")

    def __str__(self) -> str:
        return self.__global_id
