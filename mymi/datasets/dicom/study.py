from datetime import datetime
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from .series import *
from .files import RegionMap    # import from "series" first to avoid circular imports, as "series" imports from "files".

class DicomStudy:
    def __init__(
        self,
        patient: 'DicomPatient',
        id: StudyID,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None):
        self.__id = id
        self.__patient = patient
        self.__global_id = f"{patient}:{id}"
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

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__default_rtdose'):
                self.__load_default_rtplan_and_rtdose()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def ct_data(self) -> Optional[CtImage]:
        return self.default_ct.data if self.default_ct is not None else None

    @property
    def ct_fov(self) -> Optional[Point3D]:
        return self.default_ct.fov if self.default_ct is not None else None

    @property
    def ct_offset(self) -> Optional[Point3D]:
        return self.default_ct.offset if self.default_ct is not None else None

    @property
    def ct_size(self) -> Optional[Size3D]:
        return self.default_ct.size if self.default_ct is not None else None 

    @property
    def ct_spacing(self) -> Optional[Spacing3D]:
        return self.default_ct.spacing if self.default_ct is not None else None

    @property
    def datetime(self) -> datetime:
        return self.default_ct.study_datetime

    @property
    def default_ct(self) -> Optional[CtSeries]:
        series_ids = self.list_series('CT')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'CT')

    @property
    def default_mr(self) -> Optional[MrSeries]:
        series_ids = self.list_series('MR')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'MR')

    @property
    @ensure_loaded
    def default_rtdose(self) -> RtDoseSeries:
        # Why not 'list_series' here?
        return self.__default_rtdose

    @property
    @ensure_loaded
    def default_rtplan(self) -> RtPlanSeries:
        # Why not 'list_series' here?
        return self.__default_rtplan
    
    @property
    def default_rtstruct(self) -> Optional[RtStructSeries]:
        series_ids = self.list_series('RTSTRUCT')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'RTSTRUCT')

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dose_data(self):
        return self.default_rtdose.data

    @property
    def dose_offset(self):
        return self.default_rtdose.offset

    @property
    def dose_size(self):
        return self.default_rtdose.size

    @property
    def dose_spacing(self):
        return self.default_rtdose.spacing

    @property
    def first_ct(self):
        return self.default_ct.first_ct

    @property
    def has_ct(self):
        return True if len(self.list_series('CT')) > 0 else False

    def has_landmark(self, *args, **kwargs):
        return self.default_rtstruct.has_landmark(*args, **kwargs)

    def has_regions(self, *args, **kwargs):
        return self.default_rtstruct.has_regions(*args, **kwargs)

    def landmark_data(self, *args, **kwargs):
        return self.default_rtstruct.landmark_data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs):
        return self.default_rtstruct.list_landmarks(*args, **kwargs)

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def list_series(
        self,
        modalities: Optional[Union[DicomModality, List[DicomModality]]] = None) -> List[SeriesID]:
        modalities = arg_to_list(modalities, DicomModality)
        index = self.__index.copy()
        if modalities is not None:
            index = index[index.modality.isin(modalities)]
        index = index.sort_values(['series-date', 'series-time'], ascending=[True, True])
        series_ids = list(index['series-id'].unique())
        return series_ids

    @property
    def id(self) -> StudyID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    @property
    def mr_data(self) -> Optional[MrImage]:
        return self.default_mr.data if self.default_mr is not None else None

    @property
    def mr_fov(self) -> Optional[Point3D]:
        return self.default_mr.fov if self.default_mr is not None else None

    @property
    def mr_offset(self) -> Optional[Point3D]:
        return self.default_mr.offset if self.default_mr is not None else None

    @property
    def mr_size(self) -> Optional[Size3D]:
        return self.default_mr.size if self.default_mr is not None else None

    @property
    def mr_spacing(self) -> Optional[Spacing3D]:
        return self.default_mr.spacing if self.default_mr is not None else None

    @property
    def patient(self) -> str:
        return self.__patient

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)

    @property
    def region_policy(self) -> pd.DataFrame:
        return self.__region_policy

    def series(
        self,
        id: SeriesID,
        modality: Optional[DicomModality] = None,
        **kwargs: Dict) -> DicomSeries:
        if modality is None:
            modality = self.series_modality(id)

        if modality == 'CT':
            return CtSeries(self, id, **kwargs)
        elif modality == 'MR':
            return MrSeries(self, id, **kwargs)
        elif modality == 'RTDOSE':
            return RtDoseSeries(self, id, **kwargs)
        elif modality == 'RTPLAN':
            return RtPlanSeries(self, id, **kwargs)
        elif modality == 'RTSTRUCT':
            return RtStructSeries(self, id, region_dups=self.__region_dups, region_map=self.__region_map, **kwargs)
        else:
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")

    def series_modality(
        self,
        id: SeriesID) -> DicomModality:
        # Get modality from index.
        index = self.__index.copy()
        index = index[index['series-id'] == id]
        if len(index) == 0:
            raise ValueError(f"Series '{id}' not found in study '{self}'.")
        modality = index.iloc[0]['modality']
        return modality

    def __load_default_rtplan_and_rtdose(self) -> None:
        if not self.__index_policy['rtplan']['no-ref-rtstruct']['allow']:
            # Get RTPLAN/RTDOSE linked to RTSTRUCT. No guarantees in 'index' building that
            # these RTPLAN/RTDOSE files are present.
            def_rtstruct = self.default_rtstruct.default_rtstruct
            def_rt_sop_id = def_rtstruct.rtstruct.SOPInstanceUID

            # Find RTPLANs that link to default RTSTRUCT.
            linked_rtplan_sop_ids = []
            linked_rtplan_series_ids = []
            rtplan_series_ids = self.list_series('RTPLAN')
            for rtplan_series_id in rtplan_series_ids:
                rtplan_series = self.series(rtplan_series_id, 'RTPLAN')
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
            def_rtplan_series = self.series(def_rtplan_series_id, 'RTPLAN')
            self.__default_rtplan = def_rtplan_series.rtplan(def_rtplan_sop_id)

        else:
            # Choose first RTPLAN from study.
            rtplan_series_ids = self.list_series('RTPLAN')
            if len(rtplan_series_ids) > 0:
                def_rtplan_series_id = rtplan_series_ids[-1]
                def_rtplan_series = self.series(def_rtplan_series_id, 'RTPLAN')
                def_rtplan_sop_id = def_rtplan_series.list_rtplans()[-1]
                self.__default_rtplan = def_rtplan_series.rtplan(def_rtplan_sop_id) 
            else:
                self.__default_rtplan = None

        if not self.__index_policy['rtdose']['no-ref-rtplan']['allow']:
            # Get RTDOSEs linked to first RTPLAN.
            linked_rtdose_series_ids = []
            linked_rtdose_sop_ids = []
            rtdose_series_ids = self.list_series('RTDOSE')
            for rtdose_series_id in rtdose_series_ids:
                rtdose_series = self.series(rtdose_series_id, 'RTDOSE')
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
            def_rtdose_series = self.series(def_rtdose_series_id, 'RTDOSE')
            self.__default_rtdose = def_rtdose_series.rtdose(def_rtdose_sop_id)

        elif self.__index_policy['rtdose']['no-ref-rtplan']['only'] == 'at-least-one-rtplan':
            # Choose first RTDOSE from study.
            rtdose_series_ids = self.list_series('RTDOSE')
            if len(rtdose_series_ids) > 0:
                def_rtdose_series_id = rtdose_series_ids[-1]
                def_rtdose_series = self.series(def_rtdose_series_id, 'RTDOSE')
                def_rtdose_sop_id = def_rtdose_series.list_rtdoses()[-1]
                self.__default_rtdose = def_rtdose_series.rtdose(def_rtdose_sop_id) 
            else:
                self.__default_rtdose = None

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"DicomStudy '{self}' not found in index for patient '{self.__patient}'.")

    def __str__(self) -> str:
        return self.__global_id
