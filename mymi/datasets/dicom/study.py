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
        region_map: Optional[RegionMap] = None):
        self.__id = id
        self.__patient = patient
        self.__global_id = f"{patient}:{id}"
        self.__region_map = region_map

        # Get index.
        index = self.__patient.index
        index = index[index['study-id'] == id].copy()
        if len(index) == 0:
            raise ValueError(f"Study '{id}' not found for patient '{patient}'.")
        self.__index = index

        # Get policies.
        self.__index_policy = self.__patient.index_policy
        self.__region_policy = self.__patient.region_policy

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
    def default_rtdose(self) -> Optional[RtDoseSeries]:
        series_ids = self.list_series('RTDOSE')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'RTDOSE')

    @property
    def default_rtplan(self) -> Optional[RtPlanSeries]:
        series_ids = self.list_series('RTPLAN')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'RTPLAN')
    
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
    def dose_data(self) -> Optional[DoseImage]:
        return self.default_rtdose.data if self.default_rtdose is not None else None 

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
            return RtStructSeries(self, id, region_map=self.__region_map, **kwargs)
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

    def __str__(self) -> str:
        return self.__global_id
