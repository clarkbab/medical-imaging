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
    def default_ct(self) -> Optional[CtSeries]:
        series_ids = self.list_series('ct')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'ct')

    @property
    def default_mr(self) -> Optional[MrSeries]:
        series_ids = self.list_series('mr')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'mr')

    @property
    def default_rtdose(self) -> Optional[RtDoseSeries]:
        series_ids = self.list_series('rtdose')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'rtdose')

    @property
    def default_rtplan(self) -> Optional[RtPlanSeries]:
        series_ids = self.list_series('rtplan')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'rtplan')
    
    @property
    def default_rtstruct(self) -> Optional[RtStructSeries]:
        series_ids = self.list_series('rtstruct')
        if len(series_ids) == 0:
            return None
        return self.series(series_ids[-1], 'rtstruct')

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def has_ct(self):
        return True if len(self.list_series('ct')) > 0 else False

    @property
    def has_rtdose(self):
        return True if len(self.list_series('rtdose')) > 0 else False

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
    def patient(self) -> str:
        return self.__patient

    def region_images(self, *args, **kwargs):
        return self.default_rtstruct.region_images(*args, **kwargs)

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

        if modality == 'ct':
            return CtSeries(self, id, **kwargs)
        elif modality == 'mr':
            return MrSeries(self, id, **kwargs)
        elif modality == 'rtdose':
            return RtDoseSeries(self, id, **kwargs)
        elif modality == 'rtplan':
            return RtPlanSeries(self, id, **kwargs)
        elif modality == 'rtstruct':
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

# Add 'has_{mod}' properties.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'has_{m}', property(lambda self, m=m: getattr(self, f'default_{m}') is not None))

# Add image property shortcuts.
mods = ['ct', 'mr', 'rtdose']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(DicomStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(getattr(self, f'default_{m}'), p) if getattr(self, f'default_{m}') is not None else None))
