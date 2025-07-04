from datetime import datetime
import pandas as pd
from typing import *

from mymi.constants import *
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

        # Get study error index.
        error_index = self.__patient.error_index
        error_index = error_index[error_index['study-id'] == id].copy()
        self.__error_index = error_index

        # Get policies.
        self.__index_policy = self.__patient.index_policy
        self.__region_policy = self.__patient.region_policy

    @property
    def date(self) -> datetime:
        date_str = str(self.__index['study-date'].iloc[0])
        dt = datetime.strptime(date_str, DICOM_DATE_FORMAT)
        return dt

    def default_series(
        self,
        modality: DicomModality) -> Optional[DicomSeries]:
        series_ids = self.list_series(modality)
        if modality == 'ct':
            return CtSeries(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'mr':
            return MrSeries(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'rtdose':
            return RtDoseSeries(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'rtplan':
            return RtPlanSeries(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'rtstruct':
            return RtStructSeries(self, series_ids[-1], region_map=self.__region_map) if len(series_ids) > 0 else None
        else:
            raise ValueError(f"Unrecognised modality '{modality}'.")

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

# Add properties.
props = ['error_index', 'global_id', 'id', 'index', 'index_policy', 'patient', 'region_map', 'region_policy']
for p in props:
    setattr(DicomStudy, p, property(lambda self, p=p: getattr(self, f'_{DicomStudy.__name__}__{p}')))

# Add 'has_{mod}' properties.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'has_{m}', property(lambda self, m=m: self.default_series(m) is not None))

# Add 'default_{mod}' properties.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'default_{m}', property(lambda self, m=m: self.default_series(m)))

# Add image property shortcuts from 'default_series(mod)' methods.
mods = ['ct', 'mr', 'rtdose']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(DicomStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_series(m), p) if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_rtstruct'.
mods = ['landmark', 'region']
for m in mods:
    setattr(DicomStudy, f'has_{m}s', lambda self, *args, m=m, **kwargs: getattr(self.default_rtstruct, f'has_{m}s')(*args, **kwargs) if self.default_rtstruct is not None else False)
    setattr(DicomStudy, f'list_{m}s', lambda self, *args, m=m, **kwargs: getattr(self.default_rtstruct, f'list_{m}s')(*args, **kwargs) if self.default_rtstruct is not None else [])
    setattr(DicomStudy, f'{m}s_data', lambda self, *args, m=m, **kwargs: getattr(self.default_rtstruct, f'{m}s_data')(*args, **kwargs) if self.default_rtstruct is not None else None)
