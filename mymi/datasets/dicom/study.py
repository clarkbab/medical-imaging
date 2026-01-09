from datetime import datetime as dt
import pandas as pd
from typing import *

from mymi.constants import *
from mymi.typing import *
from mymi.utils import *

from ..mixins import IndexWithErrorsMixin
from ..region_map import RegionMap
from ..study import Study
from .series import *

class DicomStudy(IndexWithErrorsMixin, Study):
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        id: StudyID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any],
        index_errors: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        ct_from: Optional['DicomStudy'] = None,
        region_map: Optional[RegionMap] = None):
        super().__init__(dataset, pat, id, config=config, ct_from=ct_from, region_map=region_map)
        self.__ct_from = ct_from
        self._index = index
        self._index_errors = index_errors
        self._index_policy = index_policy

    @property
    def date(self) -> str:
        date_str = self._index['study-date'].iloc[0]
        time_str = self._index['study-time'].iloc[0]
        return f'{date_str}:{time_str}'

    @property
    def datetime(self) -> dt:
        parsed_dt = dt.strptime(self.date, f'{DICOM_DATE_FORMAT}:{DICOM_TIME_FORMAT}')
        return parsed_dt

    def default_series(
        self,
        modality: DicomModality,
        show_warning: bool = True) -> Optional[DicomSeries]:
        serieses = self.list_series(modality)
        if show_warning and len(serieses) > 1:
            logging.warning(f"More than one '{modality}' series found for '{self}', defaulting to latest.")
        return self.series(serieses[-1], modality) if len(serieses) > 0 else None

    def has_series(
        self,
        series: SeriesIDs,
        modality: DicomModality,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_series(modality, series=series, **kwargs)
        req_ids = arg_to_list(series, SeriesID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    @alias_kwargs((
        ('sd', 'show_date'),
        ('sf', 'shof_filepath'),
    ))
    def list_series(
        self,
        modality: DicomModality,
        series: SeriesIDs = 'all',
        show_date: bool = False,
        show_filepath: bool = False) -> List[SeriesID]:
        if modality not in DicomModality.__args__:
            raise ValueError(f"Unrecognised modality '{modality}'. Should be one of {DicomModality.__args__}.")
        index = self.__ct_from.index().copy() if modality == 'ct' and self.__ct_from is not None else self._index.copy()
        index = index[index['modality'] == modality]
        index = index.sort_values(['series-date', 'series-time'], ascending=[True, True])
        ids = list(index['series-id'].unique())

        # Filter by 'series'.
        if series != 'all':
            serieses = arg_to_list(series, SeriesID)
            ids = [i for i in ids if i in serieses]

        # Add extra info if requested.
        def append_info(s: SeriesID) -> str:
            series = self.series(s, modality)
            if show_date:
                s = f'{s} ({series.date})'
            if show_filepath:
                s = f'{s} ({series.filepath})'
            return s
        if show_date or show_filepath:
            ids = [append_info(i) for i in ids]

        return ids

    def series(
        self,
        id: SeriesID,
        modality: Optional[DicomModality] = None,
        **kwargs: Dict) -> DicomSeries:
        if modality is None:
            modality = self.series_modality(id)
        elif modality not in DicomModality.__args__:
            raise ValueError(f"Unrecognised modality '{modality}'. Should be one of {DicomModality.__args__}.")
        else:
            id = handle_idx_prefix(id, lambda: self.list_series(modality))

        if not self.has_series(id, modality):
            raise ValueError(f"{modality.upper()} series '{id}' not found for study '{self}'.")

        # Get series-specific data.
        index = self._index[(self._index['modality'] == modality) & (self._index['series-id'] == id)].copy()
        if modality in ['rtdose', 'rtplan', 'rtstruct']:
            index = index.iloc[0]
        index_policy = self._index_policy[modality]

        if modality == 'ct':
            if self.__ct_from is not None:
                return self.__ct_from.series(id, modality, **kwargs)
            else:
                return DicomCtSeries(self._dataset_id, self._pat_id, self._id, id, index, index_policy, **kwargs)
        elif modality == 'mr':
            return DicomMrSeries(self._dataset_id, self._pat_id, self._id, id, index, index_policy, **kwargs)
        elif modality == 'rtdose':
            return DicomRtDoseSeries(self._dataset_id, self._pat_id, self._id, id, index, index_policy, **kwargs)
        elif modality == 'rtplan':
            return DicomRtPlanSeries(self._dataset_id, self._pat_id, self._id, id, index, index_policy, **kwargs)
        elif modality == 'rtstruct':
            ref_study = self.__ct_from if self.__ct_from is not None else self
            ref_ct_id = index['mod-spec'][DICOM_RTSTRUCT_REF_CT_KEY]
            if not index_policy['no-ref-ct']['allow']:
                # Require use of the referenced CT.
                ref_ct = ref_study.series(ref_ct_id, 'ct')
            else:
                # Preference the referenced ct, but allow the first ct in the study otherwise.
                if ref_study.has_series(ref_ct_id, 'ct'):
                    ref_ct = ref_study.series(ref_ct_id, 'ct')
                else:
                    ref_ct = ref_study.default_series('ct')

            return DicomRtStructSeries(self._dataset_id, self._pat_id, self._id, id, ref_ct, index, index_policy, config=self._config, region_map=self._region_map, **kwargs)

    def series_modality(
        self,
        id: SeriesID) -> DicomModality:
        # Get modality from index.
        index = self._index.copy()
        index = index[index['series-id'] == id]
        if len(index) == 0:
            raise ValueError(f"Series '{id}' not found in study '{self}'.")
        modality = index.iloc[0]['modality']
        return modality

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add 'has_{mod}' properties.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'has_{m}', property(lambda self, m=m: self.default_series(m, show_warning=False) is not None))

# Add 'list_{mod}_series' methods.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'list_{m}_series', lambda self, m=m: self.list_series(m))

# Add '{mod}_series' methods.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'{m}_series', lambda self, series, m=m: self.series(series, m))

# Add 'default_{mod}' properties.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomStudy, f'default_{m}', property(lambda self, m=m: self.default_series(m)))

# Add image property shortcuts from 'default_series(mod)' methods.
mods = ['ct', 'mr', 'rtdose']
props = ['data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    n = 'dose' if m == 'rtdose' else m
    for p in props:
        setattr(DicomStudy, f'{n}_{p}', property(lambda self, m=m, n=n, p=p: getattr(self.default_series(m), p) if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_rtstruct'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(DicomStudy, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_rtstruct, f'has_{m[:-1]}')(*args, **kwargs) if self.default_rtstruct is not None else False)
    setattr(DicomStudy, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_rtstruct, f'list_{m}')(*args, **kwargs) if self.default_rtstruct is not None else [])
    setattr(DicomStudy, f'{m}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_rtstruct, f'{m}_data')(*args, **kwargs) if self.default_rtstruct is not None else None)
