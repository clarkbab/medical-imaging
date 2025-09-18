import numpy as np
import os
from typing import *

from mymi import config
from mymi.typing import *

from ..mixins import IndexMixin
from ..region_map import RegionMap
from ..study import Study
from .series import *

class NiftiStudy(IndexMixin, Study):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        id: StudyID,
        ct_from: Optional['NiftiStudy'] = None,
        index: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None) -> None:
        # Check that study ID exists.
        path = os.path.join(config.directories.datasets, 'nifti', str(dataset_id), 'data', 'patients', str(pat_id), str(id))
        if not os.path.exists(path):
            raise ValueError(f"NiftiStudy '{id}' not found for patient '{pat_id}'. Dirpath: {path}")
        self.__path = path
        super().__init__(dataset_id, pat_id, id, ct_from=ct_from, region_map=region_map)
        self._index = index

    def series(
        self,
        id: NiftiSeriesID,
        modality: NiftiModality) -> Union[NiftiImageSeries, NiftiLandmarksSeries]:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            id = handle_idx_prefix(id, lambda: self.list_series('ct'))
            if self._ct_from is None:
                return NiftiCtSeries(self._dataset_id, self._pat_id, self._id, id)
            else:
                return self._ct_from.series(id, modality)
        elif modality == 'dose':
            id = handle_idx_prefix(id, lambda: self.list_series('dose'))
            return NiftiDoseSeries(self._dataset_id, self._pat_id, self._id, id)
        elif modality == 'landmarks':
            id = handle_idx_prefix(id, lambda: self.list_series('landmarks'))
            ref_ct = self.default_series('ct')
            ref_dose = self.default_series('dose')
            return NiftiLandmarksSeries(self._dataset_id, self._pat_id, self._id, id, ref_ct=ref_ct, ref_dose=ref_dose)
        elif modality == 'mr':
            id = handle_idx_prefix(id, lambda: self.list_series('mr'))
            return NiftiMrSeries(self._dataset_id, self._pat_id, self._id, id)
        elif modality == 'regions':
            id = handle_idx_prefix(id, lambda: self.list_series('regions'))
            return NiftiRegionsSeries(self._dataset_id, self._pat_id, self._id, id, region_map=self._region_map)
        else:
            raise ValueError(f"Unknown NiftiSeries modality '{modality}'.")

    def default_series(
        self,
        modality: NiftiModality) -> Optional[NiftiSeries]:
        series_ids = self.list_series(modality)
        if len(series_ids) > 1:
            logging.warning(f"More than one '{modality}' series found for '{self}', defaulting to latest.")
        return self.series(series_ids[-1], modality) if len(series_ids) > 0 else None

    def has_series(
        self,
        id: NiftiSeriesID,
        modality: NiftiModality) -> bool:
        return id in self.list_series(modality)

    def list_series(
        self,
        modality: NiftiModality) -> List[NiftiSeriesID]:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            if self._ct_from is None:
                dirpath = os.path.join(self.__path, 'ct')
                ct_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
                ct_ids = [i.replace(e, '') for i in ct_ids for e in image_extensions if i.endswith(e)]
                return ct_ids
            else:
                return self._ct_from.list_series(modality)
        elif modality == 'dose':
            dirpath = os.path.join(self.__path, 'dose')
            dose_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            dose_ids = [i.replace(e, '') for i in dose_ids for e in image_extensions if i.endswith(e)]
            return dose_ids
        elif modality == 'landmarks':
            dirpath = os.path.join(self.__path, 'landmarks')
            landmarks_ids = list(sorted(f.replace('.csv', '') for f in os.listdir(dirpath))) if os.path.exists(dirpath) else []
            return landmarks_ids
        elif modality == 'mr':
            dirpath = os.path.join(self.__path, 'mr')
            mr_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            mr_ids = [i.replace(e, '') for i in mr_ids for e in image_extensions if i.endswith(e)]
            return mr_ids
        elif modality == 'regions':
            dirpath = os.path.join(self.__path, 'regions')
            regions_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            return regions_ids
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

    @property
    def origin(self) -> Dict[str, str]:
        if self._index is None:
            raise ValueError(f"No 'index.csv' provided for dataset '{self.__patient.dataset}'.")
        info = self._index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id', 'dicom-study-id']}
        return info

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self._ct_from is None:
            return f"NiftiStudy({self._id}, dataset={self._dataset_id}, pat_id={self._pat_id})"
        else:
            return f"NiftiStudy({self._id}, dataset={self._dataset_id}, pat_id={self._pat_id}, ct_from={self._ct_from.dataset_id})"
    
# Add 'has_{mod}' properties.
# 'has_landmarks/regions' are reserved for Landmark/RegionIDs, use 'has_{mod}_series' for series checks.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    n = f'{m}_series' if m in ['landmarks', 'regions'] else m
    setattr(NiftiStudy, f'has_{n}', property(lambda self, m=m: self.default_series(m) is not None))

# Add 'default_{mod}' properties.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'default_{m}', property(lambda self, m=m: self.default_series(m)))
    
# Add image filepath shortcuts from 'default_series(mod)'
mods = ['ct', 'mr', 'dose']
for m in mods:
    setattr(NiftiStudy, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_series(m), 'filepath') if self.default_series(m) is not None else None))
setattr(NiftiStudy, 'region_filepaths', lambda self, region_id: self.default_series('regions').filepaths(region_id) if self.default_series('regions') is not None else None)

# Add image property shortcuts from 'default_series(mod)'.
mods = ['ct', 'mr', 'dose']
props = ['data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_series(m), p) if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_series(mod)'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiStudy, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'has_{m[:-1]}')(*args, **kwargs) if self.default_series(m) is not None else False)
    setattr(NiftiStudy, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'list_{m}')(*args, **kwargs) if self.default_series(m) is not None else [])
    setattr(NiftiStudy, f'{m[:-1]}_data', lambda self, *args, m=m, **kwargs: self.default_series(m).data(*args, **kwargs) if self.default_series(m) is not None else None)
