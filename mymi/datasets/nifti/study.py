import numpy as np
import os
from typing import *

from mymi.typing import *

from .series import *

class NiftiStudy:
    def __init__(
        self,
        patient: 'NiftiPatient',
        id: StudyID,
        check_path: bool = True,
        index: Optional[pd.DataFrame] = None,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__global_id = f'{patient}:{id}'
        self.__id = id
        self.__index = index
        self.__path = os.path.join(patient.path, id)
        self.__patient = patient
        self.__region_map = region_map

        # Check that study ID exists.
        self.__path = os.path.join(patient.path, self.__id)
        if check_path and not os.path.exists(self.__path):
            raise ValueError(f"Study '{self}' not found. Filepath: '{self.__path}'.")

    def ct(
        self,
        id: SeriesID) -> CtImage:
        return CtImage(self, id)

    def ct_extent(
        self,
        **kwargs) -> Optional[Point3D]:
        return self.default_ct.extent(**kwargs) if self.default_ct is not None else None

    def default_series(
        self,
        modality: NiftiModality) -> Optional[SeriesID]:
        series_ids = self.list_series(modality)
        if modality == 'ct':
            return CtImage(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'dose':
            return DoseImage(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'landmarks':
            return LandmarksSeries(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'mr':
            return MrImage(self, series_ids[-1]) if len(series_ids) > 0 else None
        elif modality == 'regions':
            return RegionsImage(self, series_ids[-1], region_map=self.__region_map) if len(series_ids) > 0 else None
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

    def dose(
        self,
        id: SeriesID) -> DoseImage:
        return DoseImage(self, id)

    def has_series(
        self,
        id: SeriesID,
        modality: NiftiModality) -> bool:
        return id in self.list_series(modality)

    def list_series(
        self,
        modality: NiftiModality) -> List[SeriesID]:
        if modality == 'ct':
            filepath = os.path.join(self.__path, 'ct')
            ct_ids = list(sorted(f.replace('.nii.gz', '') for f in os.listdir(filepath))) if os.path.exists(filepath) else []
            return ct_ids
        elif modality == 'dose':
            filepath = os.path.join(self.__path, 'dose')
            dose_ids = list(sorted(f.replace('.nii.gz', '') for f in os.listdir(filepath))) if os.path.exists(filepath) else []
            return dose_ids
        elif modality == 'landmarks':
            filepath = os.path.join(self.__path, 'landmarks')
            landmarks_ids = list(sorted(f.replace('.csv', '') for f in os.listdir(filepath))) if os.path.exists(filepath) else []
            return landmarks_ids
        elif modality == 'mr':
            filepath = os.path.join(self.__path, 'mr')
            mr_ids = list(sorted(f.replace('.nii.gz', '') for f in os.listdir(filepath))) if os.path.exists(filepath) else []
            return mr_ids
        elif modality == 'regions':
            filepath = os.path.join(self.__path, 'regions')
            regions_ids = list(sorted(f.replace('.nii.gz', '') for f in os.listdir(filepath))) if os.path.exists(filepath) else []
            return regions_ids
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

    @property
    def origin(self) -> Dict[str, str]:
        if self.__index is None:
            raise ValueError(f"No 'index.csv' provided for dataset '{self.__patient.dataset}'.")
        info = self.__index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id', 'dicom-study-id']}
        return info

    def regions(
        self,
        id: SeriesID) -> RegionsImage:
        return RegionsImage(self, id, region_map=self.__region_map)

    def series(
        self,
        id: SeriesID,
        modality: NiftiModality) -> Union[NiftiImage, LandmarksSeries]:
        if modality == 'ct':
            return self.ct(id)
        elif modality == 'dose':
            return self.dose(id)
        elif modality == 'landmarks':
            return LandmarksSeries(self, id)
        elif modality == 'mr':
            return MrImage(self, id)
        elif modality == 'regions':
            return RegionsImage(self, id, region_map=self.__region_map)
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

    def __str__(self) -> str:
        return self.__global_id

# Add properties.
props = ['global_id', 'id', 'index', 'path', 'patient']
for p in props:
    setattr(NiftiStudy, p, property(lambda self, p=p: getattr(self, f'_{NiftiStudy.__name__}__{p}')))
    
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

# Add image property shortcuts from 'default_series(mod)'.
mods = ['ct', 'mr', 'dose']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_series(m), p) if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_series(mod)'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiStudy, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'list_{m}')(*args, **kwargs) if self.default_series(m) is not None else [])
    setattr(NiftiStudy, f'has_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'has_{m}')(*args, **kwargs) if self.default_series(m) is not None else False)
    setattr(NiftiStudy, f'{m}_data', lambda self, *args, m=m, **kwargs: self.default_series(m).data(*args, **kwargs) if self.default_series(m) is not None else None)
