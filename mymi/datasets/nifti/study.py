import numpy as np
import os
from typing import *

from mymi.typing import *

from ..mixins import IndexMixin
from ..region_map import RegionMap
from .series import *

class NiftiStudy(IndexMixin):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        id: StudyID,
        path: DirPath,
        check_path: bool = True,
        index: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__dataset_id = dataset_id
        self.__global_id = f'NIFTI:{dataset_id}:{pat_id}:{id}'
        self.__id = id
        self._index = index
        self.__path = path
        self.__pat_id = pat_id
        self.__region_map = region_map

    def series(
        self,
        id: NiftiSeriesID,
        modality: NiftiModality) -> Union[NiftiImageSeries, LandmarksSeries]:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            id = handle_idx_prefix(id, lambda: self.list_series('ct'))
            for e in image_extensions:
                filepath = os.path.join(self.__path, 'ct', f'{id}{e}')
                if os.path.exists(filepath):
                    return CtImageSeries(self.__dataset_id, self.__pat_id, self.__id, id, filepath)
            raise ValueError(f"No CtImageSeries '{id}' for study '{self}'.")
        elif modality == 'dose':
            id = handle_idx_prefix(id, lambda: self.list_series('dose'))
            for e in image_extensions:
                filepath = os.path.join(self.__path, 'dose', f'{id}{e}')
                if os.path.exists(filepath):
                    return DoseImageSeries(self.__dataset_id, self.__pat_id, self.__id, id, filepath)
            raise ValueError(f"No DoseImageSeries '{id}' for study '{self}'.")
        elif modality == 'landmarks':
            id = handle_idx_prefix(id, lambda: self.list_series('landmarks'))
            filepath = os.path.join(self.__path, 'landmarks', f'{id}.csv')
            if os.path.exists(filepath):
                ref_ct = self.default_series('ct')
                ref_dose = self.default_series('dose')
                return LandmarksSeries(self.__dataset_id, self.__pat_id, self.__id, id, filepath, ref_ct=ref_ct, ref_dose=ref_dose)
            else:
                raise ValueError(f"No LandmarksSeries '{id}' for study '{self}'.")
        elif modality == 'mr':
            id = handle_idx_prefix(id, lambda: self.list_series('mr'))
            for e in image_extensions:
                filepath = os.path.join(self.__path, 'mr', f'{id}{e}')
                if os.path.exists(filepath):
                    return MrImageSeries(self.__dataset_id, self.__pat_id, self.__id, id, filepath)
            raise ValueError(f"No MrImageSeries '{id}' for study '{self}'.")
        elif modality == 'regions':
            id = handle_idx_prefix(id, lambda: self.list_series('regions'))
            dirpath = os.path.join(self.__path, 'regions', str(id))
            if os.path.exists(dirpath):
                return RegionsImageSeries(self.__dataset_id, self.__pat_id, self.__id, id, dirpath, region_map=self.__region_map)
            else:
                raise ValueError(f"No RegionsImageSeries '{id}' for study '{self}'.")
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

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
            dirpath = os.path.join(self.__path, 'ct')
            ct_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            ct_ids = [i.replace(e, '') for i in ct_ids for e in image_extensions if i.endswith(e)]
            return ct_ids
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
        return self.__global_id

# Add properties.
props = ['global_id', 'id', 'path', 'patient']
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
    
# Add image filepath shortcuts from 'default_series(mod)'
mods = ['ct', 'mr', 'dose']
for m in mods:
    setattr(NiftiStudy, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_series(m), 'filepath') if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_series(mod)'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiStudy, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'list_{m}')(*args, **kwargs) if self.default_series(m) is not None else [])
    setattr(NiftiStudy, f'has_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'has_{m}')(*args, **kwargs) if self.default_series(m) is not None else False)
    setattr(NiftiStudy, f'{m[:-1]}_data', lambda self, *args, m=m, **kwargs: self.default_series(m).data(*args, **kwargs) if self.default_series(m) is not None else None)
