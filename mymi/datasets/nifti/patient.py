import numpy as np
import os
import pandas as pd
from typing import *

from mymi import config
from mymi.typing import *
from mymi.utils import *

from ..mixins import IndexMixin
from ..patient import Patient
from ..region_map import RegionMap
from .study import NiftiStudy

class NiftiPatient(IndexMixin, Patient):
    def __init__(
        self,
        dataset_id: DatasetID,
        id: PatientID,
        ct_from: Optional['NiftiPatient'] = None,
        index: Optional[pd.DataFrame] = None,
        excluded_labels: Optional[List[str]] = None,
        region_map: Optional[RegionMap] = None) -> None:
        # Check that patient ID exists.
        path = os.path.join(config.directories.datasets, 'nifti', str(dataset_id), 'data', 'patients', str(id))
        if not os.path.exists(path):
            raise ValueError(f"NiftiPatient '{id}' not found for dataset '{self}'. Dirpath: {path}")
        self.__path = path
        super().__init__(dataset_id, id, ct_from=ct_from, region_map=region_map)
        self._index = index

    @property
    def default_study(self) -> Optional[NiftiStudy]:
        study_ids = self.list_studies()
        return self.study(study_ids[-1]) if len(study_ids) > 0 else None

    def has_study(
        self,
        study_id: StudyIDs,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_studies(study_id=study_id, **kwargs)
        req_ids = arg_to_list(study_id, StudyID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def list_studies(
        self,
        study_id: StudyIDs = 'all',    # Used for filtering.
        ) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        ids = list(sorted(os.listdir(self.__path)))
        if study_id != 'all':
            study_ids = arg_to_list(study_id, StudyID)
            all_ids = ids.copy()
            ids = []
            for i, id in enumerate(all_ids):
                # Check if any of the passed 'study_ids' references this ID.
                for j, sid in enumerate(study_ids):
                    if sid.startswith('idx:'):
                        # Check if idx refer
                        idx = int(sid.split(':')[1])
                        if i == idx or (idx < 0 and i == len(all_ids) + idx):   # Allow negative indexing.
                            ids.append(id)
                            break
                    elif id == sid:
                        ids.append(id)
                        break

        return ids

    @property
    def origin(self) -> Dict[str, str]:
        if self._index is None:
            raise ValueError(f"Dataset has no 'index.csv'.")
        info = self._index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id']}
        return info

    def study(
        self,
        id: StudyID,
        **kwargs) -> NiftiStudy:
        id = handle_idx_prefix(id, self.list_studies)
        index = self._index[self._index['study-id'] == id].copy() if self._index is not None else None

        # Get 'ct_from' study.
        if self._ct_from is not None and self._ct_from.has_study(id):
            ct_from = self._ct_from.study(id)
        else:
            ct_from = None

        return NiftiStudy(self._dataset_id, self._id, id, ct_from=ct_from, index=index, region_map=self._region_map, **kwargs)

    def __str__(self) -> str:
        if self._ct_from is None:
            return f"NiftiPatient({self._id}, dataset={self._dataset_id})"
        else:
            return f"NiftiPatient({self._id}, dataset={self._dataset_id}, ct_from={self._ct_from.dataset_id})"

# Add shortcut properies from 'default_study'.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    n = f'{m}_data' if m in ['landmarks', 'regions'] else m   # 'has_landmarks/regions' is reserved for LandmarkIDs.
    setattr(NiftiPatient, f'has_{n}', property(lambda self, m=m: getattr(self.default_study, f'has_{n}') if self.default_study is not None else None))
    
# Add image filepath shortcuts from 'default_study'.
mods = ['ct', 'mr', 'dose']
for m in mods:
    setattr(NiftiPatient, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_study, f'{m}_filepath') if self.default_study is not None else None))
setattr(NiftiPatient, 'region_filepaths', lambda self, region_id: self.default_study.region_filepaths(region_id) if self.default_study is not None else None)

# Add image property shortcuts from 'default_study'.
mods = ['ct', 'dose', 'mr']
props = ['data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiPatient, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_study, f'{m}_{p}') if self.default_study is not None else None))

# Add landmark/region method shortcuts from 'default_study'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiPatient, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'has_{m[:-1]}')(*args, **kwargs) if self.default_study is not None else False)
    setattr(NiftiPatient, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}')(*args, **kwargs) if self.default_study is not None else [])
    setattr(NiftiPatient, f'{m[:-1]}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m[:-1]}_data')(*args, **kwargs) if self.default_study is not None else None)
