import numpy as np
import os
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..mixins import IndexMixin
from ..region_map import RegionMap
from .study import NiftiStudy

class NiftiPatient(IndexMixin):
    def __init__(
        self,
        dataset_id: DatasetID,
        id: PatientID,
        path: str,
        check_path: bool = True,
        ct_from: Optional['NiftiDataset'] = None,
        index: Optional[pd.DataFrame] = None,
        excluded_labels: Optional[List[str]] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__ct_from = ct_from
        self.__dataset_id = dataset_id
        self.__global_id = f'NIFTI:{dataset_id}:{id}'
        self.__id = str(id)
        self._index = index
        self.__path = path
        self.__region_map = region_map

    @property
    def default_study(self) -> Optional[NiftiStudy]:
        study_ids = self.list_studies()
        return self.study(study_ids[-1]) if len(study_ids) > 0 else None

    @property
    def origin(self) -> Dict[str, str]:
        if self._index is None:
            raise ValueError(f"Dataset has no 'index.csv'.")
        info = self._index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id']}
        return info

    def list_studies(
        self,
        study_ids: StudyIDs = 'all',    # Used for filtering.
        ) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        ids = list(sorted(os.listdir(self.__path)))
        if study_ids != 'all':
            study_ids = arg_to_list(study_ids, StudyID)
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

    def __repr__(self) -> str:
        return str(self)

    def study(
        self,
        id: StudyID,
        **kwargs) -> NiftiStudy:
        id = handle_idx_prefix(id, self.list_studies)
        index = self._index[self._index['study-id'] == id].copy() if self._index is not None else None

        # Check that study ID exists.
        path = os.path.join(self.__path, str(id))
        if not os.path.exists(path):
            raise ValueError(f"Study '{id}' not found for patient '{self}'.")

        return NiftiStudy(self.__dataset_id, self.__id, id, path, index=index, region_map=self.__region_map, **kwargs)

    def __str__(self) -> str:
        return self.__global_id

# Add properties.
props = ['global_id', 'id', 'path']
for p in props:
    setattr(NiftiPatient, p, property(lambda self, p=p: getattr(self, f'_{NiftiPatient.__name__}__{p}')))

# Add shortcut properies from 'default_study'.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    n = f'{m}_data' if m in ['landmarks', 'regions'] else m   # 'has_landmarks/regions' is reserved for LandmarkIDs.
    setattr(NiftiPatient, f'has_{n}', property(lambda self, m=m: getattr(self.default_study, f'has_{n}') if self.default_study is not None else None))

# Add image property shortcuts from 'default_study'.
mods = ['ct', 'dose', 'mr']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiPatient, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_study, f'{m}_{p}') if self.default_study is not None else None))

# Add landmark/region method shortcuts from 'default_study'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiPatient, f'has_{m}', lambda self, m=m, **kwargs: getattr(self.default_study, f'has_{m}')(**kwargs) if self.default_study is not None else False)
    setattr(NiftiPatient, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}')(*args, **kwargs) if self.default_study is not None else [])
    setattr(NiftiPatient, f'{m[:-1]}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m[:-1]}_data')(*args, **kwargs) if self.default_study is not None else None)
