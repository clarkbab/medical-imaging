import numpy as np
import os
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from .study import NiftiStudy

class NiftiPatient:
    def __init__(
        self,
        dataset: 'NiftiDataset',
        id: PatientID,
        check_path: bool = True,
        ct_from: Optional[str] = None,
        index: Optional[pd.DataFrame] = None,
        excluded_labels: Optional[List[str]] = None,
        processed_labels: Optional[List[str]] = None,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__dataset = dataset
        self.__id = str(id)
        self.__global_id = f'{dataset}:{self.__id}'
        self.__index = index
        self.__region_map = region_map

        # Check that patient ID exists.
        self.__path = os.path.join(dataset.path, 'data', 'patients', self.__id)
        if check_path and not os.path.exists(self.__path):
            raise ValueError(f"Patient '{self}' not found. Filepath: '{self.__path}'.")

    def ct_extent(
        self,
        **kwargs) -> Optional[Union[FOV3D, Size3D]]:
        return self.default_study.ct_extent(**kwargs)

    @property
    def default_study(self) -> Optional[NiftiStudy]:
        study_ids = self.list_studies()
        if len(study_ids) == 0:
            return None
        else:
            return NiftiStudy(self, study_ids[-1])

    @property
    def origin(self) -> Dict[str, str]:
        if self.index is None:
            raise ValueError(f"No 'index.csv' provided for dataset '{self.__dataset}'.")
        info = self.index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id']}
        return info

    def list_studies(
        self,
        study_ids: StudyIDs = 'all',    # Used for filtering.
        ) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        pat_study_ids = list(sorted(os.listdir(self.__path)))
        if study_ids != 'all':
            study_ids = arg_to_list(study_ids, StudyID)
            pat_study_ids = [s for s in pat_study_ids if s in study_ids]
        return pat_study_ids

    def study(
        self,
        id: StudyID,
        **kwargs) -> NiftiStudy:
        if id.startswith('idx:'):
            idx = int(id.split(':')[1])
            study_ids = self.list_studies()
            if idx > len(study_ids) - 1:
                raise ValueError(f"Index {idx} out of range for patient with {len(study_ids)} studies.")
            id = study_ids[idx]
        # Filter index for relevant rows.
        index = self.index[self.index['nifti-study-id'] == id].copy() if self.index is not None else None
        return NiftiStudy(self, id, index=index, region_map=self.__region_map, **kwargs)

    def __str__(self) -> str:
        return self.__global_id

# Add properties.
props = ['dataset', 'global_id', 'id', 'index', 'path']
for p in props:
    setattr(NiftiPatient, p, property(lambda self, p=p: getattr(self, f'_{NiftiPatient.__name__}__{p}')))

# Add shortcut properies from 'default_study'.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    n = f'{m}_series' if m in ['landmarks', 'regions'] else m   # 'has_landmarks/regions' is reserved for LandmarkIDs.
    setattr(NiftiPatient, f'has_{n}', property(lambda self, m=m: getattr(self.default_study, f'has_{n}') if self.default_study is not None else None))

# Add image property shortcuts from 'default_study'.
mods = ['ct', 'dose', 'mr']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiPatient, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_study, f'{m}_{p}') if self.default_study is not None else None))

# Add landmark/region method shortcuts from 'default_study'.
mods = ['landmark', 'region']
for m in mods:
    setattr(NiftiPatient, f'has_{m}_ids', lambda self, m=m, **kwargs: getattr(self.default_study, f'has_{m}_ids')(**kwargs) if self.default_study is not None else False)
    setattr(NiftiPatient, f'list_{m}_ids', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}_ids')(*args, **kwargs) if self.default_study is not None else [])
    setattr(NiftiPatient, f'{m}s_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m}s_data')(*args, **kwargs) if self.default_study is not None else None)
