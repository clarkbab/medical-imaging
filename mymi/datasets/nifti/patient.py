import numpy as np
import os
import pandas as pd
from typing import *

from mymi import config
from mymi.typing import *
from mymi.utils import *

from ..dicom import DicomDataset, DicomPatient
from ..mixins import IndexMixin
from ..patient import Patient
from ..region_map import RegionMap
from .study import NiftiStudy

class NiftiPatient(IndexMixin, Patient):
    def __init__(
        self,
        dataset: DatasetID,
        id: PatientID,
        ct_from: Optional['NiftiPatient'] = None,
        index: Optional[pd.DataFrame] = None,
        excluded_labels: Optional[List[str]] = None,
        region_map: Optional[RegionMap] = None) -> None:
        super().__init__(dataset, id, ct_from=ct_from, index=index, region_map=region_map)
        self.__path = os.path.join(config.directories.datasets, 'nifti', self._dataset_id, 'data', 'patients', self._id)
        if not os.path.exists(self.__path):
            raise ValueError(f"No nifti patient '{self._id}' found at path: {self.__path}")

    @property
    def default_study(self) -> Optional[NiftiStudy]:
        studys = self.list_studies()
        return self.study(studys[-1]) if len(studys) > 0 else None

    @property
    def dicom(self) -> DicomPatient:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'dicom-dataset', 'dicom-patient-id']]
        index = index[(index['dataset'] == self._dataset_id) & (index['patient-id'] == self._id)].drop_duplicates()
        assert len(index) == 1
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id'])

    def has_study(
        self,
        study: StudyIDs,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_studies(study=study, **kwargs)
        req_ids = arg_to_list(study, StudyID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def list_studies(
        self,
        study: StudyIDs = 'all',    # Used for filtering.
        ) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        ids = list(sorted(os.listdir(self.__path)))
        if study != 'all':
            studys = arg_to_list(study, StudyID)
            all_ids = ids.copy()
            ids = []
            for i, id in enumerate(all_ids):
                # Check if any of the passed 'studys' references this ID.
                for j, sid in enumerate(studys):
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
        return super().__str__(self.__class__.__name__)

# Add shortcut properies from 'default_study'.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    setattr(NiftiPatient, f'has_{m}', property(lambda self, m=m: getattr(self.default_study, f'has_{m}') if self.default_study is not None else None))
    
# Add image filepath shortcuts from 'default_study'.
mods = ['ct', 'mr', 'dose']
for m in mods:
    setattr(NiftiPatient, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_study, f'{m}_filepath') if self.default_study is not None else None))
setattr(NiftiPatient, 'region_filepaths', lambda self, region: self.default_study.region_filepaths(region) if self.default_study is not None else None)

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
    setattr(NiftiPatient, f'{m}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m}_data')(*args, **kwargs) if self.default_study is not None else None)
