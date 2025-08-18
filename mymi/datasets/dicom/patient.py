from datetime import datetime
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..mixins import IndexWithErrorsMixin
from ..region_map import RegionMap
from .series import * 
from .study import DicomStudy

class DicomPatient(IndexWithErrorsMixin):
    def __init__(
        self,
        dataset_id: DatasetID,
        id: PatientID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any],
        index_errors: pd.DataFrame,
        ct_from: Optional['DicomPatient'] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__ct_from = ct_from
        self.__dataset_id = dataset_id
        self._index_errors = index_errors
        self.__global_id = f'DICOM:{dataset_id}:{id}'
        self.__id = str(id)
        self._index = index
        self._index_policy = index_policy
        self.__region_map = region_map

    @property
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate
    
    @property
    def default_study(self) -> Optional[DicomStudy]:
        study_ids = self.list_studies()
        if len(study_ids) > 0:
            return self.study(study_ids[-1])
        else:
            return None

    @property
    def name(self) -> str:
        return self.get_cts()[0].PatientName

    @property
    def sex(self) -> str:
        return self.get_cts()[0].PatientSex

    @property
    def size(self) -> str:
        return getattr(self.get_cts()[0], 'PatientSize', '')

    @property
    def weight(self) -> str:
        return getattr(self.get_cts()[0], 'PatientWeight', '')

    def info(self) -> pd.DataFrame:
        # Define dataframe structure.
        cols = {
            'age': str,
            'birth-date': str,
            'name': str,
            'sex': str,
            'size': str,
            'weight': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        for col in cols.keys():
            col_method = col.replace('-', '_')
            data[col] = getattr(self, col_method)

        # Add row.
        df = append_row(df, data)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    def has_studies(
        self,
        study_ids: StudyIDs,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_studies(study_ids=study_ids, **kwargs)
        req_ids = arg_to_list(study_ids, StudyID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def list_studies(
        self,
        show_datetime: bool = False,
        study_ids: StudyIDs = 'all') -> List[StudyID]:
        # Sort studies by date/time - oldest first.
        ids = list(self._index.sort_values(['study-date', 'study-time'])['study-id'].unique())
        
        # Filter by 'study_ids'.
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

        if show_datetime:
            ids = [f'{i} ({self.study(i).date})' for i in ids]

        return ids

    def __repr__(self) -> str:
        return str(self)

    def study(
        self,
        id: StudyID) -> DicomStudy:
        id = handle_idx_prefix(id, self.list_studies)
        if not self.has_studies(id):
            raise ValueError(f"Study '{id}' not found for patient '{self}'.")
        index = self._index[self._index['study-id'] == str(id)].copy()
        index_errors = self._index_errors[self._index_errors['study-id'] == str(id)].copy()
        ct_from = self.__ct_from.study(id) if self.__ct_from is not None and self.__ct_from.has_studies(id) else None
        return DicomStudy(self.__dataset_id, self.__id, id, index, self._index_policy, index_errors, ct_from=ct_from, region_map=self.__region_map)

    def __str__(self) -> str:
        return self.__global_id
    
# Add properties.
props = ['ct_from', 'global_id', 'id', 'index_policy', 'path', 'region_map']
for p in props:
    setattr(DicomPatient, p, property(lambda self, p=p: getattr(self, f'_{DicomPatient.__name__}__{p}')))

# Add 'default_{mod}' property shortcuts from 'default_study'.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))

# Add 'has_{mod}' property shortcuts from 'default_study'.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomPatient, f'has_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))

# Add image property shortcuts from 'default_study'.
mods = ['ct', 'mr', 'rtdose']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    n = 'dose' if m == 'rtdose' else m  # Rename 'rtdose' to 'dose'.
    for p in props:
        setattr(DicomPatient, f'{n}_{p}', property(lambda self, m=m, n=n, p=p: getattr(self.default_study, f'{n}_{p}') if self.default_study is not None else None))

# Add landmark/region method shortcuts from 'default_study'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(DicomPatient, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'has_{m[:-1]}')(*args, **kwargs) if self.default_study is not None else False)
    setattr(DicomPatient, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}')(*args, **kwargs) if self.default_study is not None else [])
    setattr(DicomPatient, f'{m[:-1]}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m[:-1]}_data')(*args, **kwargs) if self.default_study is not None else None)
 