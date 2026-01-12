from datetime import datetime
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..mixins import IndexWithErrorsMixin
from ..patient import Patient
from ..region_map import RegionMap
from .series import * 
from .study import DicomStudy

class DicomPatient(IndexWithErrorsMixin, Patient):
    def __init__(
        self,
        dataset: 'DicomDataset',
        id: PatientID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any],
        index_errors: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        ct_from: Optional['DicomPatient'] = None,
        region_map: Optional[RegionMap] = None) -> None:
        super().__init__(dataset, id, config=config, ct_from=ct_from, region_map=region_map)
        self._index_errors = index_errors
        self._index = index
        self._index_policy = index_policy

    @property
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate
    
    @property
    def default_study(self) -> Optional[DicomStudy]:
        studys = self.list_studies()
        if len(studys) > 0:
            return self.study(studys[-1])
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
        show_datetime: bool = False,
        sort: Optional[Callable[DicomStudy, int]] = None,
        study: StudyIDs = 'all') -> List[StudyID]:
        # Sort studies by date/time - oldest first.
        ids = list(self._index.sort_values(['study-date', 'study-time'])['study-id'].unique())
        
        # Filter by 'study'.
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

        # Sort by custom function.
        if sort is not None:
            studies = sorted([self.study(i) for i in ids], key=sort)
            ids = [s.id for s in studies]

        if show_datetime:
            ids = [f'{i} ({self.study(i).date})' for i in ids]

        return ids

    def study(
        self,
        id: StudyID,
        sort: Optional[Callable[DicomStudy, int]] = None,  # For 'idx:n' calls.
        ) -> DicomStudy:
        id = handle_idx_prefix(id, lambda: self.list_studies(sort=sort))
        if not self.has_study(id):
            raise ValueError(f"Study '{id}' not found for patient '{self}'.")
        index = self._index[self._index['study-id'] == str(id)].copy()
        index_errors = self._index_errors[self._index_errors['study-id'] == str(id)].copy()
        ct_from = self._ct_from.study(id) if self._ct_from is not None and self._ct_from.has_study(id) else None
        return DicomStudy(self._dataset, self, id, index, self._index_policy, index_errors, config=self._config, ct_from=ct_from, region_map=self._region_map)

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
    
# Add properties.
props = ['index_policy']
for p in props:
    setattr(DicomPatient, p, property(lambda self, p=p: getattr(self, f'_{DicomPatient.__name__}__{p}')))

# Add properties/methods from 'default_study'.
mods = ['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
for m in mods:
    setattr(DicomPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    setattr(DicomPatient, f'has_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    setattr(DicomPatient, f'{m}_series', lambda self, *args, m=m: self.default_study.series(*args, m) if self.default_study is not None else None)
    setattr(DicomPatient, f'list_{m}_series', lambda self, *args, m=m: self.default_study.list_series(*args, m) if self.default_study is not None else None)
setattr(DicomPatient, 'list_series', lambda self, *args, **kwargs: self.default_study.list_series(*args, **kwargs) if self.default_study is not None else None)

# Add image property shortcuts from 'default_study'.
mods = ['ct', 'mr', 'rtdose']
props = ['data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    n = 'dose' if m == 'rtdose' else m  # Rename 'rtdose' to 'dose'.
    for p in props:
        setattr(DicomPatient, f'{n}_{p}', property(lambda self, m=m, n=n, p=p: getattr(self.default_study, f'{n}_{p}') if self.default_study is not None else None))

# Add landmark/region method shortcuts from 'default_study'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(DicomPatient, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'has_{m[:-1]}')(*args, **kwargs) if self.default_study is not None else False)
    setattr(DicomPatient, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}')(*args, **kwargs) if self.default_study is not None else [])
    setattr(DicomPatient, f'{m}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m}_data')(*args, **kwargs) if self.default_study is not None else None)
 