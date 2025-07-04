from datetime import datetime
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from .study import DicomStudy
from .files import RegionMap
from .series import *

class DicomPatient:
    def __init__(
        self,
        dataset: 'DicomDataset',
        id: PatientID,
        ct_from: Optional['DicomPatient'] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self.__id = str(id)
        self.__global_id = f'{dataset}:{id}'
        self.__ct_from = ct_from
        self.__dataset = dataset
        self.__region_map = region_map

        # Get patient index.
        index = self.__dataset.index
        index = index[index['patient-id'] == str(id)].copy()
        if len(index) == 0:
            raise ValueError(f"Patient '{id}' not found in dataset '{dataset}'.")
        self.__index = index

        # Get patient error index.
        error_index = self.__dataset.error_index
        error_index = error_index[error_index['patient-id'] == str(id)].copy()
        self.__error_index = error_index

        # Get policies.
        self.__index_policy = self.__dataset.index_policy
        self.__region_policy = self.__dataset.region_policy

        # Check that patient ID exists.
        if len(index) == 0:
            raise ValueError(f"Patient '{self}' not found in index for dataset '{dataset}'.")

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

    def list_studies(
        self,
        study_ids: StudyIDs = 'all') -> List[StudyID]:
        # Sort studies by date/time - oldest first.
        ids = list(self.__index.sort_values(['study-date', 'study-time'])['study-id'].unique())
        
        # Filter by 'study_ids'.
        if study_ids != 'all':
            ids = [i for i in ids if i in study_ids]

        return ids

    def study(
        self,
        id: StudyID) -> DicomStudy:
        if id.startswith('idx:'):
            idx = int(id.split(':')[1])
            study_ids = self.list_studies()
            if idx > len(study_ids) - 1:
                raise ValueError(f"Index {idx} out of range for patient with {len(study_ids)} studies.")
            id = study_ids[idx]
        return DicomStudy(self, id, region_map=self.__region_map)

    def __str__(self) -> str:
        return self.__global_id
    
# Add properties.
props = ['ct_from', 'dataset', 'error_index', 'global_id', 'id', 'index', 'index_policy', 'path', 'region_map', 'region_policy']
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
        setattr(DicomPatient, f'{n}_{p}', property(lambda self, m=m, n=n, p=p: getattr(self.default_study, f'{n}_{p}') if default_study is not None else None))

# Add landmark/region method shortcuts from 'default_study'.
mods = ['landmark', 'region']
for m in mods:
    setattr(DicomPatient, f'has_{m}_ids', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'has_{m}_ids')(*args, **kwargs) if self.default_study is not None else False)
    setattr(DicomPatient, f'list_{m}_ids', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}_ids')(*args, **kwargs) if self.default_study is not None else [])
    setattr(DicomPatient, f'{m}s_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m}s_data')(*args, **kwargs) if self.default_study is not None else None)
 