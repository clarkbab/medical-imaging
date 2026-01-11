from enum import Enum
from typing import *

from mymi.typing import *
from mymi.utils import load_yaml

from .region_map import * 

CT_FROM_REGEXP = r'^__CT_FROM_(.*)__$'

class Dataset:
    def __init__(
        self,
        id: DatasetID,
        ct_from: Optional['Dataset'] = None) -> None:
        self._id = str(id)
        self._ct_from = ct_from
        filepath = os.path.join(self._path, 'config.yaml')
        self._config = load_yaml(filepath) if os.path.exists(filepath) else {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_index'):
                self._load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def groups(self) -> pd.DataFrame:
        return self._groups

    @property
    def id(self) -> DatasetID:
        return self._id

    @ensure_loaded
    def list_groups(self) -> List[PatientGroup]:
        if self._groups is None:
            raise ValueError(f"File 'groups.csv' not found for dicom dataset '{self._id}'.")
        group_ids = list(sorted(self._groups['group-id'].unique()))
        return group_ids

    @property
    def path(self) -> DirPath:
        return self._path

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            id=self._id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"

class DatasetType(Enum):
    DICOM = 0
    NIFTI = 1
    TRAINING = 3
    RAW = 6

def to_type(name: str) -> DatasetType:
    if name.lower() == DatasetType.DICOM.name.lower():
        return DatasetType.DICOM
    elif name.lower() == DatasetType.NIFTI.name.lower():
        return DatasetType.NIFTI
    elif name.lower() == DatasetType.TRAINING.name.lower():
        return DatasetType.TRAINING
    elif name.lower() == DatasetType.RAW.name.lower():
        return DatasetType.RAW
    else:
        raise ValueError(f"Dataset type '{name}' not recognised.")
