from enum import Enum

from mymi.typing import *

from .region_map import * 

CT_FROM_REGEXP = r'^__CT_FROM_(.*)__$'

class Dataset:
    def __init__(
        self,
        id: DatasetID,
        ct_from: Optional['Dataset'] = None) -> None:
        self._id = str(id)
        self._ct_from = ct_from

    @property
    def id(self) -> DatasetID:
        return self._id

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        raise ValueError("Subclasses of 'Dataset' must implement '__str__' method.")

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
