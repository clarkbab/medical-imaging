from enum import Enum

from .region_map import * 

CT_FROM_REGEXP = r'^__CT_FROM_(.*)__$'

class Dataset:
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.global_id

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
