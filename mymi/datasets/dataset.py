from enum import Enum

class Dataset:
    def __str__(self) -> str:
        return self.global_id

class DatasetType(Enum):
    DICOM = 0
    NIFTI = 1
    NRRD = 2
    TRAINING = 3
    RAW = 6

def to_type(name: str) -> DatasetType:
    if name.lower() == DatasetType.DICOM.name.lower():
        return DatasetType.DICOM
    elif name.lower() == DatasetType.NIFTI.name.lower():
        return DatasetType.NIFTI
    elif name.lower() == DatasetType.NRRD.name.lower():
        return DatasetType.NRRD
    elif name.lower() == DatasetType.TRAINING.name.lower():
        return DatasetType.TRAINING
    elif name.lower() == DatasetType.RAW.name.lower():
        return DatasetType.RAW
    else:
        raise ValueError(f"Dataset type '{name}' not recognised.")
