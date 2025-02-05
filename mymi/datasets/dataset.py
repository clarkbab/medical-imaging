from enum import Enum

class Dataset:
    @property
    def description(self):
        raise ValueError('Should be overridden')

class DatasetType(Enum):
    DICOM = 0
    NIFTI = 1
    NRRD = 2
    TRAINING = 3
    TRAINING_ADAPTIVE = 4
    OTHER = 5

def to_type(name: str) -> DatasetType:
    if name.lower() == DatasetType.DICOM.name.lower():
        return DatasetType.DICOM
    elif name.lower() == DatasetType.NIFTI.name.lower():
        return DatasetType.NIFTI
    elif name.lower() == DatasetType.NRRD.name.lower():
        return DatasetType.NRRD
    elif name.lower() == DatasetType.TRAINING.name.lower():
        return DatasetType.TRAINING
    elif name.lower() == 'training-adaptive':
        return DatasetType.TRAINING_ADAPTIVE
    elif name.lower() == DatasetType.OTHER.name.lower():
        return DatasetType.OTHER
    else:
        raise ValueError(f"Dataset type '{name}' not recognised.")
