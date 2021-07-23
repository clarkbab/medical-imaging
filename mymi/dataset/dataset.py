from enum import Enum

class Dataset:
    @property
    def description(self):
        raise ValueError('Should be overridden')

class DatasetType(Enum):
    DICOM = 0
    NIFTI = 1
    PROCESSED = 2
