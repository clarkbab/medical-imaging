from enum import Enum

class DICOMModality(Enum):
    CT = 0
    RTSTRUCT = 1

class DICOMSeries:
    @property
    def modality(self) -> DICOMModality:
        raise ValueError('Child class must implement.')
