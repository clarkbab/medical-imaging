from enum import Enum

SeriesInstanceUID = str

class DICOMModality(Enum):
    CT = 0
    RTSTRUCT = 1
    RTPLAN = 2
    RTDOSE = 3

# Abstract class.
class DICOMSeries:
    @property
    def modality(self) -> DICOMModality:
        raise NotImplementedError("Child class must implement 'modality'.")
