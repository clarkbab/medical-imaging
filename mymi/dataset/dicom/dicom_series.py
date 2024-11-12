from enum import Enum

SeriesInstanceUID = str

class Modality(str, Enum):
    CT = 'CT'
    RTSTRUCT = 'RTSTRUCT'
    RTPLAN = 'RTPLAN'
    RTDOSE = 'RTDOSE'

# Abstract class.
class DICOMSeries:
    @property
    def modality(self) -> Modality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'DICOMStudy':
        raise NotImplementedError("Child class must implement 'study'.")
