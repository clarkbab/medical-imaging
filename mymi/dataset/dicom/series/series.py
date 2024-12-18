from enum import Enum

SeriesInstanceUID = str

class Modality(str, Enum):
    CT = 'CT'
    RTSTRUCT = 'RTSTRUCT'
    RTPLAN = 'RTPLAN'
    RTDOSE = 'RTDOSE'

# Abstract class.
class DicomSeries:
    @property
    def modality(self) -> Modality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'DicomStudy':
        raise NotImplementedError("Child class must implement 'study'.")
