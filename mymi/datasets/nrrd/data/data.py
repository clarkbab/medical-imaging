from enum import Enum

from mymi.typing import SeriesID

class Modality(str, Enum):
    CT = 'CT'
    DOSE = 'DOSE'
    LANDMARKS = 'LANDMARKS'
    REGIONS = 'REGIONS'

# Abstract class.
class NrrdData:
    @property
    def id(self) -> SeriesID:
        raise NotImplementedError("Child class must implement 'id'.")

    @property
    def modality(self) -> Modality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'NrrdStudy':
        raise NotImplementedError("Child class must implement 'study'.")
