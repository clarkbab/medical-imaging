from typing import *

from mymi.typing import *

Modality = Literal['CT', 'MR', 'RTSTRUCT', 'RTPLAN', 'RTDOSE']

# Abstract class.
class DicomSeries:
    @property
    def id(self) -> SeriesID:
        raise NotImplementedError("Child class must implement 'id'.")

    @property
    def modality(self) -> Modality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'DicomStudy':
        raise NotImplementedError("Child class must implement 'study'.")
 