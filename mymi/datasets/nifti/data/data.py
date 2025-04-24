from typing import *

from mymi.typing import *

Modality = Literal['CT', 'DOSE', 'LANDMARKS', 'REGIONS']

# Abstract class.
class NiftiData:
    @property
    def id(self) -> SeriesID:
        raise NotImplementedError("Child class must implement 'id'.")

    @property
    def modality(self) -> Modality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'NiftiStudy':
        raise NotImplementedError("Child class must implement 'study'.")
