from typing import *

from mymi.typing import *

# Abstract class.
class NrrdData:
    @property
    def id(self) -> SeriesID:
        raise NotImplementedError("Child class must implement 'id'.")

    @property
    def modality(self) -> NrrdModality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'NrrdStudy':
        raise NotImplementedError("Child class must implement 'study'.")
