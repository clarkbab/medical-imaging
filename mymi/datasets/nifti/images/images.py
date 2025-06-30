from typing import *

from mymi.typing import *

# Abstract class.
class NiftiImage:
    @property
    def id(self) -> SeriesID:
        raise NotImplementedError("Child class must implement 'id'.")

    @property
    def modality(self) -> NiftiModality:
        raise NotImplementedError("Child class must implement 'modality'.")

    @property
    def study(self) -> 'NiftiStudy':
        raise NotImplementedError("Child class must implement 'study'.")
