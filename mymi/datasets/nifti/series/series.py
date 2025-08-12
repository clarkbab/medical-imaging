from typing import *

class NiftiSeries:
    @property
    def date(self) -> Optional[str]:
        # May implement in dicom -> nifti processing in future.
        return None

    @property
    def global_id(self) -> str:
        return self._global_id

    def __repr(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self._global_id
