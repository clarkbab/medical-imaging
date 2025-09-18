from typing import *

from ...series import Series

class NiftiSeries(Series):
    def __init__(
        self,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def date(self) -> Optional[str]:
        # May implement in dicom -> nifti processing in future.
        return None
