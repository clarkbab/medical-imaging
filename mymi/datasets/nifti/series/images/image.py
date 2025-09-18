from typing import *

from mymi.typing import *

from ..series import NiftiSeries

# Abstract class.
class NiftiImageSeries(NiftiSeries):
    def __init__(
        self,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
