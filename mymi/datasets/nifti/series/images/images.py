from typing import *

from mymi.typing import *

from ..series import NiftiSeries

# Abstract class.
class NiftiImage(NiftiSeries):
    def __str__(self) -> str:
        return self.__global_id
