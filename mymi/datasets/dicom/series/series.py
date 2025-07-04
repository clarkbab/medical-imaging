from typing import *

from mymi.typing import *

# Abstract class.
class DicomSeries:
    def __str__(self) -> str:
        return self.global_id
 