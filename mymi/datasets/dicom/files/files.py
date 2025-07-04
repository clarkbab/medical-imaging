from mymi.typing import *

# Abstract class.
class DicomFile:
    def __str__(self) -> str:
        return self.global_id
