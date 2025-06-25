from mymi.typing import *

# Abstract class.
class DicomFile:
    @property
    def id(self) -> DicomSOPInstanceUID:
        return NotImplementedError("Child class must implement 'id'.")

    @property
    def filepath(self) -> str:
        raise NotImplementedError("Child class must implement 'filepath'.")
