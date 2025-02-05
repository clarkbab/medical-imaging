SOPInstanceUID = str

# Abstract class.
class DicomFile:
    @property
    def path(self) -> str:
        raise NotImplementedError("Child class must implement 'path'.")
