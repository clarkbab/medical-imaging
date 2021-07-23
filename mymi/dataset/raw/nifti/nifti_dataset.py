from ...dataset import Dataset, DatasetType

class NIFTIDataset(Dataset):
    def __init__(
        self,
        name: str):
        self._name = name
    
    @property
    def description(self) -> str:
        return f"NIFTI: {self._name}"

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI
