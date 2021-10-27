from enum import Enum

class Dataset:
    @property
    def description(self):
        raise ValueError('Should be overridden')

class DatasetType(Enum):
    DICOM = 0
    NIFTI = 1
    TRAINING = 2

def to_type(name: str) -> DatasetType:
    """
    returns: the DatasetType from string.
    args:
        name: the type string.
    """
    if name.lower() == DatasetType.DICOM.name.lower():
        return DatasetType.DICOM
    elif name.lower() == DatasetType.NIFTI.name.lower():
        return DatasetType.NIFTI
    elif name.lower() == DatasetType.TRAINING.name.lower():
        return DatasetType.TRAINING
    else:
        raise ValueError(f"Dataset type '{name}' not recognised.")

def get(
    name: str,
    type: Optional[str] = None) -> Dataset:
    """
    returns: the dataset.
    args:
        name: the dataset name.
        type_str: the dataset string. Auto-detected if not present.
    raises:
        ValueError: if the dataset isn't found.
    """
    if type:
        # Convert from string to type.
        type = to_type(type)
    
        # Create dataset.
        if type == DatasetType.DICOM:
            return DICOMDataset(name)
        elif type == DatasetType.NIFTI:
            return NIFTIDataset(name)
        elif type == DatasetType.TRAINING:
            return TrainingDataset(name)
        else:
            raise ValueError(f"Dataset type '{type}' not found.")
    else:
        # Preference 1: TRAINING.
        proc_ds = list_training()
        if name in proc_ds:
            return TrainingDataset(name)

        # Preference 2: NIFTI.
        nifti_ds = list_nifti()
        if name in nifti_ds:
            return NIFTIDataset(name)

        # Preference 3: DICOM.
        dicom_ds = list_dicom()
        if name in dicom_ds:
            return DICOMDataset(name) 
