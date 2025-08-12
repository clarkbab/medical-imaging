from typing import List, Optional

from .dataset import Dataset, DatasetType, to_type
from .dicom import DicomDataset
from .dicom import list as list_dicom
from .nifti import NiftiDataset
from .nifti import list as list_nifti
from .raw import RawDataset
from .raw import list as list_raw
from .training import TrainingDataset
from .training import list as list_training

def list(type_str: str) -> List[str]:
    # Convert from string to type.
    type = to_type(type_str)
    
    if type == DatasetType.DICOM:
        return list_dicom()
    elif type == DatasetType.NIFTI:
        return list_nifti()
    elif type == DatasetType.RAW:
        return list_raw()
    elif type == DatasetType.TRAINING:
        return list_training()
    else:
        raise ValueError(f"Dataset type '{type}' not found.")

def get(
    name: str,
    type_str: Optional[str] = None,
    **kwargs) -> Dataset:
    if type_str:
        # Convert from string to type.
        type = to_type(type_str)
    
        # Create dataset.
        if type == DatasetType.DICOM:
            return DicomDataset(name, **kwargs)
        elif type == DatasetType.NIFTI:
            return NiftiDataset(name, **kwargs)
        elif type == DatasetType.RAW:
            return RawDataset(name, **kwargs)
        elif type == DatasetType.TRAINING:
            return TrainingDataset(name, **kwargs)
        else:
            raise ValueError(f"Dataset type '{type}' not found.")
    else:
        # Preference 1: TRAINING.
        proc_ds = list_training()
        if name in proc_ds:
            return TrainingDataset(name, **kwargs)

        # Preference 2: NIFTI.
        nifti_ds = list_nifti()
        if name in nifti_ds:
            return NiftiDataset(name, **kwargs)

        # Preference 3: DICOM.
        dicom_ds = list_dicom()
        if name in dicom_ds:
            return DicomDataset(name, **kwargs)

        # Preference : OTHER.
        raw_ds = list_raw()
        if name in raw_ds:
            return RawDataset(name, **kwargs)

def default() -> Optional[Dataset]:
    """
    returns: the default active dataset.
    """
    # Preference 1: Training.
    proc_ds = list_training()
    if len(proc_ds) != 0:
        return get(proc_ds[0])

    # Preference 2: NIFTI.
    nifti_ds = list_nifti()
    if len(nifti_ds) != 0:
        return get(nifti_ds[0])

    # Preference 3: DICOM.
    dicom_ds = list_dicom()
    if len(dicom_ds) != 0:
        return get(dicom_ds[0])

    # Preference 4: RAW.
    raw_ds = list_raw()
    if len(raw_ds) != 0:
        return get(raw_ds[0])

    return None

ds = None

def select(
    name: str,
    type: Optional[str] = None) -> None:
    global ds
    ds = get(name, type)

def active() -> Optional[str]:
    if ds:
        return ds.description
    else:
        return None

# DicomDataset API.

def list_patients(*args, **kwargs):
    return ds.list_patients(*args, **kwargs)

def list_regions(*args, **kwargs):
    return ds.list_regions(*args, **kwargs)

def info(*args, **kwargs):
    return ds.info(*args, **kwargs)

def ct_distribution(*args, **kwargs):
    return ds.ct_distribution(*args, **kwargs)

def ct_summary(*args, **kwargs):
    return ds.ct_summary(*args, **kwargs)

def patient(*args, **kwargs):
    return ds.patient(*args, **kwargs)

def region_summary(*args, **kwargs):
    return ds.region_summary(*args, **kwargs)

def trimmed_errors(*args, **kwargs):
    return ds.trimmed_errors(*args, **kwargs)

# NIFTIDataset API.

def list_patients(*args, **kwargs):
    return ds.list_patients(*args, **kwargs)

def list_regions(*args, **kwargs):
    return ds.list_regions(*args, **kwargs)

def object(*args, **kwargs):
    return ds.object(*args, **kwargs)

# TrainingDataset API.

def index(*args, **kwargs):
    return ds.index(*args, **kwargs)

def params(*args, **kwargs):
    return ds.params(*args, **kwargs)

def class_frequencies(*args, **kwargs):
    return ds.class_frequencies(*args, **kwargs)

def partition(*args, **kwargs):
    return ds.partition(*args, **kwargs)
