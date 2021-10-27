from typing import Optional

from .dataset import Dataset, DatasetType, to_type
from .dicom import DICOMDataset
from .dicom import list as list_dicom
from .nifti import NIFTIDataset
from .nifti import list as list_nifti
from .processed import ProcessedDataset
from .processed import list as list_processed

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
        elif type == DatasetType.PROCESSED:
            return ProcessedDataset(name)
        else:
            raise ValueError(f"Dataset type '{type}' not found.")
    else:
        # Preference 1: Processed.
        proc_ds = list_processed()
        if name in proc_ds:
            return ProcessedDataset(name)

        # Preference 2: NIFTI.
        nifti_ds = list_nifti()
        if name in nifti_ds:
            return NIFTIDataset(name)

        # Preference 3: DICOM.
        dicom_ds = list_dicom()
        if name in dicom_ds:
            return DICOMDataset(name) 

def default() -> Optional[Dataset]:
    """
    returns: the default active dataset.
    """
    # Preference 1: Processed.
    proc_ds = list_processed()
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

    return None

ds = default()

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

# DICOMDataset API.

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

def trimmed_summary(*args, **kwargs):
    return ds.trimmed_summary(*args, **kwargs)

# NIFTIDataset API.

def list_patients(*args, **kwargs):
    return ds.list_patients(*args, **kwargs)

def list_regions(*args, **kwargs):
    return ds.list_regions(*args, **kwargs)

def object(*args, **kwargs):
    return ds.object(*args, **kwargs)

# ProcessedDataset API.

def manifest(*args, **kwargs):
    return ds.manifest(*args, **kwargs)

def params(*args, **kwargs):
    return ds.params(*args, **kwargs)

def class_frequencies(*args, **kwargs):
    return ds.class_frequencies(*args, **kwargs)

def partition(*args, **kwargs):
    return ds.partition(*args, **kwargs)
