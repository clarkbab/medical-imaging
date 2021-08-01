import os
import shutil
from typing import *

from .dataset import Dataset, DatasetType, to_type
from .raw import list as list_raw
from .raw import detect_type as detect_raw_type
from .raw.dicom import DICOMDataset, process_dicom
from .raw.nifti import NIFTIDataset, process_nifti
from .processed import ProcessedDataset
from .processed import list as list_processed

def get(
    name: str,
    type_str: Optional[str] = None) -> Dataset:
    """
    returns: the dataset.
    args:
        name: the dataset name.
        type_str: the dataset string. Auto-detected if not present.
    raises:
        ValueError: if the dataset isn't found.
    """
    # Check if 'type' is set.
    if type_str is None:
        # Preference raw datasets.
        raw_ds = list_raw()
        if name in raw_ds:
            # Auto-detect type.
            type = detect_raw_type(name)

            # Create raw dataset.
            if type == DatasetType.DICOM:
                return DICOMDataset(name)
            elif type == DatasetType.NIFTI:
                return NIFTIDataset(name)

        # Check processed datasets secondarily.
        proc_ds = list_processed()
        if name in proc_ds:
            return ProcessedDataset(name)
        else:
            raise ValueError(f"Dataset '{name}' not found.")
    else:
        # Convert from string to type.
        type = to_type(type_str)
    
        # Create dataset.
        if type == DatasetType.DICOM:
            return DICOMDataset(name)
        elif type == DatasetType.NIFTI:
            return NIFTIDataset(name)
        elif type == DatasetType.PROCESSED:
            return ProcessedDataset(name)
        else:
            raise ValueError(f"Dataset '{type.name}: {name}' not found.")

def default() -> Optional[Dataset]:
    """
    returns: the default active dataset.
    """
    # Preference raw datasets.
    raw_ds = list_raw()
    if len(raw_ds) != 0:
        return get(raw_ds[0])
    
    # Check processed datasets secondarily.
    proc_ds = list_processed()
    if len(proc_ds) != 0:
        return get(proc_ds[0])

    return None

ds = default()

def select(
    name: str,
    type_str: Optional[str] = None) -> None:
    """
    effect: sets the dataset as active.
    args:
        name: the dataset name.
        type_str: the dataset string. Auto-detected if not present.
    raises:
        ValueError: if the dataset isn't found.
    """
    global ds
    ds = get(name, type_str)

def active() -> str:
    """
    returns: active dataset name.
    """
    if ds is None:
        return "No active dataset."
    else:
        return ds.description

# DICOMDataset API.

def info(*args, **kwargs):
    return ds.info(*args, **kwargs)

def ct_distribution(*args, **kwargs):
    return ds.ct_distribution(*args, **kwargs)

def ct_summary(*args, **kwargs):
    return ds.ct_summary(*args, **kwargs)

def list_patients(*args, **kwargs):
    return ds.list_patients(*args, **kwargs)
    
def patient(*args, **kwargs):
    return ds.patient(*args, **kwargs)

def region_map(*args, **kwargs):
    return ds.region_map(*args, **kwargs)

def region_names(*args, **kwargs):
    return ds.region_names(*args, **kwargs)

def region_summary(*args, **kwargs):
    return ds.region_summary(*args, **kwargs)

# NIFTIDataset API.

def list_ids(*args, **kwargs):
    return ds.list_ids(*args, **kwargs)

def object(*args, **kwargs):
    return ds.object(*args, **kwargs)

def region_names(*args, **kwargs):
    return ds.region_names(*args, **kwargs)

# ProcessedDataset API.

def manifest(*args, **kwargs):
    return ds.manifest(*args, **kwargs)

def class_frequencies(*args, **kwargs):
    return ds.class_frequencies(*args, **kwargs)

def partition(*args, **kwargs):
    return ds.partition(*args, **kwargs)
