import os
import shutil
from typing import *

from mymi import config

from .dataset import Dataset, DatasetType
from .raw import list as list_raw
from .raw import detect_type as detect_raw_type
from .raw.dicom import DICOMDataset
from .raw.nifti import NIFTIDataset
from .processed import ProcessedDataset
from .processed import list as list_processed

def default() -> Optional[Dataset]:
    """
    returns: the default active dataset.
    """
    # Check raw datasets.
    raw_ds = list_raw()
    if len(raw_ds) != 0:
        # Auto-detect type.
        name = raw_ds[0]
        type = detect_raw_type(name)

        # Set dataset.
        if type == DatasetType.DICOM:
            return DICOMDataset(name)
        elif type == DatasetType.NIFTI:
            return NIFTIDataset(name)

    # Check processed datasets.
    processed_ds = list_processed()
    if len(processed_ds) != 0:
        return ProcessedDataset(processed_ds[0])

    return None

ds = default()

def active() -> str:
    """
    returns: active dataset name.
    """
    if ds is None:
        return "No active dataset."
    else:
        return ds.description

def select(
    name: str,
    type: Optional[Union[Literal[DatasetType.DICOM.name, DatasetType.NIFTI.name, DatasetType.PROCESSED.name]]] = None):
    """
    effect: sets the new dataset as active.
    args:
        name: the dataset name.
        type: the dataset type. Auto-detected if not present.
    """
    global ds

    # Check if 'type' is set.
    if type is None:
        # Check raw datasets.
        raw_ds = list_raw()
        if name in raw_ds:
            # Auto-detect type.
            type = detect_raw_type(name)

            # Set dataset.
            if type == DatasetType.DICOM:
                ds = DICOMDataset(name)
            elif type == DatasetType.NIFTI:
                ds = NIFTIDataset(name)

            return

        # Check processed datasets.
        processed_ds = list_processed()
        if name in processed_ds:
            ds = ProcessedDataset(name)
    else:
        if type.lower() == DatasetType.DICOM.name.lower():
            ds = DICOMDataset(name)
        elif type.lower() == DatasetType.NIFTI.name.lower():
            ds = NIFTIDataset(name)
        elif type.lower() == DatasetType.PROCESSED.name.lower():
            ds = ProcessedDataset(name)
        else:
            raise ValueError(f"Dataset type '{type}' not recognised.")

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

def input(*args, **kwargs):
    return ds.input(*args, **kwargs)

def label(*args, **kwargs):
    return ds.label(*args, **kwargs)

def list_samples(*args, **kwargs):
    return ds.list_samples(*args, **kwargs)

def sample(*args, **kwargs):
    return ds.sample(*args, **kwargs)
