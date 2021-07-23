import os
import shutil
from typing import *

from mymi import config

from .dicom import DicomDataset
from .processed import ProcessedDataset
from .types import types

def list() -> Tuple[str]:
    """
    returns: list of available datasets.
    """
    return tuple(sorted(os.listdir(config.directories.datasets)))

def create(name: str) -> None:
    """
    effect: creates a dataset.
    args:
        name: the name of the dataset.
    """
    # Create dataset folder.
    ds_path = os.path.join(config.directories.datasets, name)
    os.makedirs(ds_path)

def destroy(name: str) -> None:
    """
    effect: destroys a dataset.
    args:
        name: the name of the dataset.
    """
    ds_path = os.path.join(config.directories.datasets, name)
    shutil.rmtree(ds_path)

# Make first dataset active.
sets = list()
ds = DicomDataset(sets[0])

def active() -> str:
    """
    returns: active dataset name.
    """
    return ds.description()

def select(
    name: str,
    type: int = types.DICOM):
    """
    effect: sets the new dataset as active.
    args:
        name: the name of the new dataset.
        type: the type of dataset.
    """
    # Set current dataset.
    global ds
    if type == types.DICOM:
        ds = DicomDataset(name)
    elif type == types.PROCESSED:
        ds = ProcessedDataset(name)
    else:
        type_names = [t.name for t in types]
        type_string = ', '.join(type_names)
        raise ValueError(f"Invalid dataset type '{type}', expected one of 'dataset.types.{{{type_string}}}'.")

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

##
# Processed dataset API.
##

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
