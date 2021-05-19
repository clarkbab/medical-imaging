import logging
import os
import pandas as pd
import sys
from typing import *

from mymi import config

from .dicom import DicomDataset
from .processed import ProcessedDataset
from .types import Types

def list_datasets() -> Sequence[str]:
    """
    returns: list of available datasets.
    """
    return list(sorted(os.listdir(config.directories.datasets)))

# Make first dataset active.
sets = list_datasets()
ds = DicomDataset(sets[0])

def active() -> str:
    """
    returns: active dataset name.
    """
    return ds.name

def select(
    name: str,
    ct_from: str = None,
    type: int = Types.DICOM):
    """
    effect: sets the new dataset as active.
    args:
        ct_from: get CT data from other dataset.
        name: the name of the new dataset.
        type: the type of dataset.
    """
    # Set current dataset.
    global ds
    if type == Types.DICOM:
        ds = DicomDataset(name, ct_from=ct_from)
    elif type == Types.PROCESSED:
        ds = ProcessedDataset(name)

def info(*args, **kwargs):
    return ds.info(*args, **kwargs)

def ct_distribution(*args, **kwargs):
    return ds.ct_distribution(*args, **kwargs)

def ct_summary(*args, **kwargs):
    return ds.ct_summary(*args, **kwargs)

def label_map(*args, **kwargs):
    return ds.label_map(*args, **kwargs)

def label_names(*args, **kwargs):
    return ds.label_names(*args, **kwargs)

def label_summary(*args, **kwargs):
    return ds.label_summary(*args, **kwargs)

def list_patients(*args, **kwargs):
    return ds.list_patients(*args, **kwargs)
    
def patient(*args, **kwargs):
    return ds.patient(*args, **kwargs)

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
