import logging
from .dataset import Dataset

# Create dataset.
dataset = None

def config(**kwargs):
    """
    dataset: the dataset to read from.
    """
    global dataset
    dataset = kwargs.pop('dataset', None)
    if dataset is None:
        from mymi.dataset.datasets import HN1  # Importing sets HN1 as default.

def has_id(pat_id):
    if dataset is None:
        config()

    return dataset.has_id(pat_id)

def list_ct(pat_id):
    if dataset is None:
        config()

    return dataset.list_ct(pat_id)

def list_patients():
    if dataset is None:
        config()

    return dataset.list_patients()

def get_rtstruct(pat_id):
    if dataset is None:
        config()

    return dataset.get_rtstruct(pat_id)

def patient_info(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_info(*args, **kwargs)

def ct(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.ct(*args, **kwargs)

def patient_ct(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_ct(*args, **kwargs)
    
def regions(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.regions(*args, **kwargs)

def patient_regions(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_regions(*args, **kwargs)

def summary(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.summary(*args, **kwargs)

def patient_summary(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_summary(*args, **kwargs)

def region_count(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.region_count(*args, **kwargs)