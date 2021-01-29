import logging
from .dataset import Dataset

# Create dataset.
dataset = None

def config(**kwargs):
    """
    effect: configures the dataset.
    kwargs:
        dataset: the dataset to read from.
    """
    global dataset
    dataset = kwargs.pop('dataset', None)
    if dataset is None:
        from mymi.dataset.datasets import HN1  # Importing sets HN1 as default.

###
# Basic queries.
###

def has_id(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.has_id(*args, **kwargs)

def list_ct(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.list_ct(*args, **kwargs)

def list_patients():
    if dataset is None:
        config()

    return dataset.list_patients()

def get_rtstruct(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.get_rtstruct(*args, **kwargs)

###
# Raw data.
###

def patient_data(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_data(*args, **kwargs)

def patient_labels(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_labels(*args, **kwargs)

###
# Summaries.
###

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

def region_count(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.region_count(*args, **kwargs)

def summary(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.summary(*args, **kwargs)

def patient_summary(*args, **kwargs):
    if dataset is None:
        config()

    return dataset.patient_summary(*args, **kwargs)
