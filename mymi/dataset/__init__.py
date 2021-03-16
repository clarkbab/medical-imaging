import logging
import sys

from .dicom import HN1 as default

# Create dataset.
active = default

def config(dataset=None):
    """
    effect: configures the dataset module.
    kwargs:
        dataset: the dataset to use.
    """
    global active
    if dataset:
        active = dataset

###
# Basic queries.
###

def has_id(*args, **kwargs):
    return active.has_id(*args, **kwargs)

def list_ct(*args, **kwargs):
    return active.list_ct(*args, **kwargs)

def list_patients():
    return active.list_patients()

def get_rtstruct(*args, **kwargs):
    return active.get_rtstruct(*args, **kwargs)

###
# Raw data.
###

def patient_data(*args, **kwargs):
    return active.patient_data(*args, **kwargs)

def patient_labels(*args, **kwargs):
    return active.patient_labels(*args, **kwargs)

###
# Summaries.
###

def ct(*args, **kwargs):
    return active.ct(*args, **kwargs)

def patient_ct(*args, **kwargs):
    return active.patient_ct(*args, **kwargs)
    
def regions(*args, **kwargs):
    return active.regions(*args, **kwargs)

def patient_regions(*args, **kwargs):
    return active.patient_regions(*args, **kwargs)

def region_count(*args, **kwargs):
    return active.region_count(*args, **kwargs)

def summary(*args, **kwargs):
    return active.summary(*args, **kwargs)

def patient_summary(*args, **kwargs):
    return active.patient_summary(*args, **kwargs)
