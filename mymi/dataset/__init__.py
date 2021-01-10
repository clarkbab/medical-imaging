import logging

# Create dataset.
dataset = None

def config(**kwargs):
    """
    dataset: the dataset to read from.
    """
    global dataset
    dataset = kwargs.pop('dataset', None)
    if dataset is None:
        from mymi.dataset.dicom import HN1  # Importing sets HN1 as default.

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
