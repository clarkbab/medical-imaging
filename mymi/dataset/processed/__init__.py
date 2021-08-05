import os
import shutil
from typing import List

from mymi import config

from .processed_dataset import ProcessedDataset
from .processed_partition import ProcessedPartition

def list() -> List[str]:
    """
    returns: list of raw datasets.
    """
    path = os.path.join(config.directories.datasets, 'processed')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> ProcessedDataset:
    """
    effect: creates a dataset.
    args:
        name: the name of the dataset.
    """
    # Create root folder.
    ds_path = os.path.join(config.directories.datasets, 'processed', name)
    os.makedirs(ds_path)

    return ProcessedDataset(name)

def destroy(name: str) -> None:
    """
    effect: destroys a dataset.
    args:
        name: the name of the dataset.
    """
    ds_path = os.path.join(config.directories.datasets, 'processed', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    """
    effect: destroys and creates a dataset.
    args:
        name: the name of the dataset.
    """
    destroy(name)
    return create(name)
