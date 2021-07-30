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
    shutil.rmtree(ds_path)
