import os
import shutil
from typing import List

from mymi import config

from .data import Modality
from .dataset import NrrdDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'nrrd')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> NrrdDataset:
    ds_path = os.path.join(config.directories.datasets, 'nrrd', name)
    os.makedirs(ds_path)
    return NrrdDataset(name)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nrrd', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> NrrdDataset:
    destroy(name)
    return create(name)
