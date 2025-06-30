import os
import shutil
from typing import List

from mymi import config

from .images import *
from .dataset import *

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'nifti')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    os.makedirs(ds_path)
    return NiftiDataset(name)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)
    
def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    return os.path.exists(ds_path)

def recreate(name: str) -> NiftiDataset:
    destroy(name)
    return create(name)
