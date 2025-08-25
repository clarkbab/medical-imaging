import os
import shutil
from typing import List

from mymi import config

from .dataset import *
from .series import *
from .utils import *

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

def destroy(
    name: str,
    dry_run: bool) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    if os.path.exists(ds_path):
        with_dry_run(dry_run, lambda: shutil.rmtree(ds_path), f"Destroying nifti dataset '{name}' at {ds_path}.")
    
def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'nifti', name)
    return os.path.exists(ds_path)

def recreate(
    name: str,
    dry_run: bool) -> NiftiDataset:
    destroy(name, dry_run=dry_run)
    return create(name)
