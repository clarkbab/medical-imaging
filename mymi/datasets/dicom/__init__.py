import os
import shutil
from typing import List

from mymi import config

from .dataset import *
from .index import *
from .series import *
from .study import *
from .utils import *

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'dicom')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> DicomDataset:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    if os.path.exists(ds_path):
        raise FileExistsError(f"Dataset '{name}' already exists at {ds_path}.")
    os.makedirs(ds_path)
    return DicomDataset(name)

def destroy(
    name: str,
    dry_run: bool = True) -> None:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    if os.path.exists(ds_path):
        with_makeitso(dry_run, lambda: shutil.rmtree(ds_path), f"Destroying dicom dataset '{name}' at {ds_path}.")
    
def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    return os.path.exists(ds_path)

def recreate(
    name: str,
    dry_run: bool = True) -> DicomDataset:
    destroy(name, dry_run=dry_run)
    return create(name)
