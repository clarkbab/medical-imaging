import os
import shutil
from typing import List

from mymi import config

from .dataset import *
from .index import *
from .series import *
from .utils import *

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'dicom')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def create(name: str) -> DicomDataset:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    os.makedirs(ds_path)
    return DicomDataset(name)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)
    
def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'dicom', name)
    return os.path.exists(ds_path)

def recreate(name: str) -> DicomDataset:
    destroy(name)
    return create(name)
