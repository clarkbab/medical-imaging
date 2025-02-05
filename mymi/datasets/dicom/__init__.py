import os
import shutil
from typing import List

from mymi import config

from .index import build_index
from .dataset import DicomDataset
from .dicom import DATE_FORMAT, TIME_FORMAT
from .files import ROIData, RtstructConverter
from .series import Modality

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

def recreate(name: str) -> DicomDataset:
    destroy(name)
    return create(name)
