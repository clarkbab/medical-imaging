import os
import shutil
from typing import Optional, Tuple

from mymi import config

from ..dataset import DatasetType, to_type

def list() -> Tuple[str]:
    """
    returns: list of raw datasets.
    """
    return tuple(sorted(os.listdir(os.path.join(config.directories.datasets, 'raw'))))

def create(
    name: str,
    type_str: Optional[str] = 'dicom') -> None:
    """
    effect: creates a dataset.
    args:
        name: the name of the dataset.
    kwargs:
        type_str: the type of the dataset.
    """
    type = to_type(type_str)
    ds_path = os.path.join(config.directories.datasets, 'raw', name)
    os.makedirs(ds_path)
    if type == DatasetType.DICOM:
        return DICOMDataset(name)
    elif type == DatasetType.NIFTI:
        return NIFTIDataset(name)

def destroy(name: str) -> None:
    """
    effect: destroys a dataset.
    args:
        name: the name of the dataset.
    """
    ds_path = os.path.join(config.directories.datasets, 'raw', name)
    shutil.rmtree(ds_path)

def recreate(
    name: str,
    type_str: Optional[str] = 'dicom') -> None:
    """
    effect: destroys and creates a dataset.
    args:
        name: the name of the dataset.
    kwargs:
        type_str: the type of the dataset.
    """
    destroy(name)
    return create(name, type_str=type_str)

def detect_type(name: str) -> DatasetType:
    """
    returns: the auto-detected type of the raw dataset.
    args:
        name: the dataset name.
    raises:
        ValueError: the dataset type couldn't be detected.
    """
    dataset_path = os.path.join(config.directories.datasets, 'raw', name)
    for _, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith('.dcm'):
                return DatasetType.DICOM
            elif f.lower().endswith('.nii.gz'):
                return DatasetType.NIFTI

    raise ValueError(f"Type couldn't be detected for dataset '{name}'.")
