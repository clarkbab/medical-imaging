import os
import shutil
from typing import List

from mymi import config

from .training_adaptive_dataset import TrainingAdaptiveDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'training-adaptive')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def get(name: str) -> TrainingAdaptiveDataset:
    if exists(name):
        return TrainingAdaptiveDataset(name)
    else:
        raise ValueError(f"TrainingAdaptiveDataset '{name}' doesn't exist.")

def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'training-adaptive', name)
    return os.path.exists(ds_path)

def create(name: str) -> TrainingAdaptiveDataset:
    ds_path = os.path.join(config.directories.datasets, 'training-adaptive', name)
    os.makedirs(ds_path)
    return TrainingAdaptiveDataset(name, check_processed=False)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'training-adaptive', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

def recreate(name: str) -> None:
    destroy(name)
    return create(name)
