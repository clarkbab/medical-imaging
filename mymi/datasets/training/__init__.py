import os
import shutil
from typing import List

from mymi import config
from mymi.utils import with_makeitso

from .dataset import TrainingDataset

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'training')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def get(name: str) -> TrainingDataset:
    if exists(name):
        return TrainingDataset(name)
    else:
        raise ValueError(f"TrainingDataset '{name}' doesn't exist.")

def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    return os.path.exists(ds_path)

def create(name: str) -> TrainingDataset:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    os.makedirs(ds_path)
    return TrainingDataset(name, check_processed=False)

def destroy(
    name: str,
    makeitso: bool = False) -> None:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    if os.path.exists(ds_path):
        with_makeitso(
            makeitso,
            lambda: shutil.rmtree(ds_path),
            f"Destroying training dataset '{name}' at {ds_path}."
        )
    
def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'training', name)
    return os.path.exists(ds_path)

def load(name: str) -> TrainingDataset:
    if exists(name):
        return TrainingDataset(name)
    else:
        raise FileNotFoundError(f"Training dataset '{name}' does not exist.")

def recreate(
    name: str,
    makeitso: bool = False,
    ) -> TrainingDataset:
    destroy(name, makeitso=makeitso)
    if not makeitso:
        if exists(name):
            return TrainingDataset(name)
        else:
            # Creating is fine with makeitso=False.
            create(name)
    else:
        return create(name)
