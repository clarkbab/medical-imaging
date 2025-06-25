import shutil
from typing import *

from .dataset import *

def list() -> List[str]:
    path = os.path.join(config.directories.datasets, 'raw')
    if os.path.exists(path):
        return sorted(os.listdir(path))
    else:
        return []

def get(name: str) -> RawDataset:
    if exists(name):
        return RawDataset(name)
    else:
        raise ValueError(f"RawDataset '{name}' doesn't exist.")

def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'raw', name)
    return os.path.exists(ds_path)

def create(name: str) -> RawDataset:
    ds_path = os.path.join(config.directories.datasets, 'raw', name)
    os.makedirs(ds_path)
    return RawDataset(name, check_processed=False)

def destroy(name: str) -> None:
    ds_path = os.path.join(config.directories.datasets, 'raw', name)
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)
    
def exists(name: str) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'raw', name)
    return os.path.exists(ds_path)

def recreate(name: str) -> None:
    destroy(name)
    return create(name)
