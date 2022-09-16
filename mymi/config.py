from collections import namedtuple
import os
import pandas as pd
from typing import Dict, List, Optional

from mymi import logging

class Directories:
    @property
    def cache(self):
        return os.path.join(self.root, 'cache')

    @property
    def models(self):
        return os.path.join(self.root, 'models')

    @property
    def datasets(self):
        return os.path.join(self.root, 'datasets')

    @property
    def files(self):
        return os.path.join(self.root, 'files')
    
    @property
    def evaluations(self):
        return os.path.join(self.root, 'evaluations')

    @property
    def root(self):
        return os.environ['MYMI_DATA']

    @property
    def runs(self):
        return os.path.join(self.root, 'runs')

    @property
    def temp(self):
        return os.path.join(self.root, 'tmp')

    @property
    def tensorboard(self):
        return os.path.join(self.root, 'reports', 'tensorboard')

    @property
    def wandb(self):
        return os.path.join(self.root, 'reports')

class Formatting:
    @property
    def metrics(self):
        return '.6f'

    @property
    def sample_digits(self):
        return 5

directories = Directories()
formatting = Formatting()

def environ(name: str) -> Optional[str]:
    if name in os.environ:
        return os.environ[name]
    else:
        return None

def save_csv(
    data: pd.DataFrame,
    *path: List[str],
    index: bool = False,
    overwrite: bool = False) -> None:
    filepath = os.path.join(directories.files, *path)
    dirpath = os.path.dirname(filepath)
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(dirpath, exist_ok=True)
            data.to_csv(filepath, index=index)
        else:
            logging.error(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(dirpath, exist_ok=True)
        data.to_csv(filepath, index=index)

def load_csv(
    *path: List[str],
    raise_error: bool = True,
    **kwargs: Dict[str, str]) -> Optional[pd.DataFrame]:
    filepath = os.path.join(directories.files, *path)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, **kwargs)
    elif raise_error:
        raise ValueError(f"CSV at path '{path}' not found.")
    else:
        return None
