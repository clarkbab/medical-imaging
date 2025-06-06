import os
import pandas as pd
import json
from typing import *
import yaml

from mymi import config

def load_csv(
    filepath: str,
    exists_only: bool = False,
    map_cols: Dict[str, str] = {},
    map_types: Dict[str, Any] = {},
    **kwargs: Dict[str, str]) -> Optional[pd.DataFrame]:
    if not os.path.exists(filepath):
        if exists_only:
            return False
        else:
            raise ValueError(f"CSV at filepath '{filepath}' not found.")
    elif exists_only:
        return True

    # Load CSV.
    map_types['patient-id'] = str
    map_types['study-id'] = str
    df = pd.read_csv(filepath, dtype=map_types, **kwargs)

    # Map column names.
    df = df.rename(columns=map_cols)

    return df

def load_files_csv(
    *path: List[str],
    exists_only: bool = False,
    map_cols: Dict[str, str] = {},
    map_types: Dict[str, Any] = {},
    **kwargs: Dict[str, str]) -> Optional[pd.DataFrame]:
    filepath = os.path.join(config.directories.files, *path)
    if not os.path.exists(filepath):
        if exists_only:
            return False
        else:
            raise ValueError(f"CSV at filepath '{filepath}' not found.")
    elif exists_only:
        return True

    # Load CSV.
    map_types['patient-id'] = str
    map_types['study-id'] = str
    df = pd.read_csv(filepath, dtype=map_types, **kwargs)

    # Map column names.
    df = df.rename(columns=map_cols)

    return df

def load_json(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return json.load(f)

def load_yaml(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_csv(
    data: pd.DataFrame,
    filepath: str,
    index: bool = False,
    overwrite: bool = True) -> None:
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data.to_csv(filepath, index=index)
        else:
            raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=index)

def save_files_csv(
    data: pd.DataFrame,
    *path: List[str],
    index: bool = False,
    header: bool = True,
    overwrite: bool = True) -> None:
    filepath = os.path.join(config.directories.files, *path)
    assert filepath.split('.')[-1] == 'csv'
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data.to_csv(filepath, header=header, index=index)
        else:
            raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, header=header, index=index)

def save_json(
    data: Any,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    assert filepath.split('.')[-1] == 'json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def save_yaml(
    data: Any,
    filepath: str) -> None:
    with open(filepath, 'w') as f:
        yaml.dump(data, f)
