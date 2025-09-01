import ast
import nibabel as nib
import os
import pandas as pd
import json
from typing import *
import yaml

from mymi import config
from mymi.typing import *

from .args import arg_to_list
from .nifti import from_nifti
from .python import delegates
from .sitk import from_sitk_image, to_sitk_image

def handle_filepath(f: FilePath) -> FilePath:
    if f.startswith('files:'):
        f = os.path.join(config.directories.files, f[6:])
    return f

def load_csv(
    filepath: str,
    exists_only: bool = False,
    filters: Dict[str, Any] = {},
    map_cols: Dict[str, str] = {},
    map_types: Dict[str, Any] = {},
    parse_cols: Union[str, List[str]] = [],
    **kwargs: Dict[str, str]) -> Optional[pd.DataFrame]:
    if filepath.startswith('files:'):
        filepath = os.path.join(config.directories.files, filepath[6:])
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"CSV at filepath '{filepath}' not found.")

    # Load CSV.
    map_types['patient-id'] = str
    map_types['study-id'] = str
    df = pd.read_csv(filepath, dtype=map_types, **kwargs)

    # Map column names.
    df = df.rename(columns=map_cols)

    # Evaluate columns as literals.
    parse_cols = arg_to_list(parse_cols, str)
    for c in parse_cols:
        df[c] = df[c].apply(lambda s: ast.literal_eval(s))

    # Apply filters.
    for k, v in filters.items():
        df = df[df[k] == v]

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

@delegates(from_nifti)
def load_nifti(
    filepath: str,
    **kwargs) -> Tuple[ImageArray, Spacing3D, Point3D]:
    assert filepath.endswith('.nii') or filepath.endswith('.nii.gz'), "Filepath must end with .nii or .nii.gz"
    img = nib.load(filepath)
    return from_nifti(img, **kwargs)

def load_numpy(
    filepath: str,
    keys: Union[str, List[str]] = 'data') -> Union[ImageArray, List[ImageArray]]:
    assert filepath.endswith('.npz'), "Filepath must end with .npz"
    keys = arg_to_list(keys, str)
    data = np.load(filepath)
    items = [data[k] for k in keys]
    items = items[0] if len(items) == 1 else items
    return items

def load_yaml(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_csv(
    data: pd.DataFrame,
    filepath: str,
    index: bool = False,
    overwrite: bool = True) -> None:
    filepath = handle_filepath(filepath)
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data.to_csv(filepath, index=index)
        else:
            raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=index)

def save_json(
    data: Any,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    assert filepath.split('.')[-1] == 'json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def save_text(
    data: str,
    filepath: FilePath,
    overwrite: bool = True) -> None:
    if filepath.startswith('files:'):
        filepath = os.path.join(config.directories.files, filepath[6:])
    if os.path.exists(filepath) and not overwrite:
        raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(data)

def save_yaml(
    data: Any,
    filepath: str) -> None:
    with open(filepath, 'w') as f:
        yaml.dump(data, f)

def sitk_load_image(filepath: FilePath) -> Tuple[ImageArray, Spacing3D, Point3D]:
    if filepath.endswith('.mha'):
        img_type = 'mha'
    elif filepath.endswith('.nii.gz') or filepath.endswith('.nii'):
        img_type = 'nii'
    else:
        raise ValueError(f'Unsupported file type: {filepath}.')
    img = sitk.ReadImage(filepath)
    return from_sitk_image(img, img_type=img_type)

def sitk_save_image(
    data: ImageArray,
    spacing: Spacing3D,
    offset: Point3D,
    filepath: FilePath) -> None:
    img = to_sitk_image(data, spacing, offset)
    sitk.WriteImage(img, filepath)
