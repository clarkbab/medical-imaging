import nibabel as nib
import numpy as np
import os
from typing import *

from mymi import config
from mymi.typing import *

from .args import arg_to_list
from .python import delegates

def from_nifti(
    img: nib.nifti1.Nifti1Image,
    ) -> Tuple[Volume, Affine]:
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def to_nifti(
    data: ImageArray,
    affine: Affine,
    ) -> nib.nifti1.Nifti1Image:
    # Convert data types.
    if data.dtype == bool:
        data = data.astype(np.uint32)
    return nib.nifti1.Nifti1Image(data, affine)

def _resolve_filepath(filepath: FilePath) -> FilePath:
    if filepath.startswith('files:'):
        filepath = os.path.join(config.directories.files, filepath[6:])
    return filepath

def save_nifti(
    data: ImageArray,
    affine: Affine,
    filepath: str,
    ) -> None:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    filepath = _resolve_filepath(filepath)
    assert filepath.endswith('.nii.gz') or filepath.endswith('.nii'), "Filepath must end with .nii or .nii.gz"
    img = to_nifti(data, affine)
    dirname = os.path.dirname(filepath)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    nib.save(img, filepath)

def save_numpy(
    data: np.ndarray | torch.Tensor,
    filepath: str,
    ) -> None:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    filepath = _resolve_filepath(filepath)
    assert filepath.endswith('.npy') or filepath.endswith('.npz'), "Filepath must end with .npy or .npz"
    if filepath.endswith('.npz'):
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)
