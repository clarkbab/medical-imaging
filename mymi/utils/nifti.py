import nibabel as nib
import os
from typing import Tuple

from mymi.types import *

def from_nifti(img: nib.nifti1.Nifti1Image) -> Tuple[np.ndarray, ImageSpacing3D, PointMM3D]:
    data = img.get_fdata()
    affine = img.get_affine()
    spacing = (affine[0][0], affine[1][1], affine[2][2])
    offset = (affine[0][3], affine[1][3], affine[2][3])
    return data, spacing, offset

def load_nifti(filepath: str) -> Tuple[np.ndarray, ImageSpacing3D, PointMM3D]:
    img = nib.load(filepath)
    return from_nifti(img)

def to_nifti(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D) -> nib.nifti1.Nifti1Image:
    # Convert data types.
    if data.dtype == bool:
        data = data.astype(np.uint32)

    # Create coordinate transform.
    affine = np.array([
        [spacing[0], 0, 0, offset[0]],
        [0, spacing[1], 0, offset[1]],
        [0, 0, spacing[2], offset[2]],
        [0, 0, 0, 1]])
    
    return nib.nifti1.Nifti1Image(data, affine)

def save_as_nifti(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D,
    filepath: str) -> None:
    img = to_nifti(data, spacing, offset)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nib.save(img, filepath)
