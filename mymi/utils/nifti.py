import nibabel as nib
import os
from typing import *

from mymi.typing import *

from .python import delegates
from .utils import arg_to_list

def from_nifti(
    img: nib.nifti1.Nifti1Image,
    coords: Literal['lps', 'ras'] = 'lps') -> Tuple[Image, ImageSpacing3D, Point3D]:
    data = img.get_fdata()
    affine = img.affine
    spacing = (affine[0][0], affine[1][1], affine[2][2])
    offset = (affine[0][3], affine[1][3], affine[2][3])
    if coords == 'ras':
        # Our code uses LPS, but the nifti file could be stored using any system.
        data = np.flip(data, axis=(0, 1))
        offset = (-offset[0], -offset[1], offset[2])
    return data, spacing, offset

@delegates(from_nifti)
def load_nifti(
    filepath: str,
    **kwargs) -> Tuple[Image, ImageSpacing3D, Point3D]:
    assert filepath.endswith('.nii') or filepath.endswith('.nii.gz'), "Filepath must end with .nii or .nii.gz"
    img = nib.load(filepath)
    return from_nifti(img, **kwargs)

def load_numpy(
    filepath: str,
    keys: Union[str, List[str]] = 'data') -> Union[Image, List[Image]]:
    assert filepath.endswith('.npz'), "Filepath must end with .npz"
    keys = arg_to_list(keys, str)
    data = np.load(filepath)
    items = [data[k] for k in keys]
    items = items[0] if len(items) == 1 else items
    return items

def to_nifti(
    data: Image,
    spacing: ImageSpacing3D,
    offset: Point3D) -> nib.nifti1.Nifti1Image:
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

def save_nifti(
    data: Image,
    spacing: ImageSpacing3D,
    offset: Point3D,
    filepath: str) -> None:
    assert filepath.endswith('.nii.gz'), "Filepath must end with .nii.gz"
    img = to_nifti(data, spacing, offset)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nib.save(img, filepath)

def save_numpy(
    data: Image,
    filepath: str) -> None:
    assert filepath.endswith('.npz'), "Filepath must end with .npz"
    np.savez_compressed(filepath, data=data)
