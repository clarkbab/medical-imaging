import nibabel as nib
import os
from typing import *

from mymi.typing import *

from .python import delegates
from .utils import arg_to_list

def from_nifti(img: nib.nifti1.Nifti1Image) -> Tuple[Image, Spacing3D, Point3D]:
    data = img.get_fdata()
    affine = img.affine
    spacing = (float(affine[0][0]), float(affine[1][1]), float(affine[2][2]))
    offset = (float(affine[0][3]), float(affine[1][3]), float(affine[2][3]))
    return data, spacing, offset

@delegates(from_nifti)
def load_nifti(
    filepath: str,
    **kwargs) -> Tuple[Image, Spacing3D, Point3D]:
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
    spacing: Spacing3D,
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
    filepath: str,
    spacing: Spacing3D = (1, 1, 1),
    offset: Point3D = (0, 0, 0)) -> None:
    assert filepath.endswith('.nii.gz'), "Filepath must end with .nii.gz"
    img = to_nifti(data, spacing, offset)
    dirname = os.path.dirname(filepath)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    nib.save(img, filepath)

def save_numpy(
    data: Image,
    filepath: str) -> None:
    assert filepath.endswith('.npz'), "Filepath must end with .npz"
    np.savez_compressed(filepath, data=data)
