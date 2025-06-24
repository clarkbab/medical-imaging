import numpy as np
import SimpleITK as sitk
from typing import *

from mymi.typing import *

from .utils import transpose_image

def from_sitk_image(
    img: sitk.Image,
    img_type: Literal['mha', 'nii'] = 'nii') -> Tuple[Image, Spacing3D, Point3D]:
    data = sitk.GetArrayFromImage(img)
    # SimpleITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = tuple(img.GetSpacing())
    offset = list(img.GetOrigin())
    if img_type == 'mhd':
        pass
    elif img_type == 'nii':
        # ITK assumes loaded nifti data is using RAS coordinates, so they set negative offsets
        # and directions. For all images we write to nifti, they'll be in LPS, so undo ITK changes.
        # The image data is not flipped by ITK.
        offset[0], offset[1] = -offset[0], -offset[1]
    offset = tuple(offset)
    return data, spacing, offset

def sitk_load_image(filepath: str) -> Tuple[Image, Spacing3D, Point3D]:
    if filepath.endswith('.mhd'):
        img_type = 'mhd'
    elif filepath.endswith('.nii.gz') or filepath.endswith('.nii'):
        img_type = 'nii'
    else:
        raise ValueError(f'Unsupported file type: {filepath}.')
    img = sitk.ReadImage(filepath)
    return from_sitk_image(img, img_type=img_type)

def to_sitk_image(
    data: Union[Image, VectorImage],   # We use LPS coordinates - the same as SimpleITK!
    spacing: Spacing3D = (1, 1, 1),
    offset: Point3D = (0, 0, 0),
    vector: bool = False) -> sitk.Image:
    # Convert to SimpleITK data types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    spacing = tuple(float(s) for s in spacing)
    offset = tuple(float(o) for o in offset)
    
    # SimpleITK **sometimes** flips the data coordinates (x, y, z) -> (z, y, x) when converting from numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    # Preprocessing, such as np.transpose and np.moveaxis can change the numpy array indexing style
    # from the default C-style to Fortran-style. SimpleITK will flip coordinates for C-style but not F-style.
    data = transpose_image(data, vector=vector)
    # We can use 'copy' to reset the indexing to C-style and ensure that SimpleITK flips coordinates. If we
    # don't do this, code called before 'to_sitk' could affect the behaviour of 'GetImageFromArray', which
    # was very confusing for me.
    data = data.copy()
    if vector:
        assert data.shape[0] == 3
        # Move our channels to last dim - for sitk.
        data = np.moveaxis(data, 0, -1)
    img = sitk.GetImageFromArray(data, isVector=vector)
    img.SetSpacing(spacing)
    img.SetOrigin(offset)

    return img

def dvf_to_sitk_transform(
    dvf: VectorImage,   # (3, X, Y, Z)
    spacing: Spacing3D = (1, 1, 1),
    offset: Point3D = (0, 0, 0)) -> sitk.Transform:
    dvf = dvf.astype(np.float64)
    assert dvf.shape[0] == 3
    dvf_sitk = to_sitk_image(dvf, spacing, offset, vector=True)
    dvf_transform = sitk.DisplacementFieldTransform(dvf_sitk)
    return dvf_transform
