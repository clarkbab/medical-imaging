import numpy as np
import os
import SimpleITK as sitk
from typing import *

from mymi.typing import *

from .utils import transpose_image

def from_sitk_image(img: sitk.Image) -> Tuple[np.ndarray, ImageSpacing3D, PointMM3D]:
    data = sitk.GetArrayFromImage(img)
    # SimpleITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = tuple(img.GetSpacing())
    offset = list(img.GetOrigin())

    # Although SimpleITK uses LPS coordinates, the direction matrix can change this.
    # Convert to LPS coordinates.
    direction = img.GetDirection()
    if direction[0] == -1:
        # Convert x from R -> L coordinates.
        data = np.flip(data, axis=0)
        offset[0] = -offset[0]
    if direction[4] == -1:
        # Convert y from A -> P coordinates.
        data = np.flip(data, axis=1)
        offset[1] = -offset[1]
    if direction[8] == -1:
        # Convert z from I -> S coordinates.
        data = np.flip(data, axis=2)
        offset[2] = -offset[2]

    offset = tuple(offset)
    return data, spacing, offset

def load_sitk(filepath: str) -> sitk.Image:
    # When SimpleITK loads an image, it converts it to LPS coordinates.
    # This means that all SimpleITK images will have -x/y coordinates for
    # origin and direction. Our code handles this when converting to/from numpy.
    img = sitk.ReadImage(filepath)
    return img

def load_sitk_transform(
    filepath: str) -> sitk.Transform:
    transform = sitk.ReadTransform(filepath)

    # Cast to correct type.
    t = transform.GetTransformEnum()
    if t == sitk.sitkDisplacementField:
        transform = sitk.DisplacementFieldTransform(transform)
    elif t == sitk.sitkEuler:
        transform = sitk.Euler3DTransform(transform)

    return transform

def save_sitk_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteTransform(transform, filepath)

def to_sitk_image(
    data: np.ndarray,   # We use LPS coordinates - the same as SimpleITK!
    spacing: ImageSpacing3D,
    offset: PointMM3D,
    is_vector: bool = False) -> sitk.Image:
    # Convert to SimpleITK data types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    spacing = tuple(float(s) for s in spacing)
    offset = tuple(float(o) for o in offset)
    
    # SimpleITK **sometimes** flips the data coordinates (x, y, z) -> (z, y, x) when converting from numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    # Preprocessing, such as np.transpose and np.moveaxis can change the numpy array indexing style
    # from the default C-style to Fortran-style. SimpleITK will flip coordinates for C-style but not F-style.
    data = transpose_image(data, is_vector=is_vector)
    # We can use 'copy' to reset the indexing to C-style and ensure that SimpleITK flips coordinates. If we
    # don't do this, code called before 'to_sitk' could affect the behaviour of 'GetImageFromArray', which
    # was very confusing for me.
    data = data.copy()
    if is_vector:
        assert data.shape[0] == 3
        # Move our channels to last dim - for sitk.
        data = np.moveaxis(data, 0, -1)
    img = sitk.GetImageFromArray(data, isVector=is_vector)
    img.SetSpacing(spacing)

    # SimpleITK uses LPS coordinates, so we need to reverse the direction matrix for
    # x/y axes to show that our image data is coming in backwards. Same for the offset.
    direction = np.eye(3)
    img.SetDirection(direction.flatten())
    img.SetOrigin(offset)

    return img

def to_sitk_transform(
    dvf: np.ndarray,   # (3, X, Y, Z)
    spacing: ImageSpacing3D,
    offset: PointMM3D) -> sitk.Transform:
    dvf = dvf.astype(np.float64)
    assert dvf.shape[0] == 3
    dvf_sitk = to_sitk_image(dvf, spacing, offset, is_vector=True)
    dvf_transform = sitk.DisplacementFieldTransform(dvf_sitk)
    return dvf_transform
