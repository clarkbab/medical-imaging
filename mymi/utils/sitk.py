import numpy as np
import os
import SimpleITK as sitk
from typing import Optional, Tuple

from mymi.typing import ImageSpacing3D, PointMM3D

from .utils import transpose_image

def sitk_convert_point_RAS_LPS(data: Tuple[float]) -> Tuple[float]:
    data = list(data)
    data[0], data[1] = -data[0], -data[1]
    return tuple(data)

def sitk_convert_matrix_RAS_LPS(data: np.ndarray) -> np.ndarray:
    assert data.shape == (3, 3) or data.shape == (4, 4)
    data[0][0], data[1][1] = -data[0][0], -data[1][1]
    return data

def from_sitk(img: sitk.Image) -> Tuple[np.ndarray, ImageSpacing3D, PointMM3D]:
    data = sitk.GetArrayFromImage(img)
    # SimpleITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = tuple(img.GetSpacing())
    # SimpleITK always loads data in LPS coordinates. -x/y offset is required to convert to RAS coordinates
    # which is our standard (and DICOM/Slicer's).
    offset = tuple(img.GetOrigin())
    offset = list(img.GetOrigin())
    offset[0], offset[1] = -offset[0], -offset[1]
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
    data: np.ndarray,   # Fed in using RAS coordinates - our standard.
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
    img = sitk.GetImageFromArray(data, isVector=is_vector)
    img.SetSpacing(spacing)

    # The rest of our code base uses RAS coordinates (like DICOM/3D Slicer),
    # whereas sitk uses LPS coordinates. Convert direction/offset values to
    # LPS coordinates.
    direction = tuple(np.eye(3).flatten())
    direction = np.eye(3)
    direction = sitk_convert_matrix_RAS_LPS(direction).flatten()
    offset = sitk_convert_point_RAS_LPS(offset)
    img.SetDirection(direction)
    img.SetOrigin(offset)

    return img
