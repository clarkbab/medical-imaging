import numpy as np
import os
import SimpleITK as sitk
from typing import Optional, Tuple

from mymi.types import ImageSpacing3D, PointMM3D

def sitk_convert_LPS_and_RAS(
    direction: Optional[Tuple[float]] = None,
    offset: Optional[PointMM3D] = None) -> Tuple[Optional[Tuple[float]], Optional[PointMM3D]]:
    if direction is not None:
        direction[0], direction[4] = -direction[0], -direction[4]
    if offset is not None:
        offset = list(offset)
        offset[0], offset[1] = -offset[0], -offset[1]
        offset = tuple(offset)
    return direction, offset

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
    # No way to load the 'TransformEnum' object? Using hard-coded values.
    if t == 14:
        transform = sitk.DisplacementFieldTransform(transform)

    return transform

def save_sitk_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteTransform(transform, filepath)

def to_sitk(
    data: np.ndarray,
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
    data = data.transpose()
    # We can use 'copy' to reset the indexing to C-style and ensure that SimpleITK flips coordinates. If we
    # don't do this, code called before 'to_sitk' could affect the behaviour of 'GetImageFromArray', which
    # was very confusing for me.
    data = data.copy()
    img = sitk.GetImageFromArray(data, isVector=is_vector)
    img.SetSpacing(spacing)

    # ITK uses LPS coordinates, but we're assuming the incoming numpy data
    # is using RAS coordinates, so convert.
    direction = np.eye(3)
    direction[0][0], direction[1][1] = -1, -1
    direction = tuple(direction.flatten())
    offset = list(offset)
    offset[0], offset[1] = -offset[0], -offset[1]
    offset = tuple(offset)
    img.SetDirection(direction)
    img.SetOrigin(offset)

    return img