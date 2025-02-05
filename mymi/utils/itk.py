import itk
import logging
import numpy as np
import os
from typing import Optional, Tuple, Union

from mymi.typing import ImageSpacing3D, PointMM3D

def itk_convert_LPS_and_RAS(
    direction: Optional[Union[np.ndarray]] = None,
    offset: Optional[PointMM3D] = None) -> Tuple[Optional[np.ndarray], Optional[PointMM3D]]:
    if direction is not None:
        direction[0][0], direction[1][1] = -direction[0][0], -direction[1][1]
    if offset is not None:
        offset = list(offset)
        offset[0], offset[1] = -offset[0], -offset[1]
        offset = tuple(offset)
    return direction, offset

def from_itk(img: itk.Image) -> Tuple[np.ndarray, ImageSpacing3D, PointMM3D]:
    data = itk.GetArrayFromImage(img)
    # ITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = tuple(img.GetSpacing())
    # ITK always loads data in LPS coordinates. -x/y offset is required to convert to RAS coordinates
    # which is our standard (and DICOM/Slicer's).
    offset = tuple(img.GetOrigin())
    _, offset = itk_convert_LPS_and_RAS(offset=offset)
    return data, spacing, offset

def load_itk(filepath: str) -> itk.Image:
    # When ITK loads an image, it converts it to LPS coordinates.
    # This means that all ITK images will have -x/y coordinates for
    # origin and direction. Our code handles this when converting to/from numpy.
    img = itk.imread(filepath)
    return img

def to_itk(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D,
    is_vector: bool = False) -> itk.Image:
    # Convert to ITK types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    spacing = tuple(float(s) for s in spacing)
    offset = tuple(float(o) for o in offset)
    
    # ITK **sometimes** flips the data coordinates (x, y, z) -> (z, y, x) when converting from numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    # Preprocessing, such as np.transpose and np.moveaxis can change the numpy array indexing style
    # from the default C-style to Fortran-style. ITK will flip coordinates for C-style but not F-style.
    data = data.transpose()
    # We can use 'copy' to reset the indexing to C-style and ensure that ITK flips coordinates. If we
    # don't do this, code called before 'to_itk' could affect the behaviour of 'GetImageFromArray', which
    # was very confusing for me.
    data = data.copy()
    img = itk.GetImageFromArray(data, is_vector=is_vector)
    img.SetSpacing(spacing)

    # ITK uses LPS coordinates, but we're assuming the incoming numpy data
    # is using RAS coordinates, so convert.
    direction = np.eye(3)
    direction, offset = itk_convert_LPS_and_RAS(direction=direction, offset=offset)
    direction = itk.matrix_from_array(direction)
    img.SetDirection(direction)
    img.SetOrigin(offset)

    return img
