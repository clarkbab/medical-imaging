import itk
import numpy as np
from typing import *

from mymi.typing import *

def from_itk_image(img: itk.Image) -> Tuple[ImageArray, Spacing3D, Point3D]:
    data = itk.GetArrayFromImage(img)
    # ITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = tuple(img.GetSpacing())
    # ITK assumes loaded nifti data is using RAS coordinates, so they set negative offsets
    # and directions. For all images we write to nifti, they'll be in LPS, so undo ITK changes.
    # The image data is not flipped by ITK.
    offset = list(img.GetOrigin())
    offset[0], offset[1] = -offset[0], -offset[1]
    offset = tuple(offset)
    return data, spacing, offset

def load_itk_image(filepath: str) -> itk.Image:
    return itk.imread(filepath)

def to_itk_image(
    data: ImageArray,
    spacing: Spacing3D,
    offset: Point3D,
    vector: bool = False,
    reverse_xy: bool = False) -> itk.Image:
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
    if vector:
        assert data.shape[0] == 3
        # Move our channels to last dim - for itk.
        data = np.moveaxis(data, 0, -1)
    img = itk.GetImageFromArray(data, is_vector=vector)
    img.SetSpacing(spacing)
    img.SetOrigin(offset)

    return img
