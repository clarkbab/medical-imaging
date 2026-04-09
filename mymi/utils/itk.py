from dicomset.typing import *
from dicomset.utils import affine_spacing, affine_origin, create_affine
import itk
import numpy as np
from typing import *

def from_itk_image(
    img: itk.Image,
    ) -> Tuple[Image, AffineMatrix3D]:
    data = itk.GetArrayFromImage(img)
    # ITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = tuple(img.GetSpacing())
    # ITK assumes loaded nifti data is using RAS coordinates, so they set negative origins
    # and directions. For all images we write to nifti, they'll be in LPS, so undo ITK changes.
    # The image data is not flipped by ITK.
    origin = list(img.GetOrigin())
    origin[0], origin[1] = -origin[0], -origin[1]
    origin = tuple(origin)
    affine = create_affine(spacing, origin)
    return data, affine

def load_itk_image(filepath: str) -> itk.Image:
    return itk.imread(filepath)

def to_itk_image(
    data: Image3D,
    affine: AffineMatrix3D | None = None,
    vector: bool = False,
    reverse_xy: bool = False,
    ) -> itk.Image:
    # Convert to ITK types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    
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
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)

    return img
