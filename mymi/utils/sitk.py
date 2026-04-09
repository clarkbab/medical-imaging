from dicomset.typing import *
import numpy as np
import SimpleITK as sitk
from typing import *

from mymi.typing import *

from .affine import create_affine, affine_origin, affine_spacing
from .utils import transpose_image

def from_sitk_image(
    img: sitk.Image,
    ) -> Tuple[ImageArray, Affine]:
    data = sitk.GetArrayFromImage(img)
    # SimpleITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    affine = create_affine(spacing, origin)
    return data, affine

def to_sitk_image(
    data: ChannelImage | Image,
    affine: AffineMatrix | None = None,
    vector: bool = False,
    n_vector_elements: int = 3,
    ) -> sitk.Image:
    # Convert to SimpleITK data types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    
    # SimpleITK **sometimes** flips the data coordinates (x, y, z) -> (z, y, x) when converting from numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    # Preprocessing, such as np.transpose and np.moveaxis can change the numpy array indexing style
    # from the default C-style to Fortran-style. SimpleITK will flip coordinates for C-style but not F-style.
    data = transpose_image(data, vector=vector, n_vector_elements=n_vector_elements)
    # We can use 'copy' to reset the indexing to C-style and ensure that SimpleITK flips coordinates. If we
    # don't do this, code called before 'to_sitk' could affect the behaviour of 'GetImageFromArray', which
    # was very confusing for me.
    data = data.copy() if isinstance(data, np.ndarray) else data.clone()
    if vector:
        assert data.shape[0] == n_vector_elements, f"Expected first dimension of data to be vector elements, got {data.shape[0]} but expected {n_vector_elements}."
        # Sitk expects vector dimension to be last.
        moveaxis_fn = np.moveaxis if isinstance(data, np.ndarray) else torch.moveaxis
        data = moveaxis_fn(data, 0, -1)
    img = sitk.GetImageFromArray(data, isVector=vector)
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)

    return img

def dvf_to_sitk_transform(
    dvf: ChannelImage,
    affine: AffineMatrix | None = None,
    ) -> sitk.Transform:
    dvf = dvf.astype(np.float64)
    assert dvf.shape[0] == 3
    dvf_sitk = to_sitk_image(dvf, affine=affine, vector=True)
    dvf_transform = sitk.DisplacementFieldTransform(dvf_sitk)
    return dvf_transform
