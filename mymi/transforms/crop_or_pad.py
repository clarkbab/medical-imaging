import numpy as np
from typing import *

from mymi.geometry import get_extent
from mymi.typing import *
from mymi.utils import *

from .shared import *

def __spatial_crop_or_pad(
    image: Image,
    bounding_box: Union[Point2DBox, Point3DBox],
    fill: Union[float, Literal['min']] = 'min',
    offset: Optional[Union[Point2D, Point3D]] = None,
    return_inverse: bool = False,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True) -> Image:
    bounding_box = replace_box_none(bounding_box, image.shape, spacing=spacing, offset=offset, use_patient_coords=use_patient_coords)
    assert_box_width(bounding_box)
    fill = np.min(image) if fill == 'min' else fill

    # Convert box to voxel coordinates.
    if use_patient_coords:
        min_mm, max_mm = bounding_box
        min = tuple(np.round((np.array(min_mm) - offset) / spacing).astype(int))
        max = tuple(np.round((np.array(max_mm) - offset) / spacing).astype(int))
        inv_box = get_extent(image, spacing=spacing, offset=offset)
    else:
        min, max = bounding_box

    # Perform padding.
    n_dims = len(image.shape)
    size = np.array(image.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = list(zip(pad_min, pad_max))
    if n_dims == 4:
        assert image.shape[0] == 3
        padding = [0, 0] + padding  # vector image.
    padding = tuple(padding)
    image = np.pad(image, padding, constant_values=fill)

    # Perform cropping.
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = list(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, image.shape))
    if n_dims == 4:
        slices = [slice(None)] + slices
    slices = tuple(slices)
    image = image[slices]

    if return_inverse:
        return image, inv_box
    else:
        return image

@delegates(__spatial_crop_or_pad)
def crop_or_pad(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop_or_pad, data, *args, **kwargs)

def __spatial_centre_crop_or_pad_vox(
    data: Image,
    size: Size3D,
    fill: Union[float, Literal['min']] = 'min') -> Image:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)

    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = __spatial_crop_or_pad(data, bounding_box, fill=fill)

    return output

@delegates(__spatial_centre_crop_or_pad_vox)
def centre_crop_or_pad_vox(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_centre_crop_or_pad_vox, data, *args, **kwargs)
