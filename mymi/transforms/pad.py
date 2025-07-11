import numpy as np
import torch
from typing import *

from mymi.geometry import get_extent
from mymi.typing import *
from mymi.utils import *

from .shared import assert_box_width, handle_non_spatial_dims

def __spatial_pad(
    image: ImageData3D,
    bounding_box: Union[PixelBox, VoxelBox],
    fill: Union[float, Literal['min']] = 'min',
    offset: Optional[Union[Point2D, Point3D]] = None,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True) -> ImageData3D:
    assert_box_width(bounding_box)
    fill = image.min() if fill == 'min' else fill
    if isinstance(fill, torch.Tensor):
        fill = fill.item()

    # Convert box to voxel coordinates.
    if use_patient_coords:
        min_mm, max_mm = bounding_box
        min = tuple(np.round((np.array(min_mm) - offset) / spacing).astype(int))
        max = tuple(np.round((np.array(max_mm) - offset) / spacing).astype(int))
    else:
        min, max = bounding_box

    # Perform padding - clip if padding is less than zero.
    size = np.array(image.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = tuple(zip(pad_min, pad_max))

    if isinstance(image, np.ndarray):
        image = np.pad(image, padding, constant_values=fill)
    elif isinstance(image, torch.Tensor):
        padding = list(reversed(padding))  # torch 'pad' operates from back to front.
        padding = tuple(torch.tensor(padding).flatten())
        image = torch.nn.functional.pad(image, padding, value=fill)

    return image

@delegates(__spatial_pad)
def pad(
    image: ImageData,
    *args, **kwargs) -> ImageData:
    assert_image(image)
    return handle_non_spatial_dims(__spatial_pad, image, *args, **kwargs)

def __spatial_centre_pad(
    image: ImageData3D,
    size: Union[Size2D, Size3D, FOV2D, FOV3D],
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True,
    **kwargs) -> ImageData3D:
    # Convert size to voxels if necessary.
    if use_patient_coords:
        assert spacing is not None
        size = tuple((np.array(size) / spacing).astype(int))

    # Determine padding amounts.
    to_pad = np.array(size) - image.shape
    box_min = -np.ceil(np.abs(to_pad / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform padding.
    output = __spatial_pad(image, bounding_box, use_patient_coords=False, **kwargs)

    return output

@delegates(__spatial_centre_pad)
def centre_pad(
    image: ImageData,
    *args, **kwargs) -> ImageData:
    assert_image(image)
    return handle_non_spatial_dims(__spatial_centre_pad, image, *args, **kwargs)
