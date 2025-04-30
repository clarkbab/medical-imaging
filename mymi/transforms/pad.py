import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

from .shared import handle_non_spatial_dims

def centre_pad(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_centre_pad, data, *args, **kwargs)

def pad(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_pad, data, *args, **kwargs)

def __spatial_centre_pad(
    data: Image,
    size: ImageSize3D,
    **kwargs) -> Image:
    # Determine padding amounts.
    to_pad = np.array(size) - data.shape
    box_min = -np.ceil(np.abs(to_pad / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform padding.
    output = __spatial_pad(data, bounding_box, **kwargs)

    return output

def __spatial_pad(
    data: Image,
    bounding_box: Union[Box2D, Box3D],
    fill: Union[float, Literal['min']] = 'min') -> Image:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)
    fill = data.min() if fill == 'min' else fill

    # Check width of padding box.
    min, max = bounding_box
    n_dims = len(data.shape)
    for i in range(n_dims):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Pad width must be positive, got '{bounding_box}'.")

    # Perform padding - clip if padding is less than zero.
    size = np.array(data.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = tuple(zip(pad_min, pad_max))

    if isinstance(data, np.ndarray):
        data = np.pad(data, padding, constant_values=fill)
    elif isinstance(data, torch.Tensor):
        padding = list(reversed(padding))  # torch 'pad' operates from back to front.
        padding = tuple(torch.tensor(padding).flatten())
        data = torch.nn.functional.pad(data, padding, value=fill)

    return data
