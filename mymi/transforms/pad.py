import numpy as np
from typing import *

from mymi import logging
from mymi.typing import *

def centre_pad(
    data: np.ndarray,
    *args, **kwargs) -> np.ndarray:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = spatial_centre_pad(data, *args, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when padding 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = spatial_centre_pad(d, *args, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to centre_pad should have (2, 3, 4) dims, got {n_dims}.")

    return output

def spatial_centre_pad(
    data: np.ndarray,
    size: ImageSize3D) -> np.ndarray:
    # Determine padding amounts.
    to_pad = np.array(size) - data.shape
    box_min = -np.ceil(np.abs(to_pad / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform padding.
    output = spatial_pad(data, bounding_box)

    return output

def pad(
    data: np.ndarray,
    *args, **kwargs) -> np.ndarray:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = spatial_pad(data, *args, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when padding 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = spatial_pad(d, *args, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to pad should have (2, 3, 4) dims, got {n_dims}.")

    return output

def spatial_pad(
    data: np.ndarray,
    bounding_box: Union[Box2D, Box3D],
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)
    fill = np.min(data) if fill == 'min' else fill

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
    data = np.pad(data, padding, constant_values=fill)

    return data
