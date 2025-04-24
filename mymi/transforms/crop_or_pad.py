import numpy as np
from typing import *

from mymi.geometry import get_box
from mymi import logging
from mymi.typing import *

def centre_crop_or_pad(
    data: np.ndarray,
    *args, **kwargs) -> np.ndarray:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = spatial_centre_crop_or_pad(data, *args, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when centre_crop_or_padding 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = spatial_centre_crop_or_pad(d, *args, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to centre_crop_or_pad should have (2, 3, 4) dims, got {n_dims}.")

    return output

def spatial_centre_crop_or_pad(
    data: np.ndarray,
    size: ImageSize3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)

    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = spatial_crop_or_pad(data, bounding_box, fill=fill)

    return output

def crop_or_pad(
    data: np.ndarray,
    *args, **kwargs) -> np.ndarray:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = spatial_crop_or_pad(data, *args, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when crop_or_padding 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = spatial_crop_or_pad(d, *args, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to crop_or_pad should have (2, 3, 4) dims, got {n_dims}.")

    return output

def spatial_crop_or_pad(
    data: np.ndarray,
    bounding_box: Box3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    fill = np.min(data) if fill == 'min' else fill
    n_dims = len(data.shape)
    assert n_dims in (2, 3)

    # Replace 'None' values.
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(n_dims):
        if min[i] is None:
            min[i] = 0
        if max[i] is None:
            max[i] = data.shape[i]
    min, max = tuple(min), tuple(max)

    # Check box width.
    for min_i, max_i in zip(min, max):
        width = max_i - min_i
        if width <= 0:
            raise ValueError(f"Crop_or_pad width must be positive, got '{bounding_box}'.")

    # Perform padding.
    size = np.array(data.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = tuple(zip(pad_min, pad_max))
    data = np.pad(data, padding, constant_values=fill)

    # Perform cropping.
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, data.shape))
    data = data[slices]

    return data

def point_crop_or_pad(
    data: np.ndarray,
    size: ImageSize3D,
    point: Voxel,
    fill: Union[float, Literal['min']] = 'min',
    return_box: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Box3D]]:
    # Perform the crop or pad.
    box = get_box(point, size)
    data = crop_or_pad(data, box, fill=fill)

    if return_box:
        return (data, box)
    else:
        return data

def top_crop_or_pad(
    data: np.ndarray,
    size: ImageSize3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    # Centre crop x/y axes.
    to_crop = data.shape[:2] - np.array(size[:2])
    xy_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    xy_max = xy_min + size[:2]

    # Top crop z axis to maintain HN region.
    z_max = data.shape[2]
    z_min = z_max - size[2]

    # Perform crop or padding.
    bounding_box = ((*xy_min, z_min), (*xy_max, z_max)) 
    output = crop_or_pad(data, bounding_box, fill=fill)

    return output

def centre_crop_or_pad_vector_3D(
    data: np.ndarray,   # (3, X, Y, Z)
    size: ImageSize3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    # Determine cropping/padding amounts.
    size_3D = data.shape[1:]
    to_crop = size_3D - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop_or_pad_vector_3D(data, bounding_box, fill=fill)

    return output

def crop_or_pad_vector_3D(
    data: np.ndarray,   # (3, X, Y, Z)
    bounding_box: Box3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    if fill == 'min':
        fill = np.min(data)
    assert len(data.shape) == 4, f"Input 'data' must have dimension 4."

    min, max = bounding_box
    for i in range(3):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

    # Perform padding.
    original_size_3D = np.array(data.shape[1:])
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - original_size_3D).clip(0)
    padding = list(zip(pad_min, pad_max))
    padding = tuple([(0, 0)] + padding)
    data = np.pad(data, padding, constant_values=fill)

    # Perform cropping.
    padded_size_3D = data.shape[1:]    # Will have changed after padding.
    crop_min = np.array(min).clip(0)
    crop_max = (original_size_3D - max).clip(0)
    slices = list(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, padded_size_3D))
    slices = tuple([slice(None)] + slices)
    data = data[slices]

    return data
