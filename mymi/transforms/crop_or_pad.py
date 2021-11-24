import numpy as np
from typing import Tuple, Union

from mymi.geometry import get_box
from mymi import types

def crop_or_pad_2D(
    data: np.ndarray,
    bounding_box: types.Box2D,
    fill: float = 0) -> np.ndarray:
    # Convert args to 3D.
    data = np.expand_dims(data, axis=2)
    bounding_box = tuple((x, y, z) for (x, y), z in zip(bounding_box, (0, 1)))

    # Use 3D pad code.
    data = crop_or_pad_3D(data, bounding_box, fill=fill)

    # Remove final dimension.
    data = np.squeeze(data, axis=2)

    return data

def crop_or_pad_3D(
    data: np.ndarray,
    bounding_box: types.Box3D,
    fill: float = 0) -> np.ndarray:
    min, max = bounding_box
    for i in range(3):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

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

def centre_crop_or_pad_3D(
    data: np.ndarray,
    size: types.ImageSize3D,
    fill: float = 0) -> np.ndarray:
    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop_or_pad_3D(data, bounding_box, fill=fill)

    return output

def top_crop_or_pad_3D(
    data: np.ndarray,
    size: types.ImageSize3D,
    fill: float = 0) -> np.ndarray:
    # Centre crop x/y axes.
    to_crop = data.shape[:2] - np.array(size[:2])
    xy_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    xy_max = xy_min + size[:2]

    # Top crop z axis to maintain HN region.
    z_max = data.shape[2]
    z_min = z_max - size[2]

    # Perform crop or padding.
    bounding_box = ((*xy_min, z_min), (*xy_max, z_max)) 
    output = crop_or_pad_3D(data, bounding_box, fill=fill)

    return output

def point_crop_or_pad_3D(
    data: np.ndarray,
    size: types.ImageSize3D,
    point: types.Point3D,
    fill: float = 0,
    return_box: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, types.Box3D]]:
    # Perform the crop or pad.
    box = get_box(point, size)
    data = crop_or_pad_3D(data, box, fill=fill)

    if return_box:
        return (data, box)
    else:
        return data
