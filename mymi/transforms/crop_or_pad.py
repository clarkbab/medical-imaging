import numpy as np
from typing import Tuple

from mymi import types

def crop_or_pad_2D(
    input: np.ndarray,
    bounding_box: types.Box2D,
    fill: float = 0) -> np.ndarray:
    """
    returns: a 3D array with dimensions cropped or padded.
    args:
        input: the 2D input array.
        bounding_box: a 2D box defining the crop/pad.
    kwargs:
        fill: the default fill value for padded elements.
    """
    # Convert args to 3D.
    input = np.expand_dims(input, axis=2)
    bounding_box = tuple((x, y, z) for (x, y), z in zip(bounding_box, (0, 1)))

    # Use 3D pad code.
    input = crop_or_pad_3D(input, bounding_box, fill=fill)

    # Remove final dimension.
    input = np.squeeze(input, axis=2)

    return input

def crop_or_pad_3D(
    input: np.ndarray,
    bounding_box: types.Box3D,
    fill: float = 0) -> np.ndarray:
    """
    returns: a 3D array with dimensions cropped or padded.
    args:
        input: the 3D input array.
        bounding_box: a 3D box defining the crop/pad.
    kwargs:
        fill: the default fill value for padded elements.
    """
    min, max = bounding_box
    for i in range(3):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

    # Perform padding.
    size = np.array(input.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = tuple(zip(pad_min, pad_max))
    input = np.pad(input, padding, constant_values=fill)

    # Perform cropping.
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, input.shape))
    input = input[slices]

    return input

def centre_crop_or_pad_3D(
    input: np.ndarray,
    size: types.ImageSize3D,
    fill: float = 0) -> np.ndarray:
    """
    returns: an array cropped/padded along each axis. When an uneven amount is cropped 
        from an axis, more is removed from the left-hand side. When an uneven amount is
        padded on an axis, more is added to the left-hand side, thus allowing this 
        function to invert itself.
    args:
        input: the array to resize.
        size: the new size.
    kwargs:
        fill: the default padding fill value.
    """
    # Determine cropping/padding amounts.
    to_crop = input.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop_or_pad_3D(input, bounding_box, fill=fill)

    return output
