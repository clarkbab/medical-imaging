import numpy as np
from typing import Tuple

def crop_or_pad(
    input: np.ndarray,
    crop_or_padding: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    fill: float = 0) -> np.ndarray:
    """
    returns: a 3D array with dimensions cropped or padded.
    args:
        input: the input tensor.
        crop_or_padding: number of voxels to add remove from each dimension.
    kwargs:
        fill: the default fill value for padded voxels.
    """
    # Perform padding.
    padding = np.array(crop_or_padding).clip(0)
    input = np.pad(input, padding, constant_values=fill)

    # Perform cropping.
    cropping = (-np.array(crop_or_padding)).clip(0)
    mins = tuple(d[0] for d in cropping)
    maxs = tuple(s - d[1] for d, s in zip(cropping, input.shape))
    slices = tuple(slice(min, max) for min, max in zip(mins, maxs))
    input = input[slices]

    return input

def centre_crop_or_pad(
    input: np.ndarray,
    size: Tuple[int, int, int],
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
    amounts = np.array(size) - input.shape
    crop_or_padding = tuple((int(a / 2), int(a / 2)) if (a / 2).is_integer() else (int((a + np.sign(a)) / 2), int(((a + np.sign(a)) / 2) - np.sign(a))) for a in amounts)

    # Perform crop or padding.
    output = crop_or_pad(input, crop_or_padding, fill=fill)

    return output
