import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

from .shared import handle_non_spatial_dims

def centre_crop(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_centre_crop, data, *args, **kwargs)

def crop(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop, data, *args, **kwargs)

def crop_mm_3D(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: Point3D,
    bounding_box: Box3D) -> np.ndarray:
    assert len(data.shape) == 3, f"Input 'data' must have dimension 3."
    assert len(bounding_box[0]) == 3, f"Input 'bounding_box' must have dimension 3."
    assert len(bounding_box[1]) == 3, f"Input 'bounding_box' must have dimension 3."

    # Replace 'None' values.
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(3):
        if min[i] is None:
            min[i] = offset[i]
        if max[i] is None:
            max[i] = data.shape[i] * spacing[i] + offset[i]
    min, max = tuple(min), tuple(max)

    # Check box width.
    for min_i, max_i in zip(min, max):
        width = max_i - min_i
        if width <= 0:
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

    # Convert crop to voxels.
    min = (np.array(min) - offset) / spacing
    min = tuple(np.round(min, 0).astype(int))
    max = (np.array(max) - offset) / spacing
    max = tuple(np.round(max, 0).astype(int))

    # Crop the crop box to min/max allowed values.
    min = np.clip(min, a_min=0, a_max=data.shape)
    max = np.clip(max, a_min=0, a_max=data.shape)

    # Perform cropping.
    slices = tuple(slice(mn, mx) for mn, mx in zip(min, max))
    data = data[slices]

    return data

def crop_foreground(
    data: np.ndarray,
    *args, **kwargs) -> np.ndarray:
    return handle_non_spatial_dims(__spatial_crop_foreground, data, *args, **kwargs)

def crop_point(
    point: Union[Pixel, Voxel],
    crop: Union[Box2D, Box3D]) -> Optional[Union[Pixel, Voxel]]:
    # Check dimensions.
    assert len(point) == len(crop[0]) and len(point) == len(crop[1])

    crop = np.array(crop)
    point = np.array(point).reshape(1, crop.shape[1])

    # Get decision variables.
    decisions = np.stack((point >= crop[0], point < crop[1]), axis=0)

    # Check if point is in crop window.
    if np.all(decisions):
        point -= np.maximum(crop[0], 0)     # Don't pad by subtracting negative values.
        point = tuple(point.flatten())
    else:
        point = None

    return point

def crop_or_pad_point(
    point: Union[Pixel, Voxel],
    crop: Union[Box2D, Box3D]) -> Optional[Union[Pixel, Voxel]]:
    # Check dimensions.
    assert len(point) == len(crop[0]) and len(point) == len(crop[1])

    crop = np.array(crop)
    point = np.array(point).reshape(1, crop.shape[1])

    # Get decision variables.
    decisions = np.stack((point >= crop[0], point < crop[1]), axis=0)

    # Check if point is in crop window.
    if np.all(decisions):
        point -= crop[0]
        point = tuple(point.flatten())
    else:
        point = None

    return point

def crop_or_pad_box(
    box: Union[Box2D, Box3D],
    crop: Union[Box2D, Box3D]) -> Optional[Union[Box2D, Box3D]]:
    __assert_is_box(box)
    __assert_is_box(crop)

    # Return 'None' if no overlap between box and crop.
    box_min, box_max = box
    crop_min, crop_max = crop
    if not (np.all(np.array(crop_min) < box_max) and np.all(np.array(crop_max) > box_min)):
        return None

    # Otherwise use following rules to determine new box.
    box_min = tuple(np.maximum(np.array(box_min) - crop_min, 0))
    box_max = tuple(np.minimum(np.array(box_max) - crop_min, np.array(crop_max) - crop_min))
    box = (box_min, box_max)

    return box

def __assert_box_width(bounding_box: Union[Box2D, Box3D]) -> None:
    # Check box width.
    min, max = bounding_box
    for min_i, max_i in zip(min, max):
        width = max_i - min_i
        if width <= 0:
            raise ValueError(f"Box width must be positive, got '{bounding_box}'.")

def __assert_dims(
    data: np.ndarray,
    dims: Tuple[int]) -> None:
    n_dims = len(data.shape)
    if n_dims not in dims:
        raise ValueError(f"Data should have dims={dims}, got {n_dims}.")

def __assert_is_box(box: Union[Box2D, Box3D]) -> None:
    min, max = box
    if not np.all(list(mx >= mn for mn, mx in zip(min, max))):
        raise ValueError(f"Invalid box '{box}'.")
    
def __replace_box_none(
    bounding_box: Union[Box2D, Box3D],
    data_size: Union[ImageSize2D, ImageSize3D]) -> Tuple[Box2D, Box3D]:
    # Replace 'None' values.
    n_dims = len(data_size)
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(n_dims):
        if min[i] is None:
            min[i] = 0
        if max[i] is None:
            max[i] = data_size[i]
    min, max = tuple(min), tuple(max)
    return min, max

def __spatial_centre_crop(
    data: Image,
    size: Union[ImageSize2D, ImageSize3D]) -> Image:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)

    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = __spatial_crop(data, bounding_box)

    return output

def __spatial_crop(
    data: Image,
    bounding_box: Union[Box2D, Box3D]) -> Image:
    __assert_dims(data, (2, 3))
    bounding_box = __replace_box_none(bounding_box, data.shape)
    __assert_box_width(bounding_box)

    # Perform cropping.
    min, max = bounding_box
    size = np.array(data.shape)
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, size))
    data = data[slices]

    return data

def __spatial_crop_foreground(
    data: np.ndarray,
    bounding_box: Box3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    __assert_dims(data, (2, 3))
    bounding_box = __replace_box_none(bounding_box, data.shape)
    __assert_box_width(bounding_box)

    if fill == 'min':
        fill = np.min(data)
    cropped = data.copy()
    ct_size = cropped.shape

    # Crop upper bound.
    min, max = bounding_box
    for a in range(len(ct_size)):
        index = [slice(None)] * len(ct_size)
        index[a] = slice(0, min[a])
        cropped[tuple(index)] = fill

    # Crop upper bound.
    for a in range(len(ct_size)):
        index = [slice(None)] * len(ct_size)
        index[a] = slice(max[a], ct_size[a])
        cropped[tuple(index)] = fill

    return cropped
