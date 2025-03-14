import numpy as np
from typing import *

from mymi import logging
from mymi.typing import *

def centre_crop(
    data: np.ndarray,
    **kwargs) -> np.ndarray:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = spatial_centre_crop(data, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when centre-cropping 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = spatial_centre_crop(d, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to centre-crop should have (2, 3, 4) dims, got {n_dims}.")

    return output

def spatial_centre_crop(
    data: np.ndarray,
    size: Union[ImageSize2D, ImageSize3D]) -> np.ndarray:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)

    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop(data, bounding_box)

    return output

def crop(
    data: np.ndarray,
    *args, **kwargs) -> np.ndarray:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = spatial_crop(data, *args, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when cropping 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = spatial_crop(d, *args, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to crop should have (2, 3, 4) dims, got {n_dims}.")

    return output

def spatial_crop(
    data: np.ndarray,
    bounding_box: Union[Box2D, Box3D]) -> np.ndarray:
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
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

    # Perform cropping.
    size = np.array(data.shape)
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, size))
    data = data[slices]

    return data

def crop_mm_3D(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D,
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

def crop_foreground_3D(
    data: np.ndarray,
    crop: Box3D,
    background: Union[Literal['min'], float] = 'min') -> np.ndarray:
    if background == 'min':
        background = np.min(data)
    cropped = np.ones_like(data) * background
    slices = tuple(slice(min, max) for min, max in zip(*crop))
    cropped[slices] = data[slices]
    return cropped

def crop_point(
    point: Union[Point2D, Point3D],
    crop: Union[Box2D, Box3D]) -> Optional[Union[Point2D, Point3D]]:
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
    point: Union[Point2D, Point3D],
    crop: Union[Box2D, Box3D]) -> Optional[Union[Point2D, Point3D]]:
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

def __assert_is_box(box: Union[Box2D, Box3D]) -> None:
    min, max = box
    if not np.all(list(mx >= mn for mn, mx in zip(min, max))):
        raise ValueError(f"Invalid box '{box}'.")
    