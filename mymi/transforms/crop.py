import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple, Union

from mymi.geometry import get_box
from mymi.types import *

def crop_or_pad_2D(
    data: np.ndarray,
    bounding_box: Box2D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    # Convert args to 3D.
    data = np.expand_dims(data, axis=2)
    bounding_box = tuple((x, y, z) for (x, y), z in zip(bounding_box, (0, 1)))

    # Use 3D pad code.
    data = crop_or_pad_3D(data, bounding_box, fill=fill)

    # Remove final dimension.
    data = np.squeeze(data, axis=2)

    return data

def crop_2D(
    data: np.ndarray,
    bounding_box: Box2D) -> np.ndarray:
    # Convert args to 3D.
    data = np.expand_dims(data, axis=2)
    bounding_box = tuple((x, y, z) for (x, y), z in zip(bounding_box, (0, 1)))

    # Use 3D code.
    data = crop_3D(data, bounding_box)

    # Remove final dimension.
    data = np.squeeze(data, axis=2)

    return data

def crop_or_pad_4D(
    data: np.ndarray,
    bounding_box: Box3D,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    # Iterate over channel dimension, and use 3D code.
    ds = []
    for d in data:
        d = crop_or_pad_3D(d, bounding_box, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
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

def crop_or_pad_3D(
    data: np.ndarray,
    bounding_box: Box3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    if fill == 'min':
        fill = np.min(data)
    assert len(data.shape) == 3, f"Input 'data' must have dimension 3."

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

def centre_pad_4D(
    data: np.ndarray,
    size: ImageSize3D) -> np.ndarray:
    ds = []
    for d in data:
        d = centre_pad_3D(d, size)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def centre_pad_3D(
    data: np.ndarray,
    size: ImageSize3D) -> np.ndarray:
    # Determine padding amounts.
    to_pad = np.array(size) - data.shape
    box_min = -np.ceil(np.abs(to_pad / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform padding.
    output = pad_3D(data, bounding_box)

    return output

def centre_crop_3D(
    data: np.ndarray,
    size: ImageSize3D) -> np.ndarray:
    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop_3D(data, bounding_box)

    return output

def pad_2D(
    data: np.ndarray,
    bounding_box: Box2D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    assert len(data.shape) == 2, f"Input 'data' must have dimension 2."
    # Convert args to 3D.
    data = np.expand_dims(data, axis=2)
    bounding_box = tuple((x, y, z) for (x, y), z in zip(bounding_box, (0, 1)))

    # Use 3D code.
    data = pad_3D(data, bounding_box)

    # Remove final dimension.
    data = np.squeeze(data, axis=2)

    return data

def pad_3D(
    data: np.ndarray,
    bounding_box: Box3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    assert len(data.shape) == 3, f"Input 'data' must have dimension 3."
    fill = np.min(data) if fill == 'min' else fill

    # Check box coordinates.
    min, max = bounding_box
    for m in min:
        if m > 0:
            raise ValueError(f"Pad box must have min coordinates <= 0. Got '{bounding_box}'.")
    for m, d in zip(max, data.shape):
        if m < d:
            raise ValueError(f"Pad box must have max coordinates >= data shape. Got '{bounding_box}' for data shape '{data.shape}'.")

    min, max = bounding_box
    for i in range(3):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Pad width must be positive, got '{bounding_box}'.")

    # Perform padding.
    size = np.array(data.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = tuple(zip(pad_min, pad_max))
    data = np.pad(data, padding, constant_values=fill)

    return data

def pad_4D(
    data: np.ndarray,
    *args,
    **kwargs) -> np.ndarray:
    ds = []
    for d in data:
        d = pad_3D(d, *args, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def crop_3D(
    data: np.ndarray,
    bounding_box: Box3D) -> np.ndarray:
    assert len(data.shape) == 3, f"Input 'data' must have dimension 3."
    assert len(bounding_box[0]) == 3, f"Input 'bounding_box' must have dimension 3."
    assert len(bounding_box[1]) == 3, f"Input 'bounding_box' must have dimension 3."

    # Replace 'None' values.
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(3):
        if min[i] is None:
            min[i] = 0
        if max[i] is None:
            max[i] = data.shape[i]
    min, max = tuple(min), tuple(max)

    # Check box coordinates.
    for m, d in zip(min, data.shape):
        if m < 0:
            raise ValueError(f"Crop box must have min coordinates >= 0. Got '{bounding_box}'.")
        if m >= d:
            raise ValueError(f"Crop box must have min coordinates < data shape. Got '{bounding_box}' for data shape '{data.shape}'.")
    for m, d in zip(max, data.shape):
        if m <= 0:
            raise ValueError(f"Crop box must have max coordinates > 0. Got '{bounding_box}'.")
        if m > d:
            raise ValueError(f"Crop box must have max coordinates <= data shape. Got '{bounding_box}' for data shape '{data.shape}'.")

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

def crop_4D(
    data: np.ndarray,
    size: ImageSize3D,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    ds = []
    for d in data:
        d = crop_3D(d, size, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

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

def centre_crop_or_pad_3D(
    data: np.ndarray,
    size: ImageSize3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = crop_or_pad_3D(data, bounding_box, fill=fill)

    return output

def centre_crop_or_pad_4D(
    data: np.ndarray,
    size: ImageSize3D,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    ds = []
    for d in data:
        d = centre_crop_or_pad_3D(d, size, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def top_crop_or_pad_3D(
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
    output = crop_or_pad_3D(data, bounding_box, fill=fill)

    return output

def point_crop_or_pad_3D(
    data: np.ndarray,
    size: ImageSize3D,
    point: Point3D,
    fill: Union[float, Literal['min']] = 'min',
    return_box: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Box3D]]:
    # Perform the crop or pad.
    box = get_box(point, size)
    data = crop_or_pad_3D(data, box, fill=fill)

    if return_box:
        return (data, box)
    else:
        return data

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
    