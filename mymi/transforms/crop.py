import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple, Union

from mymi.geometry import get_box
from mymi.types import Box2D, Box3D, Size3D, Point2D, Point3D

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
    ds = []
    for d in data:
        d = crop_or_pad_3D(d, bounding_box, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

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
    size: Size3D) -> np.ndarray:
    ds = []
    for d in data:
        d = centre_pad_3D(d, size)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def centre_pad_3D(
    data: np.ndarray,
    size: Size3D) -> np.ndarray:
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
    size: Size3D) -> np.ndarray:
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
    fill = np.min(data) if fill == 'min' else fill

    min, max = bounding_box
    for i in range(2):
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

def pad_3D(
    data: np.ndarray,
    bounding_box: Box3D,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    assert len(data.shape) == 3, f"Input 'data' must have dimension 3."
    fill = np.min(data) if fill == 'min' else fill

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

    min, max = bounding_box
    for i in range(3):
        width = max[i] - min[i]
        if width <= 0:
            raise ValueError(f"Crop width must be positive, got '{bounding_box}'.")

    # Perform cropping.
    size = np.array(data.shape)
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, size))
    data = data[slices]

    return data

def crop_4D(
    data: np.ndarray,
    size: Size3D,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    ds = []
    for d in data:
        d = crop_3D(d, size, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def crop_foreground_3D(
    data: np.ndarray,
    crop: Box3D) -> np.ndarray:
    cropped = np.zeros_like(data).astype(bool)
    slices = tuple(slice(min, max) for min, max in zip(*crop))
    cropped[slices] = data[slices]
    return cropped

def centre_crop_or_pad_3D(
    data: np.ndarray,
    size: Size3D,
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
    size: Size3D,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    ds = []
    for d in data:
        d = centre_crop_or_pad_3D(d, size, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def top_crop_or_pad_3D(
    data: np.ndarray,
    size: Size3D,
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
    size: Size3D,
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
    