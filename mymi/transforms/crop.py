import numpy as np
from typing import *

from mymi.typing import *
from mymi.utils import *

from .shared import *

def __spatial_crop(
    image: Image,
    bounding_box: Union[Point2DBox, Point3DBox],
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    offset: Optional[Union[Point2D, Point3D]] = None,
    use_patient_coords: bool = True) -> Image:
    bounding_box = replace_box_none(bounding_box, image.shape, spacing=spacing, offset=offset, use_patient_coords=use_patient_coords)
    assert_box_width(bounding_box)

    # Convert box to voxel coordinates.
    if use_patient_coords:
        min_mm, max_mm = bounding_box
        min = tuple(np.round((np.array(min_mm) - offset) / spacing).astype(int))
        max = tuple(np.round((np.array(max_mm) - offset) / spacing).astype(int))
    else:
        min, max = bounding_box

    # Perform cropping.
    min, max = bounding_box
    size = np.array(image.shape)
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, size))
    image = image[slices]

    return image

@delegates(__spatial_crop)
def crop(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop, data, *args, **kwargs)

def __spatial_centre_crop_vox(
    data: Image,
    size: Union[Size2D, Size3D]) -> Image:
    n_dims = len(data.shape)
    assert n_dims in (2, 3)

    # Determine cropping/padding amounts.
    to_crop = data.shape - np.array(size)
    box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = __spatial_crop_vox(data, bounding_box)

    return output

@delegates(__spatial_centre_crop_vox)
def centre_crop_vox(
    data: Image,
    *args, **kwargs) -> Image:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_centre_crop_vox, data, *args, **kwargs)

def __spatial_crop_foreground_vox(
    data: LabelImage,
    bounding_box: Union[PixelBox, VoxelBox],
    fill: Union[float, Literal['min']] = 'min') -> LabelImage:
    __assert_dims(data, (2, 3))
    bounding_box = replace_box_none_vox(bounding_box, data.shape)
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

@delegates(__spatial_crop_foreground_vox)
def crop_foreground_vox(
    data: LabelImage,
    *args, **kwargs) -> LabelImage:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop_foreground_vox, data, *args, **kwargs)

def __spatial_crop_foreground_mm(
    data: LabelImage,
    bounding_box: Union[PixelBox, VoxelBox],
    spacing: Union[Spacing2D, Spacing3D],
    offset: Union[Point2D, Point3D],
    **kwargs) -> LabelImage:
    bounding_box = replace_box_none_mm(bounding_box, data.shape, spacing, offset)
    # Convert box to voxel coordinates.
    min_mm, max_mm = bounding_box
    min_vox = np.round((np.array(min_mm) - offset) / spacing)
    max_vox = np.round((np.array(max_mm) - offset) / spacing)

    return __spatial_crop_foreground_vox(data, (min_vox, max_vox), **kwargs)

@delegates(__spatial_crop_foreground_mm)
def crop_foreground_mm(
    data: LabelImage,
    *args, **kwargs) -> LabelImage:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop_foreground_mm, data, *args, **kwargs)

def crop_point(
    point: Union[Pixel, Voxel],
    crop: Union[PixelBox, VoxelBox]) -> Optional[Union[Pixel, Voxel]]:
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
    crop: Union[PixelBox, VoxelBox]) -> Optional[Union[Pixel, Voxel]]:
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
    box: Union[PixelBox, VoxelBox],
    crop: Union[PixelBox, VoxelBox]) -> Optional[Union[PixelBox, VoxelBox]]:
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

def __assert_dims(
    data: np.ndarray,
    dims: Tuple[int]) -> None:
    n_dims = len(data.shape)
    if n_dims not in dims:
        raise ValueError(f"Data should have dims={dims}, got {n_dims}.")

def __assert_is_box(box: Union[PixelBox, VoxelBox]) -> None:
    min, max = box
    if not np.all(list(mx >= mn for mn, mx in zip(min, max))):
        raise ValueError(f"Invalid box '{box}'.")

