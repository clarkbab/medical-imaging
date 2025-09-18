import numpy as np
from typing import *

from mymi.typing import *
from mymi.utils import *

from .transforms import *

def __spatial_crop(
    image: ImageArray,
    bounding_box: Union[BoxMM2D, BoxMM3D],
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    origin: Optional[Union[Point2D, Point3D]] = None,
    use_patient_coords: bool = True) -> ImageArray:
    bounding_box = replace_box_none(bounding_box, image.shape, spacing=spacing, origin=origin, use_patient_coords=use_patient_coords)
    assert_box_width(bounding_box)

    # Convert box to voxel coordinates.
    if use_patient_coords:
        min_mm, max_mm = bounding_box
        min = tuple(np.round((np.array(min_mm) - origin) / spacing).astype(int))
        max = tuple(np.round((np.array(max_mm) - origin) / spacing).astype(int))
    else:
        min, max = bounding_box

    # Perform cropping.
    size = np.array(image.shape)
    crop_min = np.array(min).clip(0)
    crop_max = (size - max).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, size))
    image = image[slices]

    return image

@delegates(__spatial_crop)
def crop(
    data: ImageArray,
    *args, **kwargs) -> ImageArray:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop, data, *args, **kwargs)

def __spatial_centre_crop(
    data: ImageArray,
    size: Union[Size, SizeMM],
    spacing: Optional[Spacing] = None,
    use_patient_coords: bool = True) -> ImageArray:

    # Determine cropping/padding amounts.
    if use_patient_coords:
        assert spacing is not None
        fov_mm = np.array(size) * spacing
        to_crop_mm = fov_mm - size
        to_crop_vox = np.round(to_crop_mm / spacing).astype(int)
    else:
        to_crop_vox = data.shape - np.array(size)
    box_min = np.sign(to_crop_vox) * np.ceil(np.abs(to_crop_vox / 2)).astype(int)
    box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    output = __spatial_crop(data, bounding_box, use_patient_coords=False)

    return output

@delegates(__spatial_centre_crop)
def centre_crop(
    data: ImageArray,
    *args, **kwargs) -> ImageArray:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_centre_crop, data, *args, **kwargs)

def __spatial_crop_foreground(
    data: LabelArray,
    bounding_box: Union[BoxMM2D, BoxMM3D, Box2D, Box3D],
    fill: Union[float, Literal['min']] = 'min',
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    origin: Optional[Union[Point2D, Point3D]] = None,
    use_patient_coords: bool = True) -> LabelArray:
    bounding_box = replace_box_none(bounding_box, data.shape, spacing=spacing, origin=origin, use_patient_coords=use_patient_coords)
    assert_box_width(bounding_box)

    # Convert box to voxel coordinates.
    min, max = bounding_box
    if use_patient_coords:
        min = np.round((np.array(min) - origin) / spacing)
        max = np.round((np.array(max) - origin) / spacing)

    if fill == 'min':
        fill = np.min(data)
    cropped = data.copy()
    ct_size = cropped.shape

    # Crop upper bound.
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

@delegates(__spatial_crop_foreground)
def crop_foreground(
    data: LabelArray,
    *args, **kwargs) -> LabelArray:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_crop_foreground, data, *args, **kwargs)

def crop_landmarks(
    landmark_data: Union[LandmarksFrame, LandmarksFrameVox],
    crop: Union[Box3D, BoxMM3D]) -> Union[LandmarksFrame, LandmarksFrameVox]:
    landmark_data = landmark_data.copy()
    lm_data = landmarks_to_data(landmark_data)
    lm_data = lm_data - crop[0]
    landmark_data[list(range(3))] = lm_data
    return landmark_data

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

