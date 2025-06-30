import numpy as np
from typing import *

from mymi.geometry import get_extent
from mymi.typing import *
from mymi.utils import *

from .shared import *

def __spatial_crop_or_pad(
    image: Image,
    bounding_box: Union[Point2DBox, Point3DBox],
    fill: Union[float, Literal['min']] = 'min',
    offset: Optional[Union[Point2D, Point3D]] = None,
    return_inverse: bool = False,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True) -> Image:
    bounding_box = replace_box_none(bounding_box, image.shape, spacing=spacing, offset=offset, use_patient_coords=use_patient_coords)
    assert_box_width(bounding_box)
    fill = np.min(image) if fill == 'min' else fill

    # Convert box to image coordinates.
    min, max = bounding_box
    if use_patient_coords:
        min = tuple(np.round((np.array(min) - offset) / spacing).astype(int))
        max = tuple(np.round((np.array(max) - offset) / spacing).astype(int))

    # Calculate inverse bounding box.
    n_dims = len(image.shape)
    spatial_size = image.shape[1:] if n_dims == 4 else image.shape
    if return_inverse:
        # Calculate the inverse bounding box.
        if use_patient_coords:
            inv_box = get_extent(image, spacing=spacing, offset=offset)
        else:
            inv_min = tuple(-np.array(min))
            new_spatial_size = np.array(max) - min
            inv_max = tuple(spatial_size + new_spatial_size - max)
            inv_box = (inv_min, inv_max)

    # Perform padding.
    pad_min = (-np.array(min)).clip(0)
    pad_max = (np.array(max) - spatial_size).clip(0)
    padding = list(zip(pad_min, pad_max))
    if n_dims == 4:
        padding = [0, 0] + padding
    padding = tuple(padding)
    image = np.pad(image, padding, constant_values=fill)

    # Perform cropping.
    crop_min = np.array(min).clip(0)
    padded_size = image.shape[1:] if n_dims == 4 else image.shape
    crop_max = (np.array(spatial_size) - max).clip(0)
    slices = list(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, padded_size))
    if n_dims == 4:
        slices = [slice(None)] + slices
    slices = tuple(slices)
    image = image[slices]

    if return_inverse:
        return image, inv_box
    else:
        return image

@delegates(__spatial_crop_or_pad)
def crop_or_pad(
    image: Image,
    *args, **kwargs) -> Image:
    assert_image(image)
    return handle_non_spatial_dims(__spatial_crop_or_pad, image, *args, **kwargs)

@delegates(__spatial_crop_or_pad)
def __spatial_centre_crop_or_pad(
    image: Image,
    size: Union[Size2D, Size3D, FOV2D, FOV3D],
    offset: Optional[Union[Point2D, Point3D]] = None,
    return_inverse: bool = False,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True,
    **kwargs) -> Image:

    # Determine cropping/padding amounts.
    n_dims = len(image.shape)
    spatial_size = image.shape[1:] if n_dims == 4 else image.shape
    if use_patient_coords:
        assert offset is not None
        assert spacing is not None
        fov = np.array(spatial_size) * spacing
        to_crop = fov - size
        box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int) + offset
        box_max = box_min + fov
    else:
        to_crop = np.array(spatial_size) - size
        box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
        box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    okwargs = dict(
        return_inverse=return_inverse,
        use_patient_coords=use_patient_coords,
    )
    res = __spatial_crop_or_pad(image, bounding_box, **okwargs, **kwargs)

    if return_inverse:
        # Convert inverse box to size/fov.
        inv_box_min, inv_box_max = res[1]
        inv_size = tuple(np.array(inv_box_max) - inv_box_min)
        return res[0], inv_size
    else:
        return res

@delegates(__spatial_centre_crop_or_pad)
def centre_crop_or_pad(
    image: Image,
    *args, **kwargs) -> Image:
    assert_image(image)
    return handle_non_spatial_dims(__spatial_centre_crop_or_pad, image, *args, **kwargs)

def crop_or_pad_landmarks(
    landmarks: LandmarkData,    # Should use patient coords, landmarks in image coords are only used for plotting.
    bounding_box: Union[Point3DBox, VoxelBox],
    offset: Optional[Point3D] = None,
    spacing: Optional[Spacing3D] = None,
    use_patient_coords: bool = True) -> LandmarkData:
    landmarks = landmarks.copy()
    min, max = bounding_box
            
    # Filter landmarks outside of image FOV.
    if use_patient_coords:
        fov_min, fov_max = min, max
    else:
        assert spacing is not None
        assert offset is not None
        fov_min, fov_max = np.array(min) * spacing + offset, np.array(max) * spacing + offset
    for a in range(3):
        landmarks = landmarks[(landmarks[a] >= fov_min[a]) & (landmarks[a] < fov_max[a])]
        
    # Shift landmarks.
    for a in range(3):
        landmarks[a] = landmarks[a] - fov_min[a]
        
    return landmarks
