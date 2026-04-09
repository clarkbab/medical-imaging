from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, fov
import numpy as np
from typing import *

from mymi.typing import *
from mymi.utils.assertions import assert_image
from mymi.utils.decorators import handle_non_spatial_dims
from mymi.utils.python import delegates

from .transforms import assert_box_width, replace_box_none

def __spatial_crop_or_pad(
    image: ImageArray,
    bounding_box: Union[BoxMM2D, BoxMM3D],
    affine: AffineMatrix | None = None,
    fill: Union[float, Literal['min']] = 'min',
    return_inverse: bool = False,
    ) -> ImageArray:
    bounding_box = replace_box_none(bounding_box, image.shape, affine=affine)
    assert_box_width(bounding_box)
    fill = np.min(image) if fill == 'min' else fill

    # Convert box to image coordinates.
    min, max = bounding_box
    if affine is not None:
        origin = affine_origin(affine)
        spacing = affine_spacing(affine)
        min = tuple(np.round((np.array(min) - origin) / spacing).astype(int))
        max = tuple(np.round((np.array(max) - origin) / spacing).astype(int))

    # Calculate inverse bounding box.
    n_dims = len(image.shape)
    spatial_size = image.shape[1:] if n_dims == 4 else image.shape
    if return_inverse:
        # Calculate the inverse bounding box.
        inv_box = fov(image.shape, affine=affine) 

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
    image: ImageArray,
    *args, **kwargs) -> ImageArray:
    assert_image(image)
    return handle_non_spatial_dims(__spatial_crop_or_pad, image, *args, **kwargs)

@delegates(__spatial_crop_or_pad)
def __spatial_centre_crop_or_pad(
    image: ImageArray,
    size: Union[Size, SizeMM],
    origin: Optional[Point] = None,
    return_inverse: bool = False,
    spacing: Optional[Spacing] = None,
    use_world_coords: bool = True,
    **kwargs) -> ImageArray:

    # Determine cropping/padding amounts.
    n_dims = len(image.shape)
    spatial_size = image.shape[1:] if n_dims == 4 else image.shape
    if use_world_coords:
        assert origin is not None
        assert spacing is not None
        fov_max = np.array(spatial_size) * spacing
        to_crop = fov_max - size
        box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int) + origin
        box_max = box_min + fov_max
    else:
        to_crop = np.array(spatial_size) - size
        box_min = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
        box_max = box_min + size
    bounding_box = (box_min, box_max)

    # Perform crop or padding.
    okwargs = dict(
        return_inverse=return_inverse,
        use_world_coords=use_world_coords,
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
    image: ImageArray,
    *args, **kwargs) -> ImageArray:
    assert_image(image)
    return handle_non_spatial_dims(__spatial_centre_crop_or_pad, image, *args, **kwargs)

def crop_or_pad_landmarks(
    landmarks: LandmarksFrame,    # Should use patient coords, landmarks in image coords are only used for plotting.
    bounding_box: Union[BoxMM3D, Box3D],
    origin: Optional[Point3D] = None,
    spacing: Optional[Spacing3D] = None,
    use_world_coords: bool = True) -> LandmarksFrame:
    landmarks = landmarks.copy()
    min, max = bounding_box
            
    # Filter landmarks outside of image FOV.
    if use_world_coords:
        fov_min, fov_max = min, max
    else:
        assert spacing is not None
        assert origin is not None
        fov_min, fov_max = np.array(min) * spacing + origin, np.array(max) * spacing + origin
    for a in range(3):
        landmarks = landmarks[(landmarks[a] >= fov_min[a]) & (landmarks[a] < fov_max[a])]
        
    # Shift landmarks.
    for a in range(3):
        landmarks[a] = landmarks[a] - fov_min[a]
        
    return landmarks
