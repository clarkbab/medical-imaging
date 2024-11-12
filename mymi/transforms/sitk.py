import numpy as np
import SimpleITK as sitk
from typing import Union

from mymi.types import ImageOffset3D, ImageSpacing3D
from mymi.utils import from_sitk_image, to_sitk_image

def sitk_point_transform(
    fixed_points: np.ndarray,
    fixed_spacing: ImageSpacing3D,
    moving_spacing: ImageSpacing3D,
    fixed_offset: ImageOffset3D,
    moving_offset: ImageOffset3D,
    transform: sitk.Transform,
    fill: Union[float, str] = 'min') -> np.ndarray:
    
    # Apply transform to points.
    moved_points = []
    for f in fixed_points:
        # Convert from voxel to physical coordinates.
        f_mm = fixed_offset + f * fixed_spacing

        # Transform point.
        # When we convert from numpy to sitk image, we correct the order of the axes
        # so we don't need to reverse order here.
        f_t_mm = transform.TransformPoint(f_mm)

        # Convert back to voxel coordinates.
        f_t = (np.array(f_t_mm) - moving_offset) / moving_spacing
        moved_points.append(f_t)
    moved_points = np.vstack(moved_points)

    return moved_points

def sitk_image_transform(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed_spacing: ImageSpacing3D,
    moving_spacing: ImageSpacing3D,
    fixed_offset: ImageOffset3D,
    moving_offset: ImageOffset3D,
    transform: sitk.Transform,
    fill: Union[float, str] = 'min') -> np.ndarray:
    fixed_image_sitk = to_sitk_image(fixed_image, fixed_spacing, fixed_offset)
    moving_image_sitk = to_sitk_image(moving_image, moving_spacing, moving_offset)

    # Get value for filling new voxels.
    if isinstance(fill, str):
        if fill == 'min':
            fill = float(moving_image.min())
        elif fill == 'max':
            fill = float(moving_image.max())
        else:
            raise ValueError(f"Unknown fill value '{fill}'.")
    else:
        fill = float(fill)

    # Apply transform (resample).
    resample = sitk.ResampleImageFilter()
    resample.SetDefaultPixelValue(fill)
    if moving_image.dtype == bool:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetReferenceImage(fixed_image_sitk)
    resample.SetTransform(transform)
    moved_sitk = resample.Execute(moving_image_sitk)

    # Convert back to numpy.
    moved, _, _ = from_sitk_image(moved_sitk)

    # Preserve original numpy datatypes.
    if moving_image.dtype == bool:
        moved = moved.astype(bool)

    return moved
