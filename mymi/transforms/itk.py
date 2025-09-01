import itk
import numpy as np
import os
from typing import *

from mymi.typing import *
from mymi.utils import *

def load_itk_transform(
    filepath: str) -> itk.Transform:
    if not os.path.exists(filepath):
        raise ValueError(f"ITK transform not found at filepath: {filepath}.")
    transform = itk.transformread(filepath)
    return transform

def itk_transform_image(
    data: ImageArray,
    transform: itk.Transform,
    output_size: Size3D,
    fill: Union[float, Literal['min']] = 'min',
    offset: Point3D = (0, 0, 0),
    output_offset: Point3D = (0, 0, 0),
    output_spacing: Spacing3D = (1, 1, 1), 
    spacing: Spacing3D = (1, 1, 1),
    reverse_xy: bool = False) -> Tuple[ImageArray, Spacing3D, Point3D]:
    # Load moving image.
    moving_itk = to_itk_image(data, spacing, offset)

    # Get interpolation method.
    if data.dtype == bool:
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving_itk)
    else:
        interpolator = itk.LinearInterpolateImageFunction.New(moving_itk)

    # Determine default value.
    if fill == 'min':
        default_value = float(data.min())
        if data.dtype == bool:
            default_value = int(default_value)
    else:
        default_value = fill

    # Apply transform.
    kwargs = dict(
        default_pixel_value=default_value,
        interpolator=interpolator,
        output_origin=output_offset,
        output_spacing=output_spacing,
        size=output_size,
        transform=transform,
    )
    moved_itk = itk.resample_image_filter(moving_itk, **kwargs)
    moved, _, _ = from_itk_image(moved_itk)
    return moved

def itk_transform_points(
    fixed_points: np.ndarray,
    fixed_spacing: Spacing3D,
    moving_spacing: Spacing3D,
    fixed_offset: Point3D,
    moving_offset: Point3D,
    transform: itk.Transform,
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

def itk_save_transform(
    transform: itk.Transform,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    itk.WriteTransform(transform, filepath)
