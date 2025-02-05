import itk
import numpy as np
import os
from typing import Literal, Tuple, Union

from mymi.typing import Image, PointMM3D, ImageSpacing3D, ImageSize3D
from mymi.utils import from_itk, itk_convert_LPS_and_RAS, to_itk

def itk_load_transform(
    filepath: str) -> itk.Transform:
    if not os.path.exists(filepath):
        raise ValueError(f"ITK transform not found at filepath: {filepath}.")
    transform = itk.transformread(filepath)
    return transform

def itk_transform_image(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D,
    output_size: ImageSize3D,
    output_spacing: ImageSpacing3D, 
    output_offset: PointMM3D,
    transform: itk.Transform,
    fill: Union[float, Literal['min']] = 'min') -> Tuple[Image, ImageSpacing3D, PointMM3D]:
    # Load moving image.
    moving_itk = to_itk(data, spacing, offset)

    # Convert output params to LPS coordinates - transform will use this coordinate system.
    output_direction = np.eye(3)
    output_direction, output_offset = itk_convert_LPS_and_RAS(direction=output_direction, offset=output_offset)
    output_direction = itk.matrix_from_array(output_direction)

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
        output_direction=output_direction,
        output_origin=output_offset,
        output_spacing=output_spacing,
        size=output_size,
        transform=transform,
    )
    moved_itk = itk.resample_image_filter(moving_itk, **kwargs)
    moved, _, _ = from_itk(moved_itk)
    return moved

def itk_transform_points(
    fixed_points: np.ndarray,
    fixed_spacing: ImageSpacing3D,
    moving_spacing: ImageSpacing3D,
    fixed_offset: PointMM3D,
    moving_offset: PointMM3D,
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
