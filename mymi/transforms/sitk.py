import numpy as np
import os
import SimpleITK as sitk
from typing import Literal, Tuple, Union

from mymi.types import Image, PointMM3D, ImageSpacing3D, ImageSize3D
from mymi.utils import from_sitk, sitk_convert_LPS_and_RAS, to_sitk

def sitk_load_transform(
    filepath: str) -> sitk.Transform:
    if not os.path.exists(filepath):
        raise ValueError(f"SimpleITK transform not found at filepath: {filepath}.")
    transform = sitk.ReadTransform(filepath)
    return transform

def sitk_save_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    sitk.WriteTransform(transform, filepath)

def sitk_transform_image(
    data: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D,
    output_size: ImageSize3D,
    output_spacing: ImageSpacing3D, 
    output_offset: PointMM3D,
    transform: sitk.Transform,
    fill: Union[float, Literal['min']] = 'min') -> Tuple[Image, ImageSpacing3D, PointMM3D]:
    # Load moving image.
    moving_sitk = to_sitk(data, spacing, offset)

    # Convert output params to LPS coordinates - transform will use this coordinate system.
    output_direction = np.eye(3).flatten()
    output_direction, output_offset = sitk_convert_LPS_and_RAS(direction=output_direction, offset=output_offset)

    # Get interpolation method.
    if data.dtype == bool:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    # Determine default value.
    if fill == 'min':
        default_value = float(data.min())
        if data.dtype == bool:
            default_value = int(default_value)
    else:
        default_value = fill

    # Apply transform.
    kwargs = dict(
        defaultPixelValue=default_value,
        interpolator=interpolator,
        outputDirection=output_direction,
        outputOrigin=output_offset,
        outputSpacing=output_spacing,
        size=output_size,
        transform=transform,
    )
    moved_sitk = sitk.Resample(moving_sitk, **kwargs)
    moved, _, _ = from_sitk(moved_sitk)
    return moved

def sitk_transform_points(
    points: np.ndarray,
    transform: sitk.Transform) -> np.ndarray:
    
    # Apply transform to points.
    points_t = []
    for p_mm in points:
        # Transform point.
        # When we convert from numpy to sitk image, we correct the order of the axes
        # so we don't need to reverse order here.
        p_t_mm = transform.TransformPoint(p_mm)
        points_t.append(p_t_mm)
    points_t = np.vstack(points_t)

    return points_t

def sitk_transform_voxels(
    points: np.ndarray,
    spacing: ImageSpacing3D,
    offset: PointMM3D,
    output_offset: PointMM3D,
    output_spacing: ImageSpacing3D,
    transform: sitk.Transform,
    fill: Union[float, str] = 'min') -> np.ndarray:
    
    # Apply transform to points.
    points_t = []
    for p in points:
        # Convert from voxel to physical coordinates.
        p_mm = offset + p * spacing

        # Transform point.
        # When we convert from numpy to sitk image, we correct the order of the axes
        # so we don't need to reverse order here.
        p_t_mm = transform.TransformPoint(p_mm)

        # Convert back to voxel coordinates.
        p_t = (np.array(p_t_mm) - output_offset) / output_spacing
        points_t.append(p_t)
    points_t = np.vstack(points_t)

    return points_t
