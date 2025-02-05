import numpy as np
import os
import SimpleITK as sitk
from typing import *

from mymi.typing import *
from mymi.utils import *

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
    moving_sitk = to_sitk_image(data, spacing, offset)

    # Our 'data/spacing/offset' params use RAS coordinates, convert to LPS as this is what
    # the sitk transform will use.
    output_direction = np.eye(3)
    output_direction = sitk_convert_matrix_RAS_LPS(output_direction).flatten()
    output_offset = sitk_convert_point_RAS_LPS(output_offset)

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
    points: np.ndarray,     # Patient-based coordinates (mm) in RAS system - our standard.
    transform: sitk.Transform) -> np.ndarray:
    
    # Apply transform to points.
    points_t = []
    for p in points:
        # Transform point. Requires conversion to/from LPS coordinates for sitk.
        p = sitk_convert_point_RAS_LPS(p)
        p_t = transform.TransformPoint(p)
        p_t = sitk_convert_point_RAS_LPS(p_t)
        points_t.append(p_t)
    points_t = np.vstack(points_t)

    return points_t
