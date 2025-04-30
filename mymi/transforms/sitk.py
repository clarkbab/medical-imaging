import numpy as np
import os
import SimpleITK as sitk
from typing import *

from mymi import logging
from mymi.typing import *
from mymi.utils import *

def sitk_load_transform(
    filepath: str) -> sitk.Transform:
    if not os.path.exists(filepath):
        raise ValueError(f"SimpleITK transform not found at filepath: {filepath}.")

    # Convert nifti files to transforms.
    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        transform = to_sitk_transform(*load_nifti(filepath))
        return transform

    transform = sitk.ReadTransform(filepath)
    return transform

def sitk_save_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteTransform(transform, filepath)

def sitk_transform_image(
    data: np.ndarray,
    *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, sitk.Transform]]:
    n_dims = len(data.shape)
    if n_dims in (2, 3):
        output = sitk_transform_image_spatial(data, *args, **kwargs)
    elif n_dims == 4:
        size = data.shape
        if size[0] > size[-1]:
            logging.warning(f"Channels dimension should come first when transforming 4D. Got shape {size}, is this right?")
        os = []
        for d in data:
            d = sitk_transform_image_spatial(d, *args, **kwargs)
            os.append(d)
        output = np.stack(os, axis=0)
    else:
        raise ValueError(f"Data to resample should have (2, 3, 4) dims, got {n_dims}.")

    return output

def sitk_transform_image_spatial(
    data: np.ndarray,
    transform: sitk.Transform,
    output_size: ImageSize3D,
    fill: Union[float, Literal['min']] = 'min',
    offset: Point3D = (0, 0, 0),
    output_spacing: ImageSpacing3D = (1, 1, 1), 
    output_offset: Point3D = (0, 0, 0),
    spacing: ImageSpacing3D = (1, 1, 1)) -> Tuple[Image, ImageSpacing3D, Point3D]:
    # Load moving image.
    moving_sitk = to_sitk_image(data, spacing, offset)

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
        outputOrigin=output_offset,
        outputSpacing=output_spacing,
        size=output_size,
        transform=transform,
    )
    moved_sitk = sitk.Resample(moving_sitk, **kwargs)
    moved, _, _ = from_sitk_image(moved_sitk)
    return moved

def sitk_transform_points(
    points: np.ndarray,     # N x 3
    transform: sitk.Transform) -> np.ndarray:
    if isinstance(points, pd.DataFrame):
        points = points.to_numpy()  # Iterating over 'points' produces indices when using dataframes.
    assert points.shape[1] == 3
    points = points.astype(np.float64)  # sitk expects double.
    
    # Apply transform to points.
    points_t = []
    for p in points:
        p_t = transform.TransformPoint(p)
        points_t.append(p_t)
    points_t = np.vstack(points_t)

    return points_t
