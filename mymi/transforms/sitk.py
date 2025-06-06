import numpy as np
import os
import SimpleITK as sitk
from typing import *

from mymi.typing import *
from mymi.utils import *

from .shared import handle_non_spatial_dims

def load_sitk_transform(
    filepath: str) -> sitk.Transform:
    if not os.path.exists(filepath):
        raise ValueError(f"SimpleITK transform not found at filepath: {filepath}.")

    # Convert nifti files to transforms.
    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        transform = dvf_to_sitk_transform(*load_nifti(filepath))
        return transform

    transform = sitk.ReadTransform(filepath)
    return transform

def create_sitk_affine_transform(
    offset: Optional[Point3D] = (0, 0, 0),
    output_offset: Optional[Point3D] = (0, 0, 0),
    output_spacing: Optional[Spacing3D] = (1, 1, 1),
    spacing: Optional[Spacing3D] = (1, 1, 1)) -> sitk.AffineTransform:

    # Create transform.
    transform = sitk.AffineTransform(3)
    transform.SetCenter(offset)   # Scaling should happen around this point.
    matrix = np.eye(3)
    for i in range(3):
        matrix[i, i] = spacing[i] / output_spacing[i]
    transform.SetMatrix(matrix.flatten())
    translation = tuple(np.array(output_offset) - offset)
    transform.SetTranslation(translation)

    return transform

def save_sitk_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteTransform(transform, filepath)

def __spatial_sitk_transform_image(
    data: np.ndarray,
    transform: sitk.Transform,
    output_size: Size3D,
    fill: Union[float, Literal['min']] = 'min',
    offset: Point3D = (0, 0, 0),
    output_offset: Point3D = (0, 0, 0),
    output_spacing: Spacing3D = (1, 1, 1), 
    spacing: Spacing3D = (1, 1, 1)) -> Tuple[Image, Spacing3D, Point3D]:
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

@delegates(__spatial_sitk_transform_image)
def sitk_transform_image(
    data: Image,
    *args, **kwargs) -> Image:
    raise ValueError("Use resample instead")
    assert_image(data)
    return handle_non_spatial_dims(__spatial_sitk_transform_image, data, *args, **kwargs)

def sitk_transform_points(
    points: Union[Points2D, Points3D, LandmarkData],
    transform: sitk.Transform) -> Union[Points2D, Points3D, LandmarkData]:
    is_landmarks = True if isinstance(points, LandmarkData) else False
    if is_landmarks:
        lms = points.copy()
        points = points[list(range(3))].to_numpy()
    assert points.shape[1] in (2, 3)
    points = points.astype(np.float64)  # sitk expects double.
    
    # Apply transform to points.
    points_t = []
    for p in points:
        p_t = transform.TransformPoint(p)
        points_t.append(p_t)
    points_t = np.vstack(points_t)

    if is_landmarks:
        lms[list(range(3))] = points_t
        output = lms
    else:
        output = points_t

    return output

def load_sitk_transform(
    filepath: str) -> sitk.Transform:
    transform = sitk.ReadTransform(filepath)

    # Cast to correct type.
    t = transform.GetTransformEnum()
    if t == sitk.sitkDisplacementField:
        transform = sitk.DisplacementFieldTransform(transform)
    elif t == sitk.sitkEuler:
        transform = sitk.Euler3DTransform(transform)

    return transform

def save_sitk_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    dirname = os.path.dirname(filepath)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    sitk.WriteTransform(transform, filepath)
