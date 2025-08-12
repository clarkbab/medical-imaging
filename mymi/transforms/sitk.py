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

def sitk_save_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteTransform(transform, filepath)

def sitk_transform_points(
    points: Union[Points2D, Points3D, LandmarksData],
    transform: sitk.Transform) -> Union[Points2D, Points3D, LandmarksData]:
    is_landmarks = True if isinstance(points, LandmarksData) else False
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

def sitk_load_transform(
    filepath: str) -> sitk.Transform:
    transform = sitk.ReadTransform(filepath)

    # Cast to correct type.
    t = transform.GetTransformEnum()
    if t == sitk.sitkDisplacementField:
        transform = sitk.DisplacementFieldTransform(transform)
    elif t == sitk.sitkEuler:
        transform = sitk.Euler3DTransform(transform)

    return transform

def sitk_save_transform(
    transform: sitk.Transform,
    filepath: str) -> None:
    dirname = os.path.dirname(filepath)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    sitk.WriteTransform(transform, filepath)
