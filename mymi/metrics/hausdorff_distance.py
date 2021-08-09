import numpy as np
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from typing import Union

from mymi import types

def hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> float:
    """
    returns: the Hausdorff distance between a and b.
    args:
        a: a 3D boolean array.
        b: another 3D boolean array.
        spacing: the voxel spacing used.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'hausdorff_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'hausdorff_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    # Convert types for SimpleITK.
    a = a.astype('float32')
    b = b.astype('float32')

    # Convert to SimpleITK images.
    a = sitk.GetImageFromArray(a)
    a.SetSpacing(spacing)
    b = sitk.GetImageFromArray(b)
    b.SetSpacing(spacing)

    # Calculate Hausdorff distance.
    filter = sitk.HausdorffDistanceImageFilter()
    filter.Execute(a, b)
    dist = filter.GetHausdorffDistance()
    return dist

def batch_mean_hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> float:
    """
    returns: the batch mean Hausdorff distance between a and b.
    args:
        a: a 4D boolean array.
        b: another 4D boolean array.
        spacing: the voxel spacing used.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'batch_mean_hausdorff_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'batch_mean_hausdorff_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    dists = []
    for i in range(len(a)):
        dists.append(hausdorff_distance(a[i], b[i], spacing))
    return np.mean(hd_dists)
