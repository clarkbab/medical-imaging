import numpy as np
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from SimpleITK import GetArrayViewFromImage as ArrayView
from typing import Union

from mymi import types

def percentile_hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    p: float) -> float:
    """
    returns: the Hausdorff distance between a and b.
    args:
        a: a 3D boolean array.
        b: another 3D boolean array.
        spacing: the voxel spacing used.
        p: the percentile.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'hausdorff_distance_95' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'hausdorff_distance_95' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    # Convert types for SimpleITK.
    a = a.astype('uint8')
    b = b.astype('uint8')

    # Convert to SimpleITK images.
    a = sitk.GetImageFromArray(a)
    a.SetSpacing(spacing)
    b = sitk.GetImageFromArray(b)
    b.SetSpacing(spacing)

    # Get surfaces.
    a_surface = sitk.LabelContour(a, False)
    b_surface = sitk.LabelContour(b, False)

    # Get minimum distances.
    a_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(a, squaredDistance=False, useImageSpacing=True))
    b_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(b, squaredDistance=False, useImageSpacing=True))
    a_to_b = ArrayView(b_dist_map)[ArrayView(a_surface) == 1]
    b_to_a = ArrayView(a_dist_map)[ArrayView(b_surface) == 1]

    # Calculate mean value.
    mean = (np.percentile(a_to_b, p) + np.percentile(b_to_a, p)) / 2
    return mean

def hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float]:
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
    a = a.astype('uint8')
    b = b.astype('uint8')

    # Convert to SimpleITK images.
    a = sitk.GetImageFromArray(a)
    a.SetSpacing(spacing)
    b = sitk.GetImageFromArray(b)
    b.SetSpacing(spacing)

    # Calculate Hausdorff distance.
    filter = sitk.HausdorffDistanceImageFilter()
    filter.Execute(a, b)
    max = filter.GetHausdorffDistance()
    mean = filter.GetAverageHausdorffDistance()
    return max, mean

def surface_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float, float, float]:

    # Get symmetric surface distances.
    a_surface = sitk.LabelContour(a)
    b_surface = sitk.LabelContour(b)
    a_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(a, squaredDistance=False, useImageSpacing=True))
    b_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(b, squaredDistance=False, useImageSpacing=True))
    # get a distance map on the surface
    a_to_b_dist_map = b_dist_map * sitk.Cast(a_surface, sitk.sitkFloat32)
    b_to_a_dist_map = a_dist_map * sitk.Cast(b_surface, sitk.sitkFloat32)
    # get number of surface voxels
    stat_filter = sitk.StatisticsImageFilter()
    stat_filter.Execute(a_surface)
    n_a_surface_voxels = int(stat_filter.GetSum())
    stat_filter.Execute(b_surface)
    n_b_surface_voxels = int(stat_filter.GetSum())
    # get all non-zero distances, and zero padd
    a_to_b_dist_map_array = sitk.GetArrayFromImage(a_to_b_dist_map)
    a_to_b_dists = list(a_to_b_dist_map_array[a_to_b_dist_map_array != 0])
    a_to_b_dists = a_to_b_dists + \
                        list(np.zeros(n_a_surface_voxels - len(a_to_b_dists)))
    b_to_a_distmap_array = sitk.GetArrayFromImage(b_to_a_distmap)
    b_to_a_dists = list(b_to_a_distmap_array[b_to_a_distmap_array != 0])
    b_to_a_dists = b_to_a_dists + \
                        list(np.zeros(n_b_surface_voxels - len(b_to_a_dists)))

    # Calculate statistics.
    dists = a_to_b_dists + b_to_a_dists
    mean_dist = np.mean(dists)
    median_dist = np.median(dists)
    std_dist = np.std(dists)
    max_dist = np.max(dists)

    return mean_dist, median_dist, std_dist, max_dist

def batch_mean_hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float]:
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

    maxs, means = [], []
    for i in range(len(a)):
        max, mean = hausdorff_distance(a[i], b[i], spacing)
        maxs.append(max)
        means.append(mean)
    return np.mean(maxs), np.mean(means)
