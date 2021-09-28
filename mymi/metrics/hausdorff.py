import numpy as np
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from typing import Tuple, Union

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
    a_img = sitk.GetImageFromArray(a)
    a_img.SetSpacing(spacing)
    b_img = sitk.GetImageFromArray(b)
    b_img.SetSpacing(spacing)

    # Get surfaces.
    a_surface = sitk.LabelContour(a_img, False)
    a_surface_arr = sitk.GetArrayFromImage(a_surface)
    b_surface = sitk.LabelContour(b_img, False)
    b_surface_arr = sitk.GetArrayFromImage(b_surface)

    # Get minimum distances from each voxel to surface.
    a_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(a_img, squaredDistance=False, useImageSpacing=True))
    a_dist_map_arr = sitk.GetArrayFromImage(a_dist_map)
    b_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(b_img, squaredDistance=False, useImageSpacing=True))
    b_dist_map_arr = sitk.GetArrayFromImage(b_dist_map)

    # Get distances between sets.
    a_to_b = list(b_dist_map_arr[a_surface_arr == 1])
    b_to_a = list(a_dist_map_arr[b_surface_arr == 1])

    # Calculate mean value.
    p_hd = np.max((np.percentile(a_to_b, p), np.percentile(b_to_a, p)))
    return p_hd

def hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float]:
    """
    returns: the Hausdorff Distance (HD) and Average Hausdorff Distance (AHD) between a and b.
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
    hd = filter.GetHausdorffDistance()
    avg_hd = filter.GetAverageHausdorffDistance()
    return hd, avg_hd

def symmetric_surface_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D,
    p: int) -> Tuple[float, float, float, float]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'surface_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'surface_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    # Convert types for SimpleITK.
    a = a.astype('uint8')
    b = b.astype('uint8')

    # Convert to SimpleITK images.
    a_img = sitk.GetImageFromArray(a)
    a_img.SetSpacing(spacing)
    b_img = sitk.GetImageFromArray(b)
    b_img.SetSpacing(spacing)

    # Get surfaces.
    a_surface = sitk.LabelContour(a_img, False)
    a_surface_arr = sitk.GetArrayFromImage(a_surface)
    b_surface = sitk.LabelContour(b_img, False)
    b_surface_arr = sitk.GetArrayFromImage(b_surface)

    # Get minimum distances from each voxel to surface.
    a_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(a_img, squaredDistance=False, useImageSpacing=True))
    a_dist_map_arr = sitk.GetArrayFromImage(a_dist_map)
    b_dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(b_img, squaredDistance=False, useImageSpacing=True))
    b_dist_map_arr = sitk.GetArrayFromImage(b_dist_map)

    # Get distances between sets.
    a_to_b = b_dist_map_arr[a_surface_arr == 1]
    b_to_a = a_dist_map_arr[b_surface_arr == 1]

    # Count the number of surface voxels.
    stat_filter = sitk.StatisticsImageFilter()
    stat_filter.Execute(a_surface)
    n_a_surface_voxels = int(stat_filter.GetSum())
    stat_filter.Execute(b_surface)
    n_b_surface_voxels = int(stat_filter.GetSum())

    # Gather surface distances, including zero-value distances as these affect the statistics.
    a_to_b_dist_map_array = sitk.GetArrayFromImage(a_to_b_dist_map)
    a_to_b_dists = list(a_to_b_dist_map_array[a_to_b_dist_map_array != 0])
    a_to_b_dists += list(np.zeros(n_a_surface_voxels - len(a_to_b_dists)))
    b_to_a_distmap_array = sitk.GetArrayFromImage(b_to_a_distmap)
    b_to_a_dists = list(b_to_a_distmap_array[b_to_a_distmap_array != 0])
    b_to_a_dists += list(np.zeros(n_b_surface_voxels - len(b_to_a_dists)))

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
    returns: the batch mean Hausdorff and average Hausdorff distance between a and b.
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

def batch_mean_symmetric_surface_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float, float, float]:
    """
    returns: the batch mean surface distance statistics between arrays a and b.
    args:
        a: a 4D boolean array.
        b: another 4D boolean array.
        spacing: the voxel spacing used.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'batch_mean_surface_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'batch_mean_surface_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    means, medians, stds, maxs = [], [], [], []
    for i in range(len(a)):
        mean, median, std, max = surface_distance(a[i], b[i], spacing)
        means.append(mean)
        medians.append(median)
        stds.append(std)
        maxs.append(max)
    return np.mean(means), np.mean(medians), np.mean(stds), np.mean(maxs)
