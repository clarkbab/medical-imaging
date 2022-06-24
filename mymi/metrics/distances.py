import numpy as np
import SimpleITK as sitk
from typing import Dict, Tuple

from mymi.geometry import get_extent, get_extent_centre
from mymi import types

def distances(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Dict[str, float]:
    """
    args:
        a: a 3D boolean array.
        b: another 3D boolean array.
        spacing: the voxel spacing used.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'distances' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'distances' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Convert types for SimpleITK.
    a_itk = a.astype('uint8')
    b_itk = b.astype('uint8')
    spacing = tuple(reversed(spacing))

    # Convert to SimpleITK images.
    a_itk = sitk.GetImageFromArray(a_itk)
    a_itk.SetSpacing(spacing)
    b_itk = sitk.GetImageFromArray(b_itk)
    b_itk.SetSpacing(spacing)

    # Get surface voxels.
    a_surface = sitk.GetArrayFromImage(sitk.LabelContour(a_itk, False))
    b_surface = sitk.GetArrayFromImage(sitk.LabelContour(b_itk, False))

    # Compute distance maps.
    a_dist_map = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(a_itk, useImageSpacing=True, squaredDistance=False, insideIsPositive=False))
    b_dist_map = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(b_itk, useImageSpacing=True, squaredDistance=False, insideIsPositive=False))

    # Get voxel/surface min distances.
    a_to_b_surface_min_dists = b_dist_map[a_surface == 1]
    b_to_a_surface_min_dists = a_dist_map[b_surface == 1]
    a_to_b_voxel_min_dists = b_dist_map[a == 1]
    b_to_a_voxel_min_dists = a_dist_map[b == 1]

    # Voxel - set negative distances to zero as these indicate overlapping voxels.
    # Set negative distances to zero as these indicate overlapping voxels (hence "min dist" to other set is zero).
    a_to_b_surface_min_dists[a_to_b_surface_min_dists < 0] = 0
    b_to_a_surface_min_dists[b_to_a_surface_min_dists < 0] = 0
    a_to_b_voxel_min_dists[a_to_b_voxel_min_dists < 0] = 0
    b_to_a_voxel_min_dists[b_to_a_voxel_min_dists < 0] = 0

    # Calculate statistics.
    assd = np.mean(np.concatenate((a_to_b_surface_min_dists, b_to_a_surface_min_dists)))
    surface_hd = np.max(np.concatenate((a_to_b_surface_min_dists, b_to_a_surface_min_dists)))
    surface_hd_mean = np.mean([np.mean(a_to_b_surface_min_dists), np.mean(b_to_a_surface_min_dists)])
    surface_95hd = np.mean([np.percentile(a_to_b_surface_min_dists, 95), np.percentile(b_to_a_surface_min_dists, 95)])
    voxel_hd = np.max(np.concatenate((a_to_b_voxel_min_dists, b_to_a_voxel_min_dists)))
    voxel_hd_mean = np.mean([np.mean(a_to_b_voxel_min_dists), np.mean(b_to_a_voxel_min_dists)])
    voxel_95hd = np.mean([np.percentile(a_to_b_voxel_min_dists, 95), np.percentile(b_to_a_voxel_min_dists, 95)])
     
    return {
        'assd': assd,
        'surface-hd': surface_hd,
        'surface-hd-95': surface_95hd,
        'surface-hd-mean': surface_hd_mean,
        'voxel-hd': voxel_hd,
        'voxel-hd-95': voxel_95hd,
        'voxel-hd-mean': voxel_hd_mean
    }

def batch_mean_distances(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Dict[str, float]:
    """
    returns: the mean batch distance metrics.
    args:
        a: a boolean 4D array.
        b: another boolean 4D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'batch_mean_distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'batch_mean_distances' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")

    dists = {
        'assd': [],
        'surface-hd': [],
        # 'surface-ahd': [],
        'surface-95hd': [],
        'voxel-hd': [],
        # 'voxel-ahd': [],
        'voxel-95hd': []
    }

    for a, b in zip(a, b):
        d = distances(a, b, spacing)
        for metric in dists.keys():
            dists[metric].append(d[metric])
    mean_dists = dict([(metric, np.mean(values)) for metric, values in dists.items()])
    return mean_dists

def extent_centre_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float, float]:
    """
    returns: the maximum distance between extent centres for each axis.
    args:
        a: a boolean 3D array.
        b: another boolean 3D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'extent_centre_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'extent_centre_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'extent_centre_distance' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extent centres.
    a_cent = get_extent_centre(a)
    b_cent = get_extent_centre(b)

    # Get distance between centres.
    dists = np.abs(np.array(b_cent) - np.array(a_cent))    
    dists_mm = spacing * dists
    return dists_mm

def extent_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: types.ImageSpacing3D) -> Tuple[float, float, float]:
    """
    returns: the maximum distance between extent boundaries for each axis.
    args:
        a: a boolean 3D array.
        b: another boolean 3D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'extent_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"Metric 'extent_distance' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'extent_distance' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extents.
    a_ext = get_extent(a)
    b_ext = get_extent(b)

    # Calculate distances.
    a = np.array(a_ext)
    a[1] = -a[1]
    b = np.array(b_ext)
    b[1] = -b[1]
    dists = np.max(a - b, axis=0)
    dists_mm = spacing * dists
    return dists_mm
