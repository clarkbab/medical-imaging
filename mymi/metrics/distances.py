import numpy as np
import SimpleITK as sitk
from typing import Dict

from mymi.postprocessing import get_extent_centre
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
    a_to_b_voxel_min_dists = b_dist_map[a == 1]
    a_to_b_surface_min_dists = b_dist_map[a_surface == 1]
    b_to_a_voxel_min_dists = a_dist_map[b == 1]
    b_to_a_surface_min_dists = a_dist_map[b_surface == 1]

    # Voxel - set negative distances to zero as these indicate overlapping voxels.
    a_to_b_voxel_min_dists[a_to_b_voxel_min_dists < 0] = 1
    b_to_a_voxel_min_dists[b_to_a_voxel_min_dists < 0] = 1

    # Surface - take absolute distances as we're only interested in distance to the nearest
    # surface voxel.
    a_to_b_surface_min_dists = np.abs(a_to_b_surface_min_dists)
    b_to_a_surface_min_dists = np.abs(b_to_a_surface_min_dists)

    # Calculate statistics.
    assd = np.mean(np.concatenate((a_to_b_surface_min_dists, b_to_a_surface_min_dists)))
    surface_hd = np.max(np.concatenate((a_to_b_surface_min_dists, b_to_a_surface_min_dists)))
    # surface_ahd = np.mean([np.mean(a_to_b_surface_min_dists), np.mean(b_to_a_surface_min_dists)])     # These values don't match SimpleITK closely.
    surface_95hd = np.max([np.percentile(a_to_b_surface_min_dists, 95), np.percentile(b_to_a_surface_min_dists, 95)])
    voxel_hd = np.max(np.concatenate((a_to_b_voxel_min_dists, b_to_a_voxel_min_dists)))
    # voxel_ahd = np.mean([np.mean(a_to_b_voxel_min_dists), np.mean(b_to_a_voxel_min_dists)])           # These values don't match SimpleITK closely.
    voxel_95hd = np.max([np.percentile(a_to_b_voxel_min_dists, 95), np.percentile(b_to_a_voxel_min_dists, 95)])
     
    return {
        'assd': assd,
        'surface-hd': surface_hd,
        # 'surface-ahd': surface_ahd,
        'surface-95hd': surface_95hd,
        'voxel-hd': voxel_hd,
        # 'voxel-ahd': voxel_ahd,
        'voxel-95hd': voxel_95hd
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
    spacing: types.ImageSpacing3D) -> float:
    """
    returns: the maximum distance between extent centres across all axes.
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

    # Take max across axes.
    max_dist_mm = dists_mm.max()
    return max_dist_mm
