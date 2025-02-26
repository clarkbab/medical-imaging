import numpy as np
from surface_distance import *
from typing import *

from mymi.geometry import extent, centre_of_extent
from mymi.typing import *
from mymi.utils import arg_to_list

def distances(
    a: np.ndarray,
    b: np.ndarray,
    spacing: ImageSpacing3D,
    tol: Union[int, float, List[Union[int, float]]] = []) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'distances' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")
    tols = arg_to_list(tol, (int, float))

    a, b = a.astype(np.bool_), b.astype(np.bool_)
    surf_dists = compute_surface_distances(a, b, spacing) 
    metrics = {
        'hd': compute_robust_hausdorff(surf_dists, 100),
        'msd': np.mean(compute_average_surface_distance(surf_dists)),
        'hd-95': compute_robust_hausdorff(surf_dists, 95)
    }
    for tol in tols:
        metrics[f'surface-dice-tol-{tol}'] = compute_surface_dice_at_tolerance(surf_dists, tol)

    return metrics

def apl(
    surf_dists: Tuple[np.ndarray, np.array],
    spacing: ImageSpacing3D,
    tol: float,
    unit: Literal['mm', 'voxels'] = 'mm') -> float:
    b_to_a_surf_dists = surf_dists[1]   # Only look at dists from 'GT' back to 'pred'. 
    b_to_a_non_overlap = b_to_a_surf_dists[b_to_a_surf_dists > tol]
    assert spacing[0] == spacing[1], f"In-plane spacing should be equal when calculating APL, got '{spacing}'."
    if unit == 'mm':
        return len(b_to_a_non_overlap) * spacing[0]
    else:
        return len(b_to_a_non_overlap)

def extent_centre_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: ImageSpacing3D) -> Tuple[float, float, float]:
    """
    returns: the maximum distance between extent centres for each axis.
    args:
        a: a boolean 3D array.
        b: another boolean 3D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'extent_centre_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'extent_centre_distance' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extent centres.
    a_cent = centre_of_extent(a)
    b_cent = centre_of_extent(b)

    # Get distance between centres.
    dists = np.abs(np.array(b_cent) - np.array(a_cent))    
    dists_mm = spacing * dists
    return dists_mm

def extent_distance(
    a: np.ndarray,
    b: np.ndarray,
    spacing: ImageSpacing3D) -> Tuple[float, float, float]:
    """
    returns: the maximum distance between extent boundaries for each axis.
    args:
        a: a boolean 3D array.
        b: another boolean 3D array.
        spacing: the voxel spacing.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'extent_distance' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'extent_distance' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extents.
    a_ext = extent(a)
    b_ext = extent(b)

    # Calculate distances.
    a = np.array(a_ext)
    a[1] = -a[1]
    b = np.array(b_ext)
    b[1] = -b[1]
    dists = np.max(a - b, axis=0)
    dists_mm = spacing * dists
    return dists_mm

def get_encaps_dist_vox(
    a: np.ndarray,
    b: np.ndarray) -> Tuple[int, int, int]:
    """
    returns: an asymmetric distance measuring the encapsulation of b by a along each axis.
        A negative distance implies encapsulation.
    """
    if a.shape != b.shape:
        raise ValueError(f"'get_encaps_dist_vox' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"'get_encaps_dist_vox' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extents.
    a_ext = extent(a)
    b_ext = extent(b)

    # Calculate distances.
    a = np.array(a_ext)
    a[1] = -a[1]
    b = np.array(b_ext)
    b[1] = -b[1]
    dist = np.max(a - b, axis=0)
    return dist

def get_encaps_dist_mm(
    a: np.ndarray,
    b: np.ndarray,
    spacing: ImageSpacing3D) -> Tuple[int, int, int]:
    """
    returns: an asymmetric distance measuring the encapsulation of b by a along each axis.
        A negative distance implies encapsulation.
    """
    if a.shape != b.shape:
        raise ValueError(f"'get_encaps_dist_mm' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"'get_encaps_dist_mm' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    dist = get_encaps_dist_vox(a, b)
    dist_mm = tuple(np.array(spacing) * dist)
    return dist_mm
