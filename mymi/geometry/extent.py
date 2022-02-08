import numpy as np
from typing import Optional, Tuple, Union

from mymi import types

def get_extent(a: np.ndarray) -> Optional[Union[types.Box2D, types.Box3D]]:
    if a.dtype != np.bool:
        raise ValueError(f"'get_extent' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent.
    if a.sum() > 0:
        non_zero = np.argwhere(a != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)
    else:
        box = None

    return box

def get_extent_centre(a: np.ndarray) -> Optional[Union[types.Point2D, types.Point3D]]:
    if a.dtype != np.bool:
        raise ValueError(f"'get_extent_centre' expected a boolean array, got '{a.dtype}'.")

    # Get extent.
    extent = get_extent(a)

    if extent:
        # Find the extent centre.
        centre = tuple(np.floor(np.array(extent).sum(axis=0) / 2).astype(int))
    else:
        return None

    return centre

def get_extent_width_vox(a: np.ndarray) -> Optional[Union[types.ImageSize2D, types.ImageSize3D]]:
    if a.dtype != np.bool:
        raise ValueError(f"'get_extent_width_vox' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent.
    extent = get_extent(a)
    if extent:
        min, max = extent
        width = tuple(np.array(max) - min)
        return width
    else:
        return None

def get_extent_width_mm(
    a: np.ndarray,
    spacing: Tuple[float, float, float]) -> Optional[Union[Tuple[float, float], Tuple[float, float, float]]]:
    if a.dtype != np.bool:
        raise ValueError(f"'get_extent_width_mm' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent in mm.
    ext_width_vox = get_extent_width_vox(a)
    if ext_width_vox is None:
        return None
    ext_width = tuple(np.array(ext_width_vox) * spacing)
    return ext_width

def get_encaps_dist_vox(
    a: np.ndarray,
    b: np.ndarray) -> Tuple[int, int, int]:
    """
    returns: an asymmetric distance measuring the encapsulation of b by a along each axis.
        A negative distance implies encapsulation.
    """
    if a.shape != b.shape:
        raise ValueError(f"'get_encaps_dist_vox' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"'get_encaps_dist_vox' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"'get_encaps_dist_vox' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    # Calculate extents.
    a_ext = get_extent(a)
    b_ext = get_extent(b)

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
    spacing: types.ImageSpacing3D) -> Tuple[int, int, int]:
    """
    returns: an asymmetric distance measuring the encapsulation of b by a along each axis.
        A negative distance implies encapsulation.
    """
    if a.shape != b.shape:
        raise ValueError(f"'get_encaps_dist_mm' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.dtype != np.bool or b.dtype != np.bool:
        raise ValueError(f"'get_encaps_dist_mm' expects boolean arrays. Got '{a.dtype}' and '{b.dtype}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"'get_encaps_dist_mm' can't be calculated on empty sets. Got cardinalities '{a.sum()}' and '{b.sum()}'.")

    dist = get_encaps_dist_vox(a, b)
    dist_mm = tuple(np.array(spacing) * dist)
    return dist_mm
