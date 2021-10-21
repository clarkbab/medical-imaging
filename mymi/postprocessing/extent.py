import numpy as np
from typing import Optional

from mymi import types

def get_extent(a: np.ndarray) -> Optional[types.Box3D]:
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

def get_extent_centre(a: np.ndarray) -> Optional[types.Point3D]:
    if a.dtype != np.bool:
        raise ValueError(f"'get_extent_centre' expected a boolean array, got '{a.dtype}'.")

    # Get extent.
    extent = get_extent(a)

    if extent:
        # Find the extent centre.
        centre = tuple(np.floor(np.array(box).sum(axis=0) / 2).astype(int))
    else:
        return None

    return centre
