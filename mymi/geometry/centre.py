import numpy as np
from scipy.ndimage import center_of_mass
from typing import Optional, Union

from .extent import get_extent

from mymi.types import ImageSize2D, ImageSize3D, Point2D, Point3D

def get_centre(a: np.ndarray) -> Optional[Union[Point2D, Point3D]]:
    return get_centre_from_size(a.shape)

def get_centre_from_size(s: Union[ImageSize2D, ImageSize3D]) -> Optional[Union[Point2D, Point3D]]:
    return tuple([int(np.floor(si / 2)) - 1 for si in s])

def get_centre_of_mass(a: np.ndarray) -> Optional[Union[Point2D, Point3D]]:
    return center_of_mass(a)

def get_extent_centre(a: np.ndarray) -> Optional[Union[Point2D, Point3D]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'get_extent_centre' expected a boolean array, got '{a.dtype}'.")

    # Get extent.
    extent = get_extent(a)

    if extent:
        # Find the extent centre.
        centre = tuple(np.floor(np.array(extent).sum(axis=0) / 2).astype(int))
    else:
        return None

    return centre
