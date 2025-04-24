import numpy as np
from typing import *

from mymi.typing import *

def extent(a: np.ndarray) -> Optional[Union[Box2D, Box3D]]:
    # Get OAR extent.
    if a.sum() > 0:
        non_zero = np.argwhere(a != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)
    else:
        box = None

    return box

def extent_mm(
    a: np.ndarray,
    spacing: ImageSpacing3D,
    offset: Voxel) -> Optional[Union[Box2D, Box3D]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'extent_mm' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent.
    if a.sum() > 0:
        non_zero = np.argwhere(a != 0).astype(int)
        min = non_zero.min(axis=0)
        max = non_zero.max(axis=0)
        min_mm = min * spacing + offset
        max_mm = max * spacing + offset
        box_mm = (min_mm, max_mm)
    else:
        box_mm = None

    return box_mm

def extent_edge_voxel(
    a: np.ndarray,
    axis: Axis,
    end: Literal['min', 'max'],
    view_axis: Axis) -> Voxel:
    if a.dtype != np.bool_:
        raise ValueError(f"'extent' expected a boolean array, got '{a.dtype}'.")
    assert end in ('min', 'max')

    # Find extreme voxel.
    # Returns a foreground voxel on the extent of the OAR along given 'axis' and 'end' of the axis.
    # There could be multiple extreme voxels at this end of the OAR, so we look at another 'view' axis
    # and return the central extreme voxel along this axis.
    non_zero = np.argwhere(a)
    if end == 'min':
        axis_value = non_zero[:, axis].min()
    elif end == 'max':
        axis_value = non_zero[:, axis].max()
    axis_voxels = non_zero[non_zero[:, axis] == axis_value]
    axis_voxels = axis_voxels[np.argsort(axis_voxels[:, view_axis])]
    max_voxel = tuple(axis_voxels[len(axis_voxels) // 2])
    return max_voxel

def extent_width(a: np.ndarray) -> Optional[Union[ImageSize2D, ImageSize3D]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'extent_width' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent.
    ext = extent(a)
    if ext:
        min, max = ext
        width = tuple(np.array(max) - min)
        return width
    else:
        return None

def extent_width_mm(
    a: np.ndarray,
    spacing: Tuple[float, float, float]) -> Optional[Union[Tuple[float, float], Tuple[float, float, float]]]:
    if a.dtype != np.bool_:
        raise ValueError(f"'extent_width_mm' expected a boolean array, got '{a.dtype}'.")

    # Get OAR extent in mm.
    ext_width_vox = extent_width(a)
    if ext_width_vox is None:
        return None
    ext_width = tuple(np.array(ext_width_vox) * spacing)
    return ext_width

def centre_of_extent(
    a: np.ndarray,
    smoothed_label: bool = False) -> Optional[Union[Pixel, Voxel]]:
    if smoothed_label:
        a = np.round(a)

    # Get extent.
    ext = extent(a)

    if ext:
        # Find the extent centre.
        centre = tuple(np.floor(np.array(ext).sum(axis=0) / 2).astype(int))
    else:
        return None

    return centre
