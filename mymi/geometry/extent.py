import numpy as np
from typing import *

from mymi.typing import *

def get_extent(
    image: ImageData3D,
    offset: Optional[Point3D] = None,
    spacing: Optional[Spacing3D] = None,
    use_patient_coords: bool = True) -> Optional[Union[PixelBox, VoxelBox]]:
    if use_patient_coords:
        assert spacing is not None
        assert offset is not None

    # Get voxel extent.
    n_dims = len(image.shape)
    extent_vox = ((0,) * n_dims, image.shape)
    if not use_patient_coords:
        return extent_vox

    # Get mm extent.
    extent_min_vox, extent_max_vox = extent_vox
    extent_min_mm = tuple(float(e) for e in (np.array(extent_min_vox) * spacing + offset))
    extent_max_mm = tuple(float(e) for e in (np.array(extent_max_vox) * spacing + offset))
    extent_mm = extent_min_mm, extent_max_mm
    return extent_mm

def get_foreground_extent(
    image: LabelData3D,
    offset: Optional[Point3D] = None,
    spacing: Optional[Spacing3D] = None,
    use_patient_coords: bool = True) -> Optional[Union[PixelBox, VoxelBox]]:
    # Get voxel extent.
    if image.sum() > 0:
        non_zero = np.argwhere(image != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        extent_vox = (min, max)
        if not use_patient_coords:
            return extent_vox
    else:
        return None

    # Get mm extent.
    if use_patient_coords:
        assert spacing is not None
        assert offset is not None
    extent_min_vx, extent_max_vx = extent_vox
    extent_min_mm = tuple(np.array(extent_min_vx) * spacing + offset)
    extent_max_mm = tuple(np.array(extent_max_vx) * spacing + offset)
    extent_mm = extent_min_mm, extent_max_mm
    return extent_mm

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

def extent_width(a: np.ndarray) -> Optional[Union[Size2D, Size3D]]:
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
