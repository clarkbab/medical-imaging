import numpy as np
from typing import *

from mymi.typing import *

def foreground_fov(
    data: LabelData,
    offset: Optional[Point] = None,
    spacing: Optional[Spacing] = None,
    use_patient_coords: bool = True) -> Optional[Box]:
    # Get fov of foreground objects.
    if data.sum() > 0:
        non_zero = np.argwhere(data != 0).astype(int)
        fov_vox = tuple(non_zero.min(axis=0)), tuple(non_zero.max(axis=0))
        if not use_patient_coords:
            return fov_vox
    else:
        return None

    # Get fov in mm.
    if use_patient_coords:
        assert spacing is not None
        assert offset is not None
    fov_min_vox, fov_max_vox = fov_vox
    fov_min_mm = tuple(np.array(fov_min_vox) * spacing + offset)
    fov_max_mm = tuple(np.array(fov_max_vox) * spacing + offset)
    fov_mm = fov_min_mm, fov_max_mm
    return fov_mm

def foreground_fov_centre(
    data: LabelData,
    use_patient_coords: bool = True,
    **kwargs) -> Optional[Union[Pixel, Voxel]]:
    fov_box = foreground_fov(data, use_patient_coords=True, **kwargs)
    if fov_box is not None:
        fov_c = np.floor(np.array(fov_box).sum(axis=0) / 2).astype(int)
        if not use_patient_coords:
            fov_c = fov_c.astype(int)
        fov_c = tuple(fov_c)
        return fov_c
    else:
        return None

def foreground_fov_width(
    data: LabelData,
    offset: Optional[Point] = None,
    spacing: Optional[Spacing] = None,
    use_patient_coords: bool = True) -> Optional[Size]:
    # Get foreground fov.
    fov_fg = foreground_fov(data, use_patient_coords=use_patient_coords, offset=offset, spacing=spacing)
    if fov_fg is not None:
        min, max = fov_fg
        fov_w = tuple(np.array(max) - min)
        return fov_w
    else:
        return None

def fov(
    data: LabelData,
    offset: Optional[Point] = None,
    spacing: Optional[Spacing] = None,
    raise_error: bool = True,
    use_patient_coords: bool = True) -> Box:
    if data.sum() == 0:
        raise ValueError("Input data is empty, cannot compute fov.") if raise_error else None    
    if use_patient_coords:
        assert spacing is not None
        assert offset is not None

    # Get fov in voxels.
    n_dims = len(data.shape)
    fov_vox = ((0,) * n_dims, data.shape)
    if not use_patient_coords:
        return fov_vox

    # Get fov in mm.
    fov_min_vox, fov_max_vox = fov_vox
    fov_min_mm = tuple(float(e) for e in (np.array(fov_min_vox) * spacing + offset))
    fov_max_mm = tuple(float(e) for e in (np.array(fov_max_vox) * spacing + offset))
    fov_mm = fov_min_mm, fov_max_mm

    return fov_mm

def fov_centre(
    data: LabelData,
    use_patient_coords: bool = True,
    **kwargs) -> Union[Pixel, Voxel]:
    fov_box = fov(data, use_patient_coords=True, **kwargs)
    if fov_box is not None:
        fov_c = np.floor(np.array(fov_box).sum(axis=0) / 2).astype(int)
        if not use_patient_coords:
            fov_c = fov_c.astype(int)
        fov_c = tuple(fov_c)
        return fov_c
    else:
        return None

def fov_width(
    data: LabelData,
    **kwargs) -> Size:
    fov_d = fov(data, **kwargs)
    if fov_d is not None:
        min, max = fov_d
        fov_w = tuple(np.array(max) - min)
        return fov_w
    else:
        return None

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
