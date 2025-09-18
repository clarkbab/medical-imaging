import numpy as np
from typing import *

from mymi.typing import *
from mymi.utils import *

@alias_kwargs(('upc', 'use_patient_coords'))
def foreground_fov(
    data: LabelArray,
    spacing: Optional[Spacing] = None,
    origin: Optional[Point] = None,
    use_patient_coords: bool = True) -> Optional[Box]:
    if data.sum() == 0:
        return None
    if use_patient_coords:
        assert spacing is not None
        assert origin is not None

    # Get fov of foreground objects.
    non_zero = np.argwhere(data != 0).astype(int)
    fov_vox = tuple(non_zero.min(axis=0)), tuple(non_zero.max(axis=0))
    if not use_patient_coords:
        return fov_vox

    # Get fov in mm.
    fov_min_vox, fov_max_vox = fov_vox
    fov_min_mm = tuple(np.array(fov_min_vox) * spacing + origin)
    fov_max_mm = tuple(np.array(fov_max_vox) * spacing + origin)
    fov_mm = fov_min_mm, fov_max_mm
    return fov_mm

@alias_kwargs(('upc', 'use_patient_coords'))
def foreground_fov_centre(
    data: LabelArray,
    use_patient_coords: bool = True,
    **kwargs) -> Optional[Union[Pixel, Voxel]]:
    fov_d = foreground_fov(data, use_patient_coords=True, **kwargs)
    if fov_d is None:
        return None
    fov_c = np.array(fov_d).sum(axis=0) / 2
    if not use_patient_coords:
        fov_c = np.floor(fov_c).astype(int)
    fov_c = tuple(fov_c)
    return fov_c

@alias_kwargs(('upc', 'use_patient_coords'))
def foreground_fov_width(
    data: LabelArray,
    spacing: Optional[Spacing] = None,
    origin: Optional[Point] = None,
    use_patient_coords: bool = True) -> Optional[Size]:
    # Get foreground fov.
    fov_fg = foreground_fov(data, use_patient_coords=use_patient_coords, spacing=spacing, origin=origin)
    if fov_fg is None:
        return None
    min, max = fov_fg
    fov_w = tuple(np.array(max) - min)
    return fov_w

def fov(
    data: Union[ImageArray, ImageTensor],
    spacing: Optional[Spacing] = None,
    origin: Optional[Point] = None,
    raise_error: bool = True,
    use_patient_coords: bool = True) -> Box:
    if use_patient_coords:
        assert spacing is not None
        assert origin is not None

    # Get fov in voxels.
    n_dims = len(data.shape)
    if spacing is not None:
        assert len(spacing) == n_dims, f"Expected spacing to have {n_dims} dimensions, got {spacing}."
    if origin is not None:
        assert len(origin) == n_dims, f"Expected origin to have {n_dims} dimensions, got {origin}."
    fov_vox = ((0,) * n_dims, data.shape)
    if not use_patient_coords:
        return fov_vox

    # Get fov in mm.
    fov_min_vox, fov_max_vox = fov_vox
    fov_min_mm = tuple(float(e) for e in (np.array(fov_min_vox) * spacing + origin))
    fov_max_mm = tuple(float(e) for e in (np.array(fov_max_vox) * spacing + origin))
    fov_mm = fov_min_mm, fov_max_mm

    return fov_mm

@alias_kwargs(('upc', 'use_patient_coords'))
def fov_centre(
    data: Union[ImageArray, ImageTensor],
    use_patient_coords: bool = True,
    **kwargs) -> Optional[Union[Pixel, Voxel]]:
    # Get FOV.
    fov_d = fov(data, use_patient_coords=use_patient_coords, **kwargs)
    if fov_d is None:
        return None

    # Get FOV centre.
    fov_c = np.array(fov_d).sum(axis=0) / 2
    if not use_patient_coords:
        fov_c = np.floor(fov_c).astype(int)
    fov_c = tuple(fov_c)
    return fov_c

@alias_kwargs(('upc', 'use_patient_coords'))
def fov_width(
    data: LabelArray,
    **kwargs) -> Size:
    fov_d = fov(data, **kwargs)
    if fov_d is None:
        return None
    
    # Get width.
    min, max = fov_d
    fov_w = tuple(np.array(max) - min)
    return fov_w
