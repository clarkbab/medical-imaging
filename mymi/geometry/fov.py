import numpy as np
from typing import *

from mymi.typing import *
from mymi.utils import *

@alias_kwargs(('uwc', 'use_world_coords'))
def foreground_fov(
    data: LabelArray,
    affine: Affine | None = None,
    use_world_coords: bool = True) -> Optional[Box]:
    if data.sum() == 0:
        return None
    if use_world_coords:
        assert affine is not None

    # Get fov of foreground objects.
    non_zero = np.argwhere(data != 0).astype(int)
    fov_vox = tuple(non_zero.min(axis=0)), tuple(non_zero.max(axis=0))
    if not use_world_coords:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_min_vox, fov_max_vox = fov_vox
    fov_min_mm = tuple(np.array(fov_min_vox) * spacing + origin)
    fov_max_mm = tuple(np.array(fov_max_vox) * spacing + origin)
    fov_mm = fov_min_mm, fov_max_mm
    return fov_mm

@alias_kwargs(('uwc', 'use_world_coords'))
def foreground_fov_centre(
    data: LabelArray,
    use_world_coords: bool = True,
    **kwargs) -> Optional[Union[Pixel, Voxel]]:
    fov_d = foreground_fov(data, use_world_coords=use_world_coords, **kwargs)
    if fov_d is None:
        return None
    fov_c = np.array(fov_d).sum(axis=0) / 2
    if not use_world_coords:
        fov_c = np.floor(fov_c).astype(int)
    fov_c = tuple(fov_c)
    return fov_c

@alias_kwargs(('uwc', 'use_world_coords'))
def foreground_fov_width(
    data: LabelArray,
    spacing: Optional[Spacing] = None,
    origin: Optional[Point] = None,
    use_world_coords: bool = True) -> Optional[Size]:
    # Get foreground fov.
    fov_fg = foreground_fov(data, use_world_coords=use_world_coords, spacing=spacing, origin=origin)
    if fov_fg is None:
        return None
    min, max = fov_fg
    fov_w = tuple(np.array(max) - min)
    return fov_w

def fov(
    size: Size,
    affine: Optional[Affine] = None,
    raise_error: bool = True,
    use_world_coords: bool = True,
    ) -> Box:
    if use_world_coords:
        assert affine is not None

    # Get fov in voxels.
    n_dims = len(size)
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        assert len(spacing) == n_dims, f"Expected spacing to have {n_dims} dimensions, got {spacing}."
        assert len(origin) == n_dims, f"Expected origin to have {n_dims} dimensions, got {origin}."
    fov_vox = ((0,) * n_dims, size)
    if not use_world_coords:
        return fov_vox

    # Get fov in mm.
    fov_min_vox, fov_max_vox = fov_vox
    fov_min_mm = tuple(float(e) for e in (np.array(fov_min_vox) * spacing + origin))
    fov_max_mm = tuple(float(e) for e in (np.array(fov_max_vox) * spacing + origin))
    fov_mm = fov_min_mm, fov_max_mm

    return fov_mm

@alias_kwargs(('uwc', 'use_world_coords'))
def fov_centre(
    size: Size,
    use_world_coords: bool = True,
    **kwargs,
    ) -> Point:
    # Get FOV.
    fov_d = fov(size, use_world_coords=use_world_coords, **kwargs)

    # Get FOV centre.
    fov_c = np.array(fov_d).sum(axis=0) / 2
    if not use_world_coords:
        fov_c = np.floor(fov_c).astype(int)
    fov_c = tuple(fov_c)
    return fov_c

@alias_kwargs(('uwc', 'use_world_coords'))
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
