import numpy as np
from typing import Tuple

from mymi.typing import *

def create_affine(
    spacing: Spacing,
    origin: Point,
    ) -> Affine:
    dim = len(spacing)
    assert len(origin) == dim, f"Length of 'origin' must match length of 'spacing'. Got {spacing} and {origin}." 
    if dim == 2:
        affine = np.eye(3)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[0, 2] = origin[0]
        affine[1, 2] = origin[1]
    else:
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        affine[0, 3] = origin[0]
        affine[1, 3] = origin[1]
        affine[2, 3] = origin[2]

    return affine

def affine_origin(
    affine: Affine,
    ) -> Point:
    # Get origin.
    dim = affine.shape[0] - 1
    if dim == 2:
        origin = (affine[0, 2], affine[1, 2])
    else:
        origin = (affine[0, 3], affine[1, 3], affine[2, 3])
    origin = tuple(float(o) for o in origin)
    return origin

def affine_spacing(
    affine: Affine,
    ) -> Spacing:
    # Get spacing.
    dim = affine.shape[0] - 1
    if dim == 2:
        spacing = (affine[0, 0], affine[1, 1])
    else:
        spacing = (affine[0, 0], affine[1, 1], affine[2, 2])
    spacing = tuple(float(s) for s in spacing)
    return spacing

def get_voxel_isocentre_from_affine(affine: Affine) -> Voxel:
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    iso_vox = tuple(int(i) for i in np.round(-np.array(origin) / spacing))
    return iso_vox
