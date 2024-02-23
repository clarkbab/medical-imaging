import numpy as np
import SimpleITK as sitk
from typing import List, Tuple

from mymi import types

def volume(
    a: np.ndarray,
    spacing: types.Spacing3D) -> float:
    if a.dtype != np.bool_:
        raise ValueError(f"Metric 'volume' expects boolean array. Got '{a.dtype}'.")

    # Calculate volume.
    voxel_vol = np.product(spacing)
    n_voxels = a.sum()
    vol = n_voxels * voxel_vol

    return vol
