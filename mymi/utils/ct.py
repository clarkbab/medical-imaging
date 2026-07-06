from dicomset.utils import to_numpy, logger, to_tensor, affine_spacing, fov_width
from dicomset.typing import *
from typing import Optional, Tuple
import torch

import numpy as np

from mymi import logging
from mymi.typing import ImageArray

BACKGROUND_HU_THRESHOLD = -2000
AIR_HU_RANGE = (-1500, -500)

def infer_air_hu(
    data: Image,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
    air_range: Tuple[float, float] = AIR_HU_RANGE,
) -> float:
    low, high = air_range
    data = to_numpy(data)
    air_voxels = data[(data > background_threshold) & (data >= low) & (data <= high)]
    if air_voxels.size == 0:
        raise ValueError(
            f"No voxels found in air range ({low}, {high}); "
            "cannot infer air HU from this image."
        )
    return float(np.median(air_voxels))

def has_ct_background(
    data: Image,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
    ) -> bool:
    data = to_numpy(data)
    return bool(np.any(data <= background_threshold))

def fill_ct_background(
    ct: Image,
    affine: AffineMatrix3D | None = None,
    air_hu: float | None = None,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
    ) -> Image:
    ct, return_type = to_tensor(ct, dtype=torch.float32, return_type=True)
    if air_hu is None:
        air_hu = infer_air_hu(ct, background_threshold=background_threshold)
    ct = ct.clone()
    if affine is not None:
        spacing = to_numpy(affine_spacing(affine))
        assert spacing[0] == spacing[1], "spacing must be isotropic in axial plane"
        fov_mm = fov_width(ct.shape[:2], affine)
        radius_px = float(fov_mm[0]) / 2 / spacing[0]
        nx, ny = ct.shape[0], ct.shape[1]
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        xs = np.arange(nx, dtype=np.float64) - cx
        ys = np.arange(ny, dtype=np.float64) - cy
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        outside = torch.from_numpy(xx ** 2 + yy ** 2 > (radius_px - 1) ** 2)
        if ct.ndim == 3:
            outside = outside.unsqueeze(-1).expand_as(ct)
        ct[outside] = air_hu
    else:
        ct[ct <= background_threshold] = air_hu
    if return_type is np.ndarray:
        ct = to_numpy(ct)
    return ct
