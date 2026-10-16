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

def hu_to_attenuation(
    data: Image,
    mu_water: float = 0.013,    # From the markerless codebase.
    ) -> Image:
    """Converts Hounsfield Units to (relative) linear attenuation coefficients.

    By definition HU = 1000 * (mu - mu_water) / mu_water, so mu = mu_water * (1 + HU / 1000).
    With the default mu_water=1.0, the output is attenuation relative to water (air -> 0,
    water -> 1), which is sufficient to produce physically-sensible, non-negative projections
    without needing an absolute attenuation coefficient at a specific photon energy. Values are
    clamped at 0 as HU below -1000 (air) is non-physical and can occur due to noise.
    """
    data, return_type = to_tensor(data, dtype=torch.float32, return_type=True)
    mu = mu_water * (1.0 + data / 1000.0)
    mu = torch.clamp(mu, min=0.0)
    if return_type is np.ndarray:
        mu = to_numpy(mu)
    return mu

def fill_ct_background(
    ct: Image,
    affine: AffineMatrix3D | None = None,
    air_hu: float | None = None,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
    half_only: bool = False,  # Testing the cause of the stripe pattern.
    quarter_only: bool = False,  # Testing the cause of the stripe pattern.
    ) -> Image:
    ct, return_type = to_tensor(ct, dtype=torch.float32, return_type=True)
    if air_hu is None:
        air_hu = infer_air_hu(ct, background_threshold=background_threshold)
    logger.info(f"Filling CT background with air HU value: {air_hu:.2f}")
    ct = ct.clone()
    if affine is not None:
        spacing = to_numpy(affine_spacing(affine))
        assert spacing[0] == spacing[1], "spacing must be isotropic in axial plane"
        fov_mm = fov_width(ct.shape, affine)
        radius_px = float(fov_mm[0]) / 2 / spacing[0]
        nx, ny = ct.shape[0], ct.shape[1]
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        xs = np.arange(nx, dtype=np.float64) - cx
        ys = np.arange(ny, dtype=np.float64) - cy
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        outside = torch.from_numpy(xx ** 2 + yy ** 2 > (radius_px - 1) ** 2)
        if ct.ndim == 3:
            outside = outside.unsqueeze(-1).expand_as(ct)
        if half_only:
            xx_mask = torch.from_numpy(xx > 0)
            if ct.ndim == 3:
                xx_mask = xx_mask.unsqueeze(-1).expand_as(ct)
            outside = outside & xx_mask
        elif quarter_only:
            xx_mask = torch.from_numpy(xx > 0)
            yy_mask = torch.from_numpy(yy > 0)
            if ct.ndim == 3:
                xx_mask = xx_mask.unsqueeze(-1).expand_as(ct)
                yy_mask = yy_mask.unsqueeze(-1).expand_as(ct)
            outside = outside & (xx_mask | yy_mask)
        ct[outside] = air_hu
    else:
        ct[ct <= background_threshold] = air_hu
    if return_type is np.ndarray:
        ct = to_numpy(ct)
    return ct
