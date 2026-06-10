from dicomset.utils import to_numpy, logger, to_tensor
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
    data: Image,
    air_hu: float | None = None,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
) -> Image:
    data, return_type = to_tensor(data, dtype=torch.float32, return_type=True)
    if air_hu is None:
        air_hu = infer_air_hu(data, background_threshold=background_threshold)
    data = data.clone()
    data[data <= background_threshold] = air_hu
    if return_type is np.ndarray:
        data = to_numpy(data)
    return data
