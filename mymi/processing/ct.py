from typing import Optional, Tuple

import numpy as np

from mymi import logging
from mymi.typing import ImageArray

BACKGROUND_HU_THRESHOLD = -2000
AIR_HU_RANGE = (-1500, -500)

def infer_air_hu(
    data: ImageArray,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
    air_range: Tuple[float, float] = AIR_HU_RANGE,
) -> float:
    low, high = air_range
    air_voxels = data[(data > background_threshold) & (data >= low) & (data <= high)]
    if air_voxels.size == 0:
        raise ValueError(
            f"No voxels found in air range ({low}, {high}); "
            "cannot infer air HU from this image."
        )
    return float(np.median(air_voxels))

def has_ct_background(
    data: ImageArray,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
) -> bool:
    return bool(np.any(data <= background_threshold))

def fill_ct_background(
    data: ImageArray,
    air_hu: float | None = None,
    background_threshold: float = BACKGROUND_HU_THRESHOLD,
) -> ImageArray:
    logging.log_args('Filling CT background')
    if air_hu is None:
        air_hu = infer_air_hu(data, background_threshold=background_threshold)
    data = data.copy()
    data[data <= background_threshold] = air_hu
    return data
