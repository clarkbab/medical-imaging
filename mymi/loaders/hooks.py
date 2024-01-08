import numpy as np
from typing import Callable, Dict, Tuple

from mymi.transforms import centre_crop_or_pad_3D, crop_3D
from mymi.types import ImageSpacing3D

def naive_crop(
    dataset: str,
    sample_id: str,
    input: np.ndarray,
    labels: Dict[str, np.ndarray],
    spacing: ImageSpacing3D = None):
    assert spacing is not None

    # Crop input.
    # crop_mm = (320, 520, 730)   # With 60 mm margin (30 mm either end) for each axis.
    crop_mm = (250, 400, 500)   # With 60 mm margin (30 mm either end) for each axis.
    crop = tuple(np.round(np.array(crop_mm) / spacing).astype(int))
    input = centre_crop_or_pad_3D(input, crop)

    # Crop labels.
    for r in labels.keys():
        labels[r] = centre_crop_or_pad_3D(labels[r], crop)

    return input, labels
