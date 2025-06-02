import numpy as np
from typing import Callable, Dict, Tuple

from mymi.transforms import centre_crop_or_pad
from mymi.typing import Spacing3D

def centre_crop(crop_mm: Tuple[float, float, float]) -> Callable:
    assert crop_mm is not None
    def crop_fn(
        dataset: str,
        sample_id: str,
        input: np.ndarray,
        labels: Dict[str, np.ndarray],
        spacing: Spacing3D = None):
        assert spacing is not None

        # Crop input.
        crop = tuple(np.round(np.array(crop_mm) / spacing).astype(int))
        input = centre_crop_or_pad(input, crop, use_patient_coords=False)

        # Crop labels.
        for r in labels.keys():
            labels[r] = centre_crop_or_pad(labels[r], crop, use_patient_coords=False)

        return input, labels
    return crop_fn
