import numpy as np
from typing import Optional

from mymi import types

def label_extent(label: np.ndarray) -> Optional[types.Box3D]:
    if label.sum() == 0:
        extent = None
    else:
        non_zero = np.argwhere(label != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        extent = (min, max)

    return extent
