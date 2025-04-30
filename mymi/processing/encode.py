import numpy as np
from typing import Optional

def one_hot_encode(
    a: np.ndarray,
    dims: Optional[int] = None) -> np.ndarray:
    if dims is None:
        dims = a.max() + 1
    encoded = (np.arange(dims) == a[...,None]).astype(bool)
    encoded = np.moveaxis(encoded, -1, 0)
    return encoded
