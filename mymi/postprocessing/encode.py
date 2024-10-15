import numpy as np

def one_hot_encode(a: np.ndarray) -> np.ndarray:
    encoded = (np.arange(a.max() + 1) == a[...,None]).astype(bool)
    encoded = np.moveaxis(encoded, -1, 0)
    return encoded
