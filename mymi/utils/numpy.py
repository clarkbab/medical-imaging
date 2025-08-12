import numpy as np

from typing import *

def to_tuple(a: np.ndarray) -> Tuple[Union[bool, int, float]]:
    k = a.dtype.kind
    if k == 'b':
        conv_fn = bool
    elif k == 'f':
        conv_fn = float
    elif k == 'i':
        conv_fn = int
    else:
        raise ValueError(f"Unsupported numpy array kind '{k}' for conversion to tuple.")
    return tuple(conv_fn(ai) for ai in a)
