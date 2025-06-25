import numpy as np
from typing import *

from mymi.typing import *

def round(
    x: Union[Number, List[Number], np.ndarray],
    tol: Number = 1) -> Union[Number, List[Number], np.ndarray]:
    if isinstance(x, list):
        return list(np.round(np.array(x) / tol) * tol)
    else:
        return np.round(np.array(x) / tol) * tol
