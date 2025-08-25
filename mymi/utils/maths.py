import numpy as np
from typing import *

from mymi.typing import *

from .args import arg_to_list

def array_equal(
    a: Union[List[Any], np.ndarray, torch.Tensor],
    b: Union[List[Any], np.ndarray, torch.Tensor],
    tol: Optional[Number] = None) -> None:
    a, b = np.array(a), np.array(b)
    if tol is not None:
        a, b = round(a, tol=tol), round(b, tol=tol) 
    return np.array_equal(a, b)

def round(
    x: Union[Number, List[Number], np.ndarray],
    tol: Number = 1) -> Union[Number, List[Number], np.ndarray]:
    x_type = type(x)
    x = arg_to_list(x, Number)
    x = tol * np.round(np.array(x) / tol)
    return x[0] if x_type is Number else x.tolist() if x_type is list else x 
