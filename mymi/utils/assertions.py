import numpy as np
import os
import torch
from typing import *

from mymi.typing import *

from .args import arg_to_list

def assert_can_write(filepaths: FilePaths) -> None:
    filepaths = arg_to_list(filepaths, FilePath)
    for f in filepaths:
        if os.path.exists(f) and not os.access(f, os.W_OK):
            raise PermissionError(f"File '{f}' is open or read-only, cannot overwrite.")

def assert_image(d: Any) -> None:
    if not isinstance(d, np.ndarray) and not isinstance(d, torch.Tensor):
        raise TypeError(f"Expected numpy or torch array, got {type(d)}.")
    elif len(d.shape) not in (2, 3, 4, 5):
        raise ValueError(f"Expected (2, 3, 4, or 5)-dimensional image, got {len(d.shape)}D.")

def assert_numpy(d: Any) -> None:
    if not isinstance(d, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(d)}.")
