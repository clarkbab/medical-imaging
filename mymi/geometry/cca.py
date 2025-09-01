import numpy as np

from mymi.typing import *

def largest_cc(a: LabelArray) -> LabelArray:
    # Check for foreground voxels.
    if a.sum() == 0:
        return np.zeros_like(a)
    
    # Calculate largest component.
    largest_cc = a == np.argmax(np.bincount(a.flat)[1:]) + 1

    return largest_cc
