import numpy as np
import scipy

from mymi.typing import *

def centre_of_mass(a: np.ndarray) -> Voxel:
    com = scipy.ndimage.center_of_mass(a)
    com = [int(np.round(c)) for c in com]
    return com
