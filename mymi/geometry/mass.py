import numpy as np
import scipy
from typing import *

from mymi.typing import *

def get_centre_of_mass(
    a: Union[SliceArray, ImageArray],
    origin: Optional[Point] = None,
    spacing: Optional[Spacing] = None, 
    use_world_coords: bool = True) -> Point:
    com = scipy.ndimage.center_of_mass(a)
    if use_world_coords:
        com = tuple(np.array(com) * spacing + origin)
    else:
        com = tuple([int(np.round(c)) for c in com])
    return com
