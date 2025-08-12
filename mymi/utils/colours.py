import numpy as np
from typing import *

from mymi.typing import *

def to_rgb(colour: Colour) -> Tuple[int, int, int]:
    raise ValueError('this is a nonsense function')
    return tuple((255 * np.array(colour)).astype(int))
