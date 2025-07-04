from mymi.typing import *

def to_rgb_255(colour: Colour) -> Tuple[int, int, int]:
    return tuple((255 * np.array(colour)).astype(int))
