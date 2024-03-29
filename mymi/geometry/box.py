import numpy as np

from mymi import types

def get_box(
    centre: types.Point3D,
    size: types.Size3D) -> types.Box3D:
    # Convert to box.
    size = np.array(size)
    lower_sub = np.ceil(size / 2).astype(int)
    min = tuple(centre - lower_sub)
    max = tuple(min + size)
    box = (min, max)

    return box
