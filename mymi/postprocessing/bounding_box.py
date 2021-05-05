import numpy as np

def bounding_box(a):
    """
    returns: a tuple of bounding box centre and shape.
    args:
        a: a 3D array with binary values.
    """
    # Find min/max points.
    min = np.argwhere(a == 1).min(0)
    max = np.argwhere(a == 1).max(0)

    # Get dimensions.
    dim = max - min

    return min, dim
