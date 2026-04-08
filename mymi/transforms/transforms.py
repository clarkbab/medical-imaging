from dicomset.utils import affine_origin, affine_spacing
import numpy as np
from typing import *

from mymi import logging
from mymi.typing import *

def assert_box_width(box: Union[Box2D, Box3D]) -> None:
    # Check box width.
    min, max = box
    for min_i, max_i in zip(min, max):
        width = max_i - min_i
        if width <= 0:
            raise ValueError(f"Box width must be positive, got '{box}'.")

def replace_box_none(
    bounding_box: Union[Box2D, Box3D],
    size: Union[Size2D, Size3D],
    affine: AffineMatrix | None = None,
    ) -> Tuple[Box2D, Box3D]:
    # Replace 'None' values.
    n_dims = len(size)
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(n_dims):
        if min[i] is None:
            if affine is not None:
                origin = affine_origin(affine)

                min[i] = origin[i]
            else:
                min[i] = 0
        if max[i] is None:
            if affine is not None:
                spacing = affine_spacing(affine)
                origin = affine_origin(affine)
                max[i] = size[i] * spacing[i] + origin[i]
            else:
                max[i] = size[i]
    min, max = tuple(min), tuple(max)
    return min, max
