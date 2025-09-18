import numpy as np
from typing import *

from mymi import logging
from mymi.typing import *
from mymi.utils import *

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
    origin: Optional[Union[Point2D, Point3D]] = None,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True) -> Tuple[Box2D, Box3D]:
    if use_patient_coords:
        assert spacing is not None
        assert origin is not None

    # Replace 'None' values.
    n_dims = len(size)
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(n_dims):
        if min[i] is None:
            if use_patient_coords:
                min[i] = origin[i]
            else:
                min[i] = 0
        if max[i] is None:
            if use_patient_coords:
                max[i] = size[i] * spacing[i] + origin[i]
            else:
                max[i] = size[i]
    min, max = tuple(min), tuple(max)
    return min, max
