import numpy as np
from typing import Optional, Union

from mymi.typing import ImageSize2D, ImageSize3D, Point2D, Point3D

def get_centre(a: np.ndarray) -> Optional[Union[Point2D, Point3D]]:
    return get_centre_from_size(a.shape)

def get_centre_from_size(s: Union[ImageSize2D, ImageSize3D]) -> Optional[Union[Point2D, Point3D]]:
    return tuple([int(np.floor(si / 2)) - 1 for si in s])
