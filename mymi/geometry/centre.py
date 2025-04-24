import numpy as np
from typing import Optional, Union

from mymi.typing import ImageSize2D, ImageSize3D, Pixel, Voxel

def get_centre(a: np.ndarray) -> Optional[Union[Pixel, Voxel]]:
    return get_centre_from_size(a.shape)

def get_centre_from_size(s: Union[ImageSize2D, ImageSize3D]) -> Optional[Union[Pixel, Voxel]]:
    return tuple([int(np.floor(si / 2)) - 1 for si in s])
