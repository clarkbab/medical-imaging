from typing import *

from mymi.typing import *

from ..random import RandomTransform
from .spatial import SpatialTransform

class RandomSpatialTransform(RandomTransform, SpatialTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def back_transform_points(
        self,
        points: PointsTensor,
        seed: Optional[int] = None,
        **kwargs) -> PointsTensor:
        return self.freeze().back_transform_points(points, **kwargs)
 