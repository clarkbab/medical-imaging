from typing import *

from mymi.typing import *

from ..random import RandomTransform
from .grid import GridTransform

class RandomGridTransform(RandomTransform, GridTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def transform_grid(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        return self.freeze().transform_grid(size, spacing, origin, **kwargs)
