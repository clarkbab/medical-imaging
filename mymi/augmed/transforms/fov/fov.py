from typing import *

from mymi.typing import *

from ..transform import Transform

class FovTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def transform_fov(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        raise ValueError("Subclasses of 'FovTransform' must implement 'transform_fov' method.")
