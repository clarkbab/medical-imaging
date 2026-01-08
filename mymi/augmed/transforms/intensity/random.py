from typing import *

from mymi.typing import *

from ..random import RandomTransform
from .intensity import IntensityTransform

class RandomIntensityTransform(RandomTransform, IntensityTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def transform_intensity(
        self,
        image: ImageTensor,
        **kwargs) -> ImageTensor:
        return self.freeze().transform_intensity(image, **kwargs)
