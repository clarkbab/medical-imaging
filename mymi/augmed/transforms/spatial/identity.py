from typing import *

from mymi.typing import *

from .spatial import SpatialTransform

class IdentityTransform(SpatialTransform):
    def back_transform_points(
        self,
        points: Points,
        **kwargs) -> Points:
        return points

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        **kwargs) -> Union[ImageArray, ImageTensor]:
        return image

    def transform_points(
        self,
        points: Points,
        **kwargs) -> Points:
        return points
