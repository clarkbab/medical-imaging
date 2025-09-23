import torch
from typing import *

from mymi.typing import *
from mymi.utils import alias_kwargs, arg_to_list

from ...utils import *

class TransformImageMixin:
    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Classes with 'TransformImageMixin' must implement 'back_transform_points' method.")
