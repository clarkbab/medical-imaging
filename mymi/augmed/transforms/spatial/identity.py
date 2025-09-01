from typing import *

from mymi.typing import *

from .spatial import SpatialTransform

# Doesn't use TransformMixin/TransformImageMixin as we don't need to transform
# the image using 'back_transform_points' route.
class IdentityTransform(SpatialTransform):
    def __init__(
        self, 
        **kwargs) -> None:
        self._is_homogeneous = True

    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        return points

    def transform(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        return data

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        **kwargs) -> Union[ImageArray, ImageTensor]:
        return image

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        return_filtered: bool = False,
        **kwargs) -> Union[PointsArray, PointsTensor]:
        if return_filtered:
            # Create filtered indices to match API.
            indices = to_tensor([], device=points.device, dtype=torch.int) if isinstance(points, torch.Tensor) else np.array([])
            return points, indices 
        return points
