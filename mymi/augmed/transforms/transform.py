from typing import *

from mymi.typing import *

# What is a Transform?
# Transform defines the API that any (deterministic) Transform
# and RandomTransform must follow.
# What about pipeline? Yeah, I guess so. We treat it just like a transform.
class Transform:
    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Subclasses of 'Transform' must implement 'back_transform_points' method.")
    
    # Alias for 'transform' method.
    def __call__(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        # Require this ordering of kwargs for API simplicity.
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        return self.transform(data, origin=origin, spacing=spacing, **kwargs)

    @property
    def dim(self) -> int:
        if not hasattr(self, '_dim'):
            raise ValueError("Subclasses of 'Transform' must have '_dim' attribute.")
        return self._dim

    @property
    def is_homogeneous(self) -> int:
        if not hasattr(self, '_is_homogeneous'):
            raise ValueError("Subclasses of 'Transform' must have '_is_homogeneous' attribute.")
        return self._is_homogeneous

    @property
    def params(self) -> Dict[str, Any]:
        if not hasattr(self, '_params'):
            raise ValueError("Subclasses of 'Transform' must have '_params' attribute.")
        return self._params

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        raise ValueError("Subclasses of 'Transform' must implement '__str__' method.")

    def transform(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform' method.")

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        **kwargs) -> Union[ImageArray, ImageTensor]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_image' method.")

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_points' method.")
