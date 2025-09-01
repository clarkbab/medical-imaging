from typing import *

from mymi.typing import *

from .transform import Transform

# 'transform' methods need to call 'freeze' before applying the
# transform, so we override these.
# If this the best? Should these be a subclass of transform?
# Maybe rather than overriding, we should define the API in 'Transform'
# and then have a mixin for those classes that use the 'transform', 
# 'transform_image', and 'transform_points' methods directly (i.e.)
# deterministic transforms.
class RandomTransform(Transform):
    def __init__(
        self,
        random_seed: Optional[int] = None):
        self.seed_rng(random_seed=random_seed)

    def back_transform_points(
        self,
        points: PointsTensor,
        random_seed: Optional[int] = None,
        **kwargs) -> PointsTensor:
        t = self.freeze(random_seed=random_seed)
        return t.back_transform_points(points, **kwargs)

    def seed_rng(
        self,
        random_seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed=random_seed)

    # Overrides 'Transform.transform'.
    def transform(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        random_seed: Optional[int] = None,
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        t = self.freeze(random_seed=random_seed)
        return t.transform(data, **kwargs)

    # Overrides 'Transform.transform_image'.
    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        random_seed: Optional[int] = None,
        **kwargs) -> Union[ImageArray, ImageTensor]:
        t = self.freeze(random_seed=random_seed)
        return t.transform_image(image, **kwargs)

    # Overrides 'Transform.transform_points'.
    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        random_seed: Optional[int] = None,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        t = self.freeze(random_seed=random_seed)
        return t.transform_points(points, origin=origin, size=size, spacing=spacing, **kwargs)
