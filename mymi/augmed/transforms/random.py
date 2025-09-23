from typing import *

from mymi.typing import *

from .transform import Transform

# RandomTransforms should have all the behaviour of a normal transform.
class RandomTransform(Transform):
    def __init__(
        self,
        p: Number = 1.0,    # What proportion of the time is the transform applied? Un-applied transforms resolve to 'Identity' when frozen.
        random_seed: Optional[int] = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        print('init random transform')
        self._p = p
        self.seed(random_seed=random_seed)

    def back_transform_points(
        self,
        points: PointsTensor,
        random_seed: Optional[int] = None,
        **kwargs) -> PointsTensor:
        t = self.freeze()
        return t.back_transform_points(points, **kwargs)

    def freeze(
        self,
        **kwargs) -> Transform:
        raise ValueError("Subclasses of 'RandomTransform' must implement 'freeze' method.")

    def seed(
        self,
        random_seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed=random_seed)

    def transform(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        random_seed: Optional[int] = None,
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        t = self.freeze()
        return t.transform(data, **kwargs)

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        random_seed: Optional[int] = None,
        **kwargs) -> Union[ImageArray, ImageTensor]:
        t = self.freeze()
        return t.transform_image(image, **kwargs)

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        random_seed: Optional[int] = None,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        t = self.freeze()
        return t.transform_points(points, origin=origin, size=size, spacing=spacing, **kwargs)
