from typing import *

from mymi.typing import *

from .transform import Transform

# RandomTransforms should have all the behaviour of a normal transform.
class RandomTransform(Transform):
    def __init__(
        self,
        p: Number = 1.0,    # What proportion of the time is the transform applied? Un-applied transforms resolve to 'Identity' when frozen.
        seed: Optional[int] = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self._p = p
        self.set_seed(seed)

    def freeze(
        self,
        klass: 'Object',
        params: Dict[str, Any]) -> None:
        # Copy general params from random -> frozen transform. I always forget these.
        params['dim'] = self._dim
        params['use_image_coords'] = self._use_image_coords
        return klass(**params)

    def set_seed(
        self,
        seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed=seed)

    def __str__(
        self,
        class_name: str,
        params: Dict[str, str]) -> str:
        params['p'] = self._p
        return super().__str__(class_name, params)

    def transform(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        seed: Optional[int] = None,
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        return self.freeze().transform(data, **kwargs)

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        seed: Optional[int] = None,
        **kwargs,
        ) -> Union[ImageArray, ImageTensor, List[Union[ImageArray, ImageTensor, Union[ImageGrid, List[ImageGrid]]]]]:
        return self.freeze().transform_image(image, **kwargs)

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        seed: Optional[int] = None,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        return self.freeze().transform_points(points, origin=origin, size=size, spacing=spacing, **kwargs)
