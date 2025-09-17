from typing import *

from mymi.typing import *

from ...utils import *
from ..mixins import AffineMixin
from .spatial import SpatialTransform

class Identity(AffineMixin, SpatialTransform):
    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        return points

    # This is called by 'Pipeline'. Adding this so that we don't have to include
    # special logic for skipping identity back transform points in pipeline. 
    # Could add a bit of overhead if the identity transform isn't chained with
    # other homogeneous transforms.
    def get_affine_back_transform(
        self,
        device: torch.device,
        **kwargs) -> torch.Tensor:
        return create_eye(self._dim + 1, device=device)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(dim={self._dim})"

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
