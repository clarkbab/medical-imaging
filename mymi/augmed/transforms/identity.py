from typing import *

from mymi.typing import *

from ...utils import *
from .transform import Transform

class Identity(Transform):
    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__, {})

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor, LabelArray, LabelTensor, List[Union[ImageArray, ImageTensor, LabelArray, LabelTensor]]],
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        return_grid: bool = False,  # Return a grid or list of grids as the final element.
        ) -> Union[ImageArray, ImageTensor, List[Union[ImageArray, ImageTensor, Union[ImageGrid, List[ImageGrid]]]]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_matched=True)
        sizes = [i.shape[-self._dim:] for i in images]
        spacings = arg_to_list(spacing, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(images))
        origins = arg_to_list(origin, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(images))

        image_ts = images

        # 'return_grid' just adds a list of grids at the end (or single if only one image)
        if return_grid:
            grid_ts = [(s, sp, o) for s, sp, o in zip(sizes, spacings, origins)]
            res = [image_ts[0], grid_ts[0]] if image_was_single else [*image_ts, grid_ts]
            return res
        else:
            return image_ts[0] if image_was_single else image_ts

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
            indices = to_tensor([], device=points.device, dtype=torch.int32) if isinstance(points, torch.Tensor) else np.array([])
            return points, indices 
        return points
