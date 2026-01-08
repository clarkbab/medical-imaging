from typing import *

from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..transform import Transform

class IntensityTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(
        self,
        class_name: str,
        params: Dict[str, str]) -> str:
        return super().__str__(class_name, params)

    @alias_kwargs([
        ('o', 'origin'),
        ('s', 'spacing'),
    ])
    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor, LabelArray, LabelTensor, List[Union[ImageArray, ImageTensor, LabelArray, LabelTensor]]],
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        return_grid: bool = False,
        ) -> Union[ImageArray, ImageTensor, List[Union[ImageArray, ImageTensor, Union[ImageGrid, List[ImageGrid]]]]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_matched=True)
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        origins = arg_to_list(origin, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(images))
        spacings = arg_to_list(spacing, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(images))
        images = [to_tensor(i) for i in images]
        devices = [i.device for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int32) for i in images]
        spacings = [to_tensor((1,) * self._dim, device=i.device) if s is None else to_tensor(s, device=i.device) for s, i in zip(spacings, images)]
        origins = [to_tensor((0,) * self._dim, device=i.device) if o is None else to_tensor(o, device=i.device) for o, i in zip(origins, images)]

        # Transform images.
        image_ts = []
        grid_ts = []
        for image, s, sp, o, rt in zip(images, sizes, spacings, origins, return_types):
            image_t = self.transform_intensity(image)

            # Convert to return types.
            grid_t = (s, sp, o)     # Grid isn't modified by IntensityTransforms.
            if rt == 'numpy': 
                image_t = to_array(image_t)
                if return_grid:
                    grid_t = tuple(to_array(g) for g in grid_t)
            image_ts.append(image_t)
            if return_grid:
                grid_ts.append(grid_t)

        # 'return_grid' just adds a list of grids at the end (or single if only one image)
        if return_grid:
            res = [image_ts[0], grid_ts[0]] if image_was_single else [*image_ts, grid_ts]
            return res
        else:
            return image_ts[0] if image_was_single else image_ts

    def transform_intensity(
        self,
        image: ImageTensor,
        **kwargs) -> ImageTensor:
        raise ValueError("Subclasses of 'IntensityTransform' must implement 'transform_intensity' method.")

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        filter_offgrid: bool = True,
        return_filtered: bool = False,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        if isinstance(points, np.ndarray):
            return_type = 'numpy'
        else:
            return_type = 'torch'

        if return_filtered:
            indices = np.array([]) if return_type == 'numpy' else to_tensor([], device=points.device)
            return points, indices
        else:
            return points
