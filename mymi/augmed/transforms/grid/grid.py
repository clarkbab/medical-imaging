from typing import *

from mymi.typing import *

from ...utils import *
from ..transform import Transform

class GridTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def __str__(
        self,
        class_name: str,
        params: Dict[str, str]) -> str:
        return super().__str__(class_name, params)

    def transform_grid(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        raise ValueError("Subclasses of 'GridTransform' must implement 'transform_grid' method.")

    # Just removes voxels outside the transformed FOV.
    @alias_kwargs([
        ('o', 'origin'),
        ('rf', 'return_grid'),
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

        # Crop images.
        image_ts = []
        grid_ts = []
        for image, dim, sz, sp, o, dev, rt in zip(images, dims, sizes, spacings, origins, devices, return_types):
            # Get new FOV.
            grid_t = self.transform_grid(sz, sp, o)

            # Get resample points.
            points = grid_points(*grid_t)
            points_t = to_tensor(points, device=image.device)

            # Reshape to image size.
            size_t, _, _ = grid_t
            points_t = points_t.reshape(*to_tuple(size_t), self._dim)

            # Perform resample.
            print('grid resample')
            print(image.dtype)
            image_t = grid_sample(image, sp, o, points_t)
            print(image_t.dtype)

            # Convert to return types.
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
