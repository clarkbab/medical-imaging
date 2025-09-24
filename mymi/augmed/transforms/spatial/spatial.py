from typing import *

from mymi.typing import *

from ...utils import *
from ..transform import Transform

class SpatialTransform(Transform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Subclasses of 'SpatialTransform' must implement 'back_transform_points' method.")

    def __str__(
        class_name: str,
        **kwargs) -> str:
        return super().__str__(class_name, **kwargs)

    @alias_kwargs([
        ('o', 'origin'),
        ('s', 'spacing'),
    ])
    # This function should accept lists of images/spacings/origins.
    # If two images are in the same coordinate system (group), we should only call
    # 'back_transform_points' a single time.
    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor, LabelArray, LabelTensor, List[Union[ImageArray, ImageTensor, LabelArray, LabelTensor]]],
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None) -> Union[ImageArray, ImageTensor, List[Union[ImageArray, ImageTensor]]]:
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

        # Group images by size/spacing/origin.
        groups = [0]
        image_groups = { 0: 0 }
        for i, (si, sp, o, d) in enumerate(zip(sizes[1:], spacings[1:], origins[1:], devices[1:])):
            for g in groups:
                g_si, g_sp, g_o = sizes[g].to(d), spacings[g].to(d), origins[g].to(d)
                if torch.all(si == g_si) and torch.all(sp == g_sp) and torch.all(o == g_o):
                    image_groups[i + 1] = g
                else:
                    groups.append(i + 1)
                    image_groups[i + 1] = i + 1

        # Get back transformed image points for all groups.
        group_points_mm_ts = []
        for g in groups:
            image, size, spacing, origin = images[g], sizes[g], spacings[g], origins[g]
            points_mm = grid_points(image.shape, origin=origin, spacing=spacing)
            points_mm = to_tensor(points_mm, device=image.device)

            # Perform back transform of resampling points.
            # Currently we pass all args to each transform and they can consume if they need.
            okwargs = dict(
                size=size,
                spacing=spacing,
                origin=origin,
            )
            points_mm_t = self.back_transform_points(points_mm, **okwargs)
            group_points_mm_ts.append(points_mm_t)

        # Resample images.
        image_ts = []
        for i, (image, dim, s, sp, o, dev, rt) in enumerate(zip(images, dims, sizes, spacings, origins, devices, return_types)):
            # Get resample points.
            points_mm_t = group_points_mm_ts[image_groups[i]].to(dev)

            # Reshape to image size.
            points_mm_t = points_mm_t.reshape(*to_tuple(s), self._dim)

            # Perform resample.
            image_t = grid_sample(image, points_mm_t, spacing=sp, origin=o)

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_array(image_t)
            if return_grid:
                grid_t = group_grid_ts[image_groups[i]]
                if rt == 'numpy':
                    grid_t = (to_array(f) for f in grid_t)
                image_ts.append((image_t, grid_t))
            else:
                image_ts.append(image_t)

        if image_was_single:
            return image_ts[0]
        else:
            return image_ts
