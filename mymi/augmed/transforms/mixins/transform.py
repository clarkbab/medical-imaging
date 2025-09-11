import torch
from typing import *

from mymi.typing import *
from mymi.utils import alias_kwargs, arg_to_list

from ...utils import *

# Why are these methods included using 'mixins' and not subclassing?
# This is because not all transforms want to use 'back_transform_points'
# (e.g. identity), or they may need to call other things before calling 'back_transform_points'
# (e.g. random transforms - need to call freeze). Overriding seems a bit messy:
# Here's a method, but actually, write your own. Mixins avoid this problem - only
# classes that need the methods get them.

# Implementing classes must have 'transform_image' and 'transform_points'. 
class TransformMixin:
    @alias_kwargs([
        ('o', 'origin'),
        ('s', 'spacing'),
    ])
    # Can pass a single array/tensor or a list of arrays/tensors.
    # Points arrays/tensors are inferred by their Nx2/3 shape. It's unlikely that images of this size will
    # be passed, but it would break.
    # Labels are inferred by the data type of the passed array/tensor (bool) and will be returned
    # in boolean type.
    # Will return a single transformed array/tensor or list of arrays/tensors.
    # If a single spacing/origin/size is passed, this is broadcast to all image arrays/tensors,
    # other
    def transform(
        self,
        data: Union[ImageArray, ImageTensor, LabelArray, LabelTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, LabelArray, LabelTensor, PointsArray, PointsTensor]]],
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        # This comes last because it can be inferred via adjacent images.
        size: Optional[Union[Size, SizeArray, SizeTensor, List[Union[Size, SizeArray, SizeTensor]]]] = None,
        filter_offscreen: bool = True) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        datas, data_was_single = arg_to_list(data, (np.ndarray, torch.Tensor), return_matched=True)
        sizes = arg_to_list(size, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(datas))
        spacings = arg_to_list(spacing, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(datas))
        origins = arg_to_list(origin, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(datas))

        # Infer data types.
        image_indices = []
        points_indices = []
        data_types = {}
        for i, d in enumerate(datas):
            if d.shape[-1] == 2 or d.shape[-1] == 3:
                points_indices.append(i)
                data_types[i] = 'points'
            else:
                image_indices.append(i)
                data_types[i] = 'image'

        # Infer sizes for offscreen point filtering.
        if filter_offscreen:
            for i in points_indices:
                if sizes[i] is None:
                    # Infer size from images - must all have same shape.
                    image_sizes = [datas[j].shape[-self._dim:] for j in image_indices]
                    if len(image_sizes) > 0 and np.unique(image_sizes, axis=0).shape[0] == 1:
                        sizes[i] = tuple(image_sizes[0])

        # Transform images.
        images, image_spacings, image_origins = [datas[i] for i in image_indices], [spacings[i] for i in image_indices], [origins[i] for i in image_indices]
        images_ts = self.transform_image(images, spacing=image_spacings, origin=image_origins)

        # Transform points.
        points, points_sizes, points_spacings, points_origins = [datas[i] for i in points_indices], [sizes[i] for i in points_indices], [spacings[i] for i in points_indices], [origins[i] for i in points_indices]
        points_ts = []
        for p, si, sp, o in zip(points, points_sizes, points_spacings, points_origins):
            if size is None:
                filter_offscreen = False    # Only filter if 'size' was passed or inferred from image sizes.
            points_t = self.transform_points(p, filter_offscreen=filter_offscreen, size=si, spacing=sp, origin=o)
            points_ts.append(points_t)

        # Flatten results.
        datas_t = []
        image_i, points_i = 0, 0
        for i in range(len(datas)):
            if data_types[i] == 'image':
                datas_t.append(images_ts[image_i])
                image_i += 1
            else:
                datas_t.append(points_ts[points_i])
                points_i += 1

        return datas_t[0] if data_was_single else datas_t

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor, LabelArray, LabelTensor],
        **kwargs) -> Union[ImageArray, ImageTensor, LabelArray, LabelTensor]:
        raise ValueError("Classes with 'TransformMixin' must implement 'transform_image' method.")

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Classes with 'TransformMixin' must implement 'transform_points' method.")

class TransformImageMixin:
    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Classes with 'TransformImageMixin' must implement 'back_transform_points' method.")

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
        return_dtypes = ['bool' if isinstance(i, np.ndarray) and i.dtype == np.bool_ or isinstance(i, torch.Tensor) and i.dtype == torch.bool else 'float' for i in images]
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
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int) for i in images]
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
            points_mm = image_points(image.shape, origin=origin, spacing=spacing)
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
        for i, (image, dim, s, sp, o, dev, r, rd) in enumerate(zip(images, dims, sizes, spacings, origins, devices, return_types, return_dtypes)):
            # Get resample points.
            points_mm_t = group_points_mm_ts[image_groups[i]].to(dev)

            # Reshape to image size.
            points_mm_t = points_mm_t.reshape(*to_tuple(s), self._dim)

            # Perform resample.
            image_t = grid_sample(image, points_mm_t, spacing=sp, origin=o)
            image_ts.append(image_t)

        if image_was_single:
            return image_ts[0]
        else:
            return image_ts
