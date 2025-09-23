import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..utils import *
from .fov import FovTransform
from .mixins import TransformImageMixin
from .random import RandomTransform
from .spatial import Affine, SpatialTransform
from .transform import Transform

# Is Pipeline actually a transform?
# It should follow the API fairly closely.
# It has params - a list of the params for each transform.
class Pipeline(TransformImageMixin, Transform):
    def __init__(
        self,
        transforms: List[Union[Transform]],
        # What's the thinking re 'dim'?
        # Same as 'random_seed'. Allow us to override here, rather than setting in each transform's constructor.
        # Allows, very easy setting of dim=2 for a pipeline.
        dim: Optional[int] = None,
        freeze: Optional[Union[bool, List[bool]]] = False,
        # What's the thinking around 'random_seed'?
        # If set here, override anything set in the transforms constructor. Allows easy setting of seeds in one place.
        random_seed: Optional[Union[int, List[int]]] = None) -> None:
        freezes = arg_to_list(freeze, (None, bool), broadcast=len(transforms))
        random_seeds = arg_to_list(random_seed, (None, int), broadcast=len(transforms))
        assert len(random_seeds) == len(transforms), "Random seeds ('random_seed') must have same length as 'transforms'."
        if dim is not None:
            assert dim in [2, 3], "Only 2D and 3D pipelines are supported."
            [t.set_dim(dim) for t in transforms]
        else:
            dim = transforms[0].dim
            for t in transforms:
                assert t.dim == dim, "All transforms must have same 'dim'."
        # Can be a combination of random and deterministic transforms.
        # For example, we may always want to centre crop first, or extract patches last,
        # with random augmentations in the middle.
        self._dim = dim

        # Reseed the random transforms if requested - just easier doing it during pipeline creation rather than
        # for each transform.
        [t.seed(random_seed=s) for s, t in zip(random_seeds, transforms) if s is not None and isinstance(t, RandomTransform)]

        # Freeze transforms if requested.
        transforms = [t.freeze() if f and isinstance(t, RandomTransform) else t for f, t in zip(freezes, transforms)]

        self.__transforms = transforms

    def back_transform_points(
        self,
        points: PointsTensor,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> PointsTensor:
        # Call each transform's 'back_transform_points' in reverse order.
        # This is not the most efficient way to handle the matrix multiplication.
        # We could condense down 4x4 homogeneous matrices before applying to the 
        # Nx4 point matrix. Transforms that consist of homogeneous matrix multiplications
        # should have the option to return these transforms for 'pipeline'.
        points_t = points
        print('pipeline back transform')

        # Create chains of homogeneous matrix multiplications.
        # E.g. for flip and rotate, naively we could perform each separately by 
        # running points_t = matmul(T_2, R, T_1, points.T).T, where T_1 translates centre of rotation to origin,
        # R performs rotation, and T_2 reverses the initial translation, followed by
        # points_t = matmul(T_2, F, T_1, points_t.T).T where F flips along certain axes. Note that these are performed
        # in reverse order because it's the back transform. With this approach, we perform two large matrix
        # multiplications using 3xN points.T matrix.
        # A better approach is to pull out chains of homogeneous matrix multiplications and concatenate them
        # so that the points matrix is only used once (for each chain).
        chain = []
        for i, t in enumerate(reversed(self.__transforms)):
            if not isinstance(t, SpatialTransform):
                continue

            okwargs = dict(
                origin=origin,
                size=size,
                spacing=spacing,
            )
            # Store any affine multiplications for later.
            if isinstance(t, Affine):
                t_back = t.get_affine_back_transform(points_t.device, **okwargs)
                chain.insert(0, t_back)

            # Resolve any stored chains.
            if not isinstance(t, Affine) or i == len(self.__transforms) - 1:
                if len(chain) > 0:
                    points_t_h = torch.hstack([points_t, create_ones((points_t.shape[0], 1), device=points_t.device)])  # Move to homogeneous coords.
                    points_t_h = torch.linalg.multi_dot(chain + [points_t_h.T]).T
                    points_t = points_t_h[:, :-1]
                    chain = []

            # Perform non-affine transform.
            if not isinstance(t, Affine):
                points_t = t.back_transform_points(points_t, **okwargs)

        return points_t

    # Freeze should return a new 'Pipeline' object.
    # For example, we may want to freeze many samples from a single pipeline with
    # random transforms.
    def freeze(self) -> 'Pipeline':
        transforms = [t.freeze() if isinstance(t, RandomTransform) else t for t in self.__transforms]
        # Remove identity transforms?
        return Pipeline(transforms)

    def __getitem__(
        self,
        i: int) -> Transform:
        return self.__transforms[i]

    @property
    def params(self) -> Dict[str, Any]:
        return dict((i, t.params) for i, t in enumerate(self.__transforms))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__transforms})"

    # Not forward-facing, just accept tensors.
    def transform_fov(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        size_t, spacing_t, origin_t = size.clone(), spacing.clone(), origin.clone()
        for t in self.__transforms:
            if isinstance(t, FovTransform):
                size_t, spacing_t, origin_t = t.transform_fov(size_t, spacing_t, origin_t)
        return size_t, spacing_t, origin_t

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
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        return_fov: bool = False) -> Union[ImageArrayWithFov, ImageTensorWithFov, List[Union[ImageArrayWithFov, ImageTensorWithFov]]]:
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
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int) for i in images]
        spacings = [to_tensor((1,) * self._dim, device=i.device) if s is None else to_tensor(s, device=i.device) for s, i in zip(spacings, images)]
        origins = [to_tensor((0,) * self._dim, device=i.device) if o is None else to_tensor(o, device=i.device) for o, i in zip(origins, images)]

        # Group images by size/spacing/origin.
        image_groups = { 0: 0 }     # Maps images (by index) to a group.
        groups = [0]    # Tracks groups (by index of first image).
        for i, (si, sp, o, d) in enumerate(zip(sizes[1:], spacings[1:], origins[1:], devices[1:])):
            j = i + 1
            for k, g in enumerate(groups):
                g_si, g_sp, g_o = sizes[g].to(d), spacings[g].to(d), origins[g].to(d)
                if torch.all(si == g_si) and torch.all(sp == g_sp) and torch.all(o == g_o):
                    # Add to existing group.
                    image_groups[j] = g
                elif k == len(groups) - 1:
                    # Create new group.
                    groups.append(j)
                    image_groups[j] = j

        # Get back transformed image points for all groups.
        group_points_mm_ts = []
        group_fov_ts = []
        for g in groups:
            # Get the final image fov.
            image, size, spacing, origin = images[g], sizes[g], spacings[g], origins[g]
            size_t, spacing_t, origin_t = self.transform_fov(size, spacing, origin)
            points_mm = image_points(size_t, spacing_t, origin_t)
            points_mm = to_tensor(points_mm, device=image.device)

            # Perform back transform of resampling points.
            # Currently we pass all args to each transform and they can consume if they need.
            okwargs = dict(
                size=size,
                spacing=spacing,
                origin=origin,
            )
            points_mm_t = self.back_transform_points(points_mm, **okwargs)

            # Append group results.
            group_points_mm_ts.append(points_mm_t)
            group_fov_ts.append((size_t, spacing_t, origin_t))

        # Resample images.
        image_ts = []
        for i, (image, dim, sz, sp, o, dev, rt) in enumerate(zip(images, dims, sizes, spacings, origins, devices, return_types)):
            # Get resample points.
            points_mm_t = group_points_mm_ts[image_groups[i]].to(dev)

            # Reshape to image size.
            fov_t = group_fov_ts[image_groups[i]]
            fov_t = tuple(f.to(dev) for f in fov_t)
            size_t, spacing_t, origin_t = fov_t
            points_mm_t = points_mm_t.reshape(*to_tuple(size_t), self._dim)

            # Perform resample.
            image_t = grid_sample(image, points_mm_t, spacing=sp, origin=o)

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_array(image_t)
                if return_fov:
                    fov_t = tuple(to_array(f) for f in fov_t)
            if return_fov:
                image_ts.append((image_t, fov_t))
            else:
                image_ts.append(image_t)

        if image_was_single:
            return image_ts[0]
        else:
            return image_ts


    # This is for point clouds, not for image resampling. Note that this
    # requires invertibility of the back point transform, which may not be
    # be available for some transforms (e.g. folded elastic).
    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        filter_offscreen: bool = True,
        return_filtered: bool = False,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        size = to_tensor(size, device=points.device, dtype=torch.int)
        spacing = to_tensor(spacing, device=points.device)
        origin = to_tensor(origin, device=points.device)

        # Call each transform's 'transform_points'.
        points_t = points
        chain = []
        for i, t in enumerate(self.__transforms):
            okwargs = dict(
                origin=origin,
                size=size,
                spacing=spacing,
            )
            # Store any homogeneous multiplications for later.
            if t._is_affine:
                print('adding affine transform to chain: ', i)
                t_forward = t.get_affine_transform(points_t.device, **okwargs)
                chain.append(t_forward)

            # Resolve any stored chains.
            if not t._is_affine or i == len(self.__transforms) - 1:
                if len(chain) > 0:
                    print('resolving chained affine transforms: ', len(chain))
                    points_t_h = torch.hstack([points_t, create_ones((points_t.shape[0], 1), device=points_t.device)])  # Move to homogeneous coords.
                    points_t_h = torch.linalg.multi_dot(chain + [points_t_h.T]).T
                    points_t = points_t_h[:, :-1]
                    chain = []

            # Perform non-homogeneous transform.
            if not t._is_affine:
                print('performing non-affine transform: ', i)
                points_t = t.transform_points(points_t, filter_offscreen=False, **okwargs)

        if filter_offscreen:
            assert origin is not None
            assert size is not None
            assert spacing is not None
            fov = torch.stack([origin, origin + size * spacing]).to(points.device)
            to_keep = (points_t >= fov[0]) & (points_t < fov[1])
            to_keep = to_keep.all(axis=1)
            points_t = points_t[to_keep]
            indices = torch.where(to_keep)[0]
            if return_type == 'numpy':
                points_t, indices = to_array(points_t), to_array(indices)
            if return_filtered:
                return points_t, indices
            else:
                return points_t
        else:
            if return_type == 'numpy':
                points_t = to_array(points_t)
            return points_t

    @property
    def transforms(self) -> List[Union[Transform]]:
        return self.__transforms
