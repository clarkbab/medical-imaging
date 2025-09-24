import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..utils import *
from .grid import GridTransform
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
        [t.set_random_seed(s) for s, t in zip(random_seeds, transforms) if s is not None and isinstance(t, RandomTransform)]

        # Freeze transforms if requested.
        transforms = [t.freeze() if f and isinstance(t, RandomTransform) else t for f, t in zip(freezes, transforms)]

        self.__transforms = transforms

    def back_transform_points(
        self,
        points: PointsTensor,
        sizes: List[SizeTensor],
        spacings: List[SpacingTensor],
        origins: List[PointTensor],
        **kwargs) -> PointsTensor:
        # TODO: Allow sizes/spacings/origins to be None and run a forward pass of 'transform_grid' for these.
        # Will people want to use this API?
        points_t = points
        grid_ts = list(zip(sizes, spacings, origins))

        # Create chains of homogeneous matrix multiplications.
        # E.g. for flip and rotate, naively we could perform each separately by 
        # running points_t = matmul(T_2, R, T_1, points.T).T, where T_1 translates centre of rotation to origin,
        # R performs rotation, and T_2 reverses the initial translation, followed by
        # points_t = matmul(T_2, F, T_1, points_t.T).T where F flips along certain axes. Note that these are performed
        # in reverse order because it's the back transform. With this approach, we perform two large matrix
        # multiplications using 3xN points.T matrix.
        # A better approach is to pull out chains of homogeneous matrix multiplications and concatenate them
        # so that the points matrix is only used once (for each chain).
        affine_chain = []
        for i, (t, g) in enumerate(reversed(list(zip(self.__transforms, grid_ts)))):
            size_t, spacing_t, origin_t = g
            grid_args = dict(
                size=size_t,
                spacing=spacing_t,
                origin=origin_t
            )

            # Chain resolution conditions:
            # 1. Non-affine transform.
            # 2. Final transform.
            if isinstance(t, SpatialTransform):
                # Store any affine multiplications for later.
                if isinstance(t, Affine):
                    t_affine = t.get_affine_back_transform(points_t.device, **grid_args)
                    affine_chain.insert(0, t_affine)
                else:
                    # Resolve chain.
                    if len(affine_chain) > 0:
                        points_t = self.resolve_chain(points_t, affine_chain)
                        affine_chain = []

                    # Perform current transform.
                    points_t = t.back_transform_points(points_t, **grid_args)

            # Resolve if final round.
            if i == len(self.__transforms) - 1 and len(affine_chain) > 0:
                points_t = self.resolve_chain(points_t, affine_chain)

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

    def resolve_chain(
        self,
        points: PointsTensor,
        chain: List[Affine]) -> PointsTensor:
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Move to homogeneous coords.
        points_h_t = torch.linalg.multi_dot(chain + [points_h.T]).T
        points_t = points_h_t[:, :-1]
        return points_t

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__transforms})"

    def transform_grid(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        size_t, spacing_t, origin_t = size.clone(), spacing.clone(), origin.clone()
        for t in self.__transforms:
            if isinstance(t, GridTransform):
                size_t, spacing_t, origin_t = t.transform_grid(size_t, spacing_t, origin_t)
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
        return_grid: bool = False) -> Union[ImageArrayWithFov, ImageTensorWithFov, List[Union[ImageArrayWithFov, ImageTensorWithFov]]]:
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
        group_grid_ts = []
        for g in groups:
            print('image group: ', g)

            # Calculate intermediate grid params - these are required for certain transforms, 
            # e.g. flip/rotate, which need the transform centre.
            image, size_t, spacing_t, origin_t = images[g], sizes[g], spacings[g], origins[g]
            size_ts, spacing_ts, origin_ts = [], [], []
            for t in self.__transforms:
                size_ts.append(size_t)
                spacing_ts.append(spacing_t)
                origin_ts.append(origin_t)
                if isinstance(t, GridTransform):
                    size_t, spacing_t, origin_t = t.transform_grid(size_t, spacing_t, origin_t)

            # Get final grid points.
            points_mm = grid_points(size_t, spacing_t, origin_t).to(image.device)

            # Perform back transform of resampling points.
            points_mm_t = self.back_transform_points(points_mm, size_ts, spacing_ts, origin_ts)

            # Append group results.
            group_points_mm_ts.append(points_mm_t)
            group_grid_ts.append((size_t, spacing_t, origin_t))

        # Resample images.
        image_ts = []
        for i, (image, dim, sz, sp, o, dev, rt) in enumerate(zip(images, dims, sizes, spacings, origins, devices, return_types)):
            # Get resample points.
            points_mm_t = group_points_mm_ts[image_groups[i]].to(dev)

            # Reshape to image size.
            grid_t = group_grid_ts[image_groups[i]]
            grid_t = tuple(f.to(dev) for f in grid_t)
            size_t, spacing_t, origin_t = grid_t
            points_mm_t = points_mm_t.reshape(*to_tuple(size_t), self._dim)

            # Perform resample.
            image_t = grid_sample(image, points_mm_t, spacing=sp, origin=o)

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_array(image_t)
                if return_grid:
                    grid_t = tuple(to_array(f) for f in grid_t)
            if return_grid:
                image_ts.append((image_t, grid_t))
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
        filter_offgrid: bool = True,
        return_filtered: bool = False,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        size = to_tensor(size, device=points.device, dtype=torch.int32)
        spacing = to_tensor(spacing, device=points.device)
        origin = to_tensor(origin, device=points.device)

        # Chain 'transform_points' calls for SpatialTransforms.
        grid_t = (size, spacing, origin)
        points_t = points
        affine_chain = []   # Resolve chains of 4x4 affines before applying to large Nx4 points matrix.
        for i, t in enumerate(self.__transforms):
            # Get current FOV, transform might need e.g. for centre of image for flip/crop/rotate.
            size_t, spacing_t, origin_t = grid_t

            if isinstance(t, GridTransform):
                # GridTransforms don't move points/objects.
                grid_t = t.transform_grid(*grid_t)
            elif isinstance(t, SpatialTransform):
                # SpatialTransforms don't affect the grid.
                okwargs = dict(
                    size=size_t,
                    spacing=spacing_t,
                    origin=origin_t,
                )
                if isinstance(t, Affine):
                    # Store affine for later.
                    t_affine = t.get_affine_transform(points_t.device, **okwargs)
                    affine_chain.append(t_affine)
                else:
                    # Resolve chain.
                    if len(affine_chain) > 0:
                        points_t = self.resolve_chain(points_t, affine_chain)
                        affine_chain = []

                    # Perform current transform.
                    points_t = t.transform_points(points_t, filter_offgrid=False, **okwargs)

            # Resolve if final round.
            if i == len(self.__transforms) - 1 and len(affine_chain) > 0:
                points_t = self.resolve_chain(points_t, affine_chain)

        if filter_offgrid:
            size_t, spacing_t, origin_t = grid_t
            fov = torch.stack([origin_t, origin_t + size_t * spacing_t]).to(points.device)
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
