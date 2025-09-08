import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..utils import *
from .mixins import TransformImageMixin, TransformMixin
from .random import RandomTransform
from .spatial import Flip, Rotation
from .transform import Transform

# Is Pipeline actually a transform?
# It should follow the API fairly closely.
# It has params - a list of the params for each transform.
class Pipeline(TransformImageMixin, TransformMixin, Transform):
    def __init__(
        self,
        transforms: List[Union[Transform]],
        freeze: Optional[Union[bool, List[bool]]] = False,
        random_seed: Optional[Union[int, List[int]]] = None) -> None:
        freezes = arg_to_list(freeze, (None, bool), broadcast=len(transforms))
        random_seeds = arg_to_list(random_seed, (None, int), broadcast=len(transforms))
        assert len(random_seeds) == len(transforms), "Random seeds ('random_seed') must have same length as 'transforms'."
        dim = transforms[0].dim
        for t in transforms:
            assert t.dim == dim, "All transforms must have same 'dim'."
        # Can be a combination of random and deterministic transforms.
        # For example, we may always want to centre crop first, or extract patches last,
        # with random augmentations in the middle.
        self._dim = dim

        # Reseed the random transforms if requested - just easier doing it during pipeline creation rather than
        # for each transform.
        [t.seed_rng(random_seed=s) for s, t in zip(random_seeds, transforms) if s is not None and isinstance(t, RandomTransform)]

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
        print(type(points_t))

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
            okwargs = dict(
                origin=origin,
                size=size,
                spacing=spacing,
            )
            # Store any homogeneous multiplications for later.
            if t.is_homogeneous:
                print(f'adding transform {i} to chain')
                t_back = t.get_homogeneous_back_transform(device=points_t.device, **okwargs)
                chain.insert(0, t_back)

            # Resolve any stored chains.
            if not t.is_homogeneous or i == len(self.__transforms) - 1:
                if len(chain) > 0:
                    print(f'performing chained muliplication for {len(chain)} transforms')
                    # TODO: points should be in homogeneous coordinates.h
                    points_t_h = torch.hstack([points_t, create_ones((points_t.shape[0], 1), device=points_t.device)])  # Move to homogeneous coords.
                    points_t_h = torch.linalg.multi_dot(chain + [points_t_h.T]).T
                    points_t = points_t_h[:, :-1]
                    chain = []

            # Perform non-homogeneous transform.
            if not t.is_homogeneous:
                print(f'performing non-homogeneous transform {i}')
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__transforms})"

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
        for t in self.__transforms:
            okwargs = dict(
                origin=origin,
                size=size,
                spacing=spacing,
            )
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
