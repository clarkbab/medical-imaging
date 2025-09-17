from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..mixins import AffineMixin, RandomAffineMixin, TransformImageMixin, TransformMixin
from ..random import RandomTransform
from .identity import Identity
from .spatial import SpatialTransform

class RandomFlip(RandomAffineMixin, RandomTransform):
    def __init__(
        self,
        p_flip: Union[Number, Tuple[Number]] = 0.5,
        **kwargs) -> None:
        super().__init__(**kwargs)
        p_flips = arg_to_list(p_flip, (int, float), broadcast=self._dim)
        assert len(p_flips) == self._dim, f"Expected 'p_flip' of length {self._dim} for dim={self._dim}, got {len(p_flips)}."
        self._p_flips = to_tensor(p_flips, dtype=torch.bool)
        self._params = dict(
            dim=self._dim,
            p=self._p,
            p_flips=self._p_flips,
        )

    def freeze(self) -> 'Flip':
        should_apply = self._rng.random() < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        should_flip = draw < self._p_flips
        return Flip(flip=should_flip, dim=self._dim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self._p_flips)}, dim={self._dim}, p={self._p})"
        
# Methods are resolved left to right, so overridding mixins should appear first.
# Init methods are also run left to right, so mixins that override params, e.g. _is_affine
# should appear last.
class Flip(AffineMixin, TransformImageMixin, TransformMixin, SpatialTransform):
    def __init__(
        self,
        flip: Union[bool, Tuple[bool], np.ndarray, torch.Tensor],
        **kwargs) -> None:
        super().__init__(**kwargs)
        flips = arg_to_list(flip, bool, broadcast=self._dim)
        assert len(flips) == self._dim, f"Expected 'flip' of length {self._dim} for dim={self._dim}, got {len(flips)}."
        self.__flips = to_tensor(flips, dtype=torch.bool)
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            dim=self._dim,
            flips=self.__flips,
            matrix=self.__matrix,
            matrix_complete=False,      # Do matrices define the full transform?
            requires=['centre'],        # What information is needed at transform time?
        )

    def back_transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeArray, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor]] = None,
        **kwargs) -> PointsTensor:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'

        print('performing flip back transform points')

        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device, size=size, spacing=spacing, origin=origin)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        if return_type == 'numpy':
            points_t = to_array(points_t)
        return points_t

    def __create_transforms(self) -> None:
        scaling = tuple([-1 if f else 1 for f in self.__flips] + [1])
        self.__matrix = create_eye(self._dim, scaling=scaling)
        self.__backward_matrix = self.__matrix      # Flip matrix is it's own inverse.

    def get_affine_back_transform(
        self,
        device: torch.device,
        size: Optional[Union[Size, SizeArray, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor]] = None,
        **kwargs) -> torch.Tensor:
        # Get flip centre.
        assert size is not None
        assert spacing is not None
        assert origin is not None
        size = to_tensor(size, device=device, dtype=torch.int)
        spacing = to_tensor(spacing, device=device)
        origin = to_tensor(origin, device=device)

        print('getting flip back transform')

        # Get flip centre.
        fov = torch.stack([origin, origin + size * spacing]).to(device)
        flip_centre = fov.sum(axis=0) / 2

        # Get homogeneous matrix.
        trans_matrix = create_translation(-flip_centre, device=device)
        inv_trans_matrix = create_translation(flip_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__backward_matrix.to(device), trans_matrix])
        return matrix_a

    def get_affine_transform(
        self,
        device: torch.device,
        size: Optional[Union[Size, SizeArray, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor]] = None,
        **kwargs) -> torch.Tensor:
        # Get flip centre.
        assert size is not None
        assert spacing is not None
        assert origin is not None
        size = to_tensor(size, device=device, dtype=torch.int)
        spacing = to_tensor(spacing, device=device)
        origin = to_tensor(origin, device=device)

        print('getting flip forward transform')

        # Get flip centre.
        fov = torch.stack([origin, origin + size * spacing]).to(device)
        flip_centre = fov.sum(axis=0) / 2

        # Get homogeneous matrix.
        trans_matrix = create_translation(-flip_centre, device=device)
        inv_trans_matrix = create_translation(flip_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__matrix.to(device), trans_matrix])
        return matrix_a

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__flips)}, dim={self._dim})"

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

        # Get homogeneous matrix.
        matrix_a = self.get_affine_transform(points.device, size=size, spacing=spacing, origin=origin)

        # Perform forward transform.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
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
