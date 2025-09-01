from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..mixins import TransformImageMixin, TransformMixin
from ..random import RandomTransform
from .identity import IdentityTransform
from .spatial import SpatialTransform

class RandomFlip(RandomTransform):
    def __init__(
        self,
        p_flip: Union[float, Tuple[float]] = 0.5,
        dim: int = 3,
        p: float = 1.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert dim in [2, 3], "Only 2D and 3D flips are supported."
        self._dim = dim
        p_flips = arg_to_list(p_flip, float, broadcast=dim)
        assert len(p_flips) == dim, f"Expected 'p_flip' of length {dim} for dim={dim}, got {len(p_flips)}."
        self.__p_flips = to_tensor(p_flips, dtype=torch.bool)
        self.__p = p
        self._params = dict(
            dim=self._dim,
            p=self.__p,
            p_flips=self.__p_flips,
        )

    def freeze(self) -> 'Flip':
        should_apply = self._rng.random() < self.__p
        if not should_apply:
            return IdentityTransform()
        draw = to_tensor(self._rng.random(self._dim))
        should_flip = draw < self.__p_flips
        return Flip(flip=should_flip, dim=self._dim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__p_flips)}, dim={self._dim}, p={self.__p})"
        
class Flip(TransformImageMixin, TransformMixin, SpatialTransform):
    def __init__(
        self,
        flip: Union[bool, Tuple[bool], np.ndarray, torch.Tensor],
        dim: int = 3) -> None:
        self._dim = dim
        self._is_homogeneous = True
        flips = arg_to_list(flip, bool, broadcast=dim)
        assert len(flips) == dim, f"Expected 'flip' of length {dim} for dim={dim}, got {len(flips)}."
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
        matrix_h = self.get_homogeneous_back_transform(device=points.device, size=size, spacing=spacing, origin=origin)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_h, points_h.T]).T
        points_t = points_t_h[:, :-1]
        if return_type == 'numpy':
            points_t = to_array(points_t)
        return points_t

    def __create_transforms(self) -> None:
        if self._dim == 2:
            # 2D flip matrix.
            self.__matrix = to_tensor([
                [-1 if self.__flips[0] else 1, 0, 0],
                [0, self.__flips[1], 0],
                [0, 0, 1],
            ])
        elif self._dim == 3:
            # 3D flip matrix.
            self.__matrix = to_tensor([
                [-1 if self.__flips[0] else 1, 0, 0, 0],
                [0, -1 if self.__flips[1] else 1, 0, 0],
                [0, 0, -1 if self.__flips[2] else 1, 0],
                [0, 0, 0, 1],
            ])

        # Flip matrix is it's own inverse.
        self.__backward_matrix = self.__matrix

    def get_homogeneous_back_transform(
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
        matrix_h = torch.linalg.multi_dot([inv_trans_matrix, self.__backward_matrix.to(device), trans_matrix])
        return matrix_h

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
        assert origin is not None
        assert size is not None
        assert spacing is not None
        size = to_tensor(size, device=points.device, dtype=torch.int)
        spacing = to_tensor(spacing, device=points.device)
        origin = to_tensor(origin, device=points.device)

        # Get FOV and centre.
        fov = torch.stack([origin, origin + size * spacing]).to(points.device)
        flip_centre = fov.sum(axis=0) / 2

        trans_matrix = create_translation(-flip_centre, device=points.device)
        inv_trans_matrix = create_translation(flip_centre, device=points.device)
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([inv_trans_matrix, self.__matrix.to(points.device), trans_matrix, points_h.T]).T
        points_t = points_t_h[:, :-1]

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        if filter_offscreen:
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
