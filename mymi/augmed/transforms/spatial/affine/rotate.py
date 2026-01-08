import torch
from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ....utils import *
from ...identity import Identity
from ...random import RandomTransform
from .affine import Affine, RandomAffine

# Rotate:
# - r=5 -> r=(-5, 5, -5, 5) for 2D, r=(-5, 5, -5, 5, -5, 5) for 3D.
# - r=(-3, 5) -> r=(-3, 5, -3, 5) for 2D, r=(-3, 5, -3, 5, -3, 5) for 3D.
class RandomRotate(RandomAffine):
    def __init__(
        self, 
        rotate_range: Union[Number, Tuple[Number, ...]] = 15.0,
        centre: Union[Literal['centre'], PointArray, PointTensor] = 'centre',
        # TODO: Add 'centre_offset' to allow some random displacement of the centre point.
        **kwargs) -> None:
        super().__init__(**kwargs)
        print('init random rotate transform')
        rot_range = expand_range_arg(rotate_range, dim=self._dim, negate_lower=True)
        assert len(rot_range) == 2 * self._dim, f"Expected 'rotate_range' of length {2 * self._dim}, got {len(rot_range)}."
        if isinstance(centre, tuple):
            assert len(centre) == self._dim, f"Rotate centre must have {self._dim} dimensions."
        self.__rot_range = to_tensor(rot_range).reshape(self._dim, 2)
        # TODO: Convert 'centre' to a range.
        self.__centre = to_tensor(centre) if not centre == 'centre' else centre
        self._params = dict(
            centre=self.__centre,
            dim=self._dim,
            p=self._p,
            rotate_range=self.__rot_range,
        )

    def freeze(self) -> 'Rotate':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        rot_draw = draw * (self.__rot_range[:, 1] - self.__rot_range[:, 0]) + self.__rot_range[:, 0]
        params = dict(
            rotate=rot_draw,
            centre=self.__centre,
        )
        return super().freeze(Rotate, params)

    def __str__(self) -> str:
        params = dict(
            centre=to_tuple(self.__centre) if self.__centre != 'centre' else "\"centre\"",
            rotate_range=to_tuple(self.__rot_range.flatten()),
        )
        return super().__str__(self.__class__.__name__, params)

class Rotate(Affine):
    def __init__(
        self,
        rotate: Union[Number, Tuple[Number], np.ndarray, torch.Tensor],
        centre: Union[Literal['centre'], Point, PointArray, PointTensor] = 'centre',
        **kwargs) -> None:
        super().__init__(**kwargs)
        rotate = arg_to_list(rotate, (int, float), broadcast=self._dim)
        assert len(rotate) == self._dim, f"Expected 'rotate' of length {self._dim} for dim={self._dim}, got {len(rotate)}."
        self.__rotate = to_tensor(rotate)
        self.__centre = 'centre' if centre == 'centre' else to_tensor(centre)
        self.__rotate_rad = torch.deg2rad(self.__rotate)
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            centre=self.__centre,
            dim=self._dim,
            matrix=self.__matrix,
            rotate=self.__rotate,
            rotate_rad=self.__rotate_rad,
        )

    # This is used for image resampling, not for point clouds.
    def back_transform_points(
        self,
        points: PointsTensor,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> PointsTensor:
        print('performing rotate back transform points')

        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device, size=size, spacing=spacing, origin=origin)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t

    # Defines the forward/backward transforms.
    def __create_transforms(self) -> None:
        self.__backward_matrix = create_rotation(self.__rotate_rad)
        self.__matrix = self.__backward_matrix.T     # Rotation matrix inverse is just transpose.

    def get_affine_back_transform(
        self,
        device: torch.device,
        size: Optional[Union[Size, SizeArray, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor]] = None,
        **kwargs) -> torch.Tensor:
        print('getting rotate back transform')

        # Get centre of rotation.
        if self.__centre == 'centre':
            if size is None or spacing is None or origin is None:
                raise ValueError(f"Grid params (size/spacing/origin) are required when performing rotations around image centre (centre='centre').")
            grid = torch.stack([origin, origin + size * spacing]).to(device)
            rot_centre = grid.sum(axis=0) / 2
        else:
            rot_centre = self.__centre.to(device)

        # Get homogeneous matrix.
        print('rotation centre: ', rot_centre)
        trans_matrix = create_translation(-rot_centre, device=device)
        inv_trans_matrix = create_translation(rot_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__backward_matrix.to(device), trans_matrix])
        return matrix_a

    def get_affine_transform(
        self,
        device: torch.device,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> torch.Tensor:
        print('getting rotation forward transform')

        # Get centre of rotation.
        if self.__centre == 'centre':
            if size is None or spacing is None or origin is None:
                raise ValueError(f"Grid params (size/spacing/origin) are required when performing rotations around image centre (centre='centre').")
            grid = torch.stack([origin, size * spacing + origin]).to(device)
            rot_centre = grid.sum(axis=0) / 2
        else:
            rot_centre = self.__centre.to(device)

        # Get homogeneous matrix.
        trans_matrix = create_translation(-rot_centre, device=device)
        inv_trans_matrix = create_translation(rot_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__matrix.to(device), trans_matrix])
        return matrix_a

    def __str__(self) -> str:
        params = dict(
            centre=to_tuple(self.__centre) if self.__centre != 'centre' else "\"centre\"",
            rotate=to_tuple(self.__rotate),
        )
        return super().__str__(self.__class__.__name__, params)
