from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ....utils import *
from ...identity import Identity
from .affine import Affine, RandomAffine

class RandomFlip(RandomAffine):
    def __init__(
        self,
        p_flip: Union[Number, Tuple[Number]] = 0.5,
        **kwargs) -> None:
        super().__init__(**kwargs)
        p_flip = arg_to_list(p_flip, (int, float), broadcast=self._dim)
        assert len(p_flip) == self._dim, f"Expected 'p_flip' of length {self._dim} for dim={self._dim}, got {len(p_flip)}."
        self.__p_flip = to_tensor(p_flip)
        self._params = dict(
            dim=self._dim,
            p=self._p,
            p_flip=self.__p_flip,
        )

    def freeze(self) -> 'Flip':
        should_apply = self._rng.random() < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        flip_draw = draw < self.__p_flip
        t_frozen = Flip(flip=flip_draw)
        params = dict(
            flip=flip_draw,
        )
        return super().freeze(Flip, params)

    def __str__(self) -> str:
        params = dict(
            p_flip=to_tuple(self.__p_flip),
        )
        return super().__str__(self.__class__.__name__, params)
        
class Flip(Affine):
    def __init__(
        self,
        flip: Union[bool, Tuple[bool], np.ndarray, torch.Tensor],
        # TODO: Add centre for flips about other points.
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.__flip = to_tensor(flip, broadcast=self._dim, dtype=torch.bool)
        assert len(self.__flip) == self._dim, f"Expected 'flip' of length {self._dim} for dim={self._dim}, got {len(self.__flip)}."
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            dim=self._dim,
            flip=self.__flip,
            matrix=self.__matrix,
        )

    def back_transform_points(
        self,
        points: PointsTensor,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> PointsTensor:
        print('performing flip back transform points')

        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device, size=size, spacing=spacing, origin=origin)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t

    def __create_transforms(self) -> None:
        scaling = tuple([-1 if f else 1 for f in self.__flip] + [1])
        self.__matrix = create_eye(self._dim + 1, scaling=scaling)
        self.__backward_matrix = self.__matrix      # Flip matrix is it's own inverse.

    def get_affine_back_transform(
        self,
        device: torch.device,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> torch.Tensor:
        print('getting flip back transform')

        # Get flip centre.
        if self.__centre == 'centre':
            raise ValueError(f"Grid params (size/spacing/origin) are required when performing flips around image centre (centre='centre').")
            grid = torch.stack([origin, size * spacing + origin]).to(device)
            flip_centre = grid.sum(axis=0) / 2
        else:
            flip_centre = self.__centre.to(device)

        # Get homogeneous matrix.
        trans_matrix = create_translation(-flip_centre, device=device)
        inv_trans_matrix = create_translation(flip_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__backward_matrix.to(device), trans_matrix])
        return matrix_a

    def get_affine_transform(
        self,
        device: torch.device,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> torch.Tensor:
        print('getting flip forward transform')

        # Get flip centre.
        if self.__centre == 'centre':
            if size is None or spacing is None or origin is None:
                raise ValueError(f"Grid params (size/spacing/origin) are required when performing flips around image centre (centre='centre').")
            grid = torch.stack([origin, size * spacing + origin]).to(device)
            flip_centre = grid.sum(axis=0) / 2
        else:
            flip_centre = self.__centre.to(device)

        # Get homogeneous matrix.
        trans_matrix = create_translation(-flip_centre, device=device)
        inv_trans_matrix = create_translation(flip_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__matrix.to(device), trans_matrix])
        return matrix_a

    def __str__(self) -> str:
        params = dict(
            flip=to_tuple(self.__flip),
        )
        return super().__str__(self.__class__.__name__, params)
