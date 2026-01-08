import torch
from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ....utils import *
from ...identity import Identity
from ...random import RandomTransform
from ...transform import Transform
from .affine import Affine, RandomAffine

# translate:
# - t=5 -> t=((-5, 5), (-5, 5)) for 2D, t=((-5, 5), (-5, 5), (-5, 5)) for 3D.
# - t=(-3, 5) -> t=(-3, 5, -3, 5) for 2D, t=(-3, 5, -3, 5, -3, 5) for 3D.
class RandomTranslate(RandomAffine):
    def __init__(
        self, 
        translate_range: Union[Number, Tuple[Number, ...]] = 100.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        trans_range = expand_range_arg(translate_range, dim=self._dim, negate_lower=True)
        assert len(trans_range) == 2 * self._dim, f"Expected 'translate_range' of length {2 * self._dim}, got {len(trans_range)}."
        self.__trans_range = to_tensor(trans_range).reshape(self._dim, 2)
        self._params = dict(
            dim=self._dim,
            p=self._p,
            translate_range=self.__trans_range,
        )

    def freeze(self) -> 'Translate':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        trans_draw = draw * (self.__trans_range[:, 1] - self.__trans_range[:, 0]) + self.__trans_range[:, 0]
        params = dict(
            translate=trans_draw,
        )
        return super().freeze(Translate, params)
        
    def __str__(self) -> str:
        params = dict(
            translate_range=to_tuple(self.__trans_range.flatten()),
        )
        return super().__str__(self.__class__.__name__, params)

class Translate(Affine):
    def __init__(
        self,
        translate: Union[Number, Tuple[Number], np.ndarray, torch.Tensor],
        **kwargs) -> None:
        super().__init__(**kwargs)
        translate = arg_to_list(translate, (int, float), broadcast=self._dim)
        assert len(translate) == self._dim, f"Expected 'translate' of length {self._dim} for dim={self._dim}, got {len(translate)}."
        self.__translate = to_tensor(translate)
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            dim=self._dim,
            matrix=self.__matrix,
            translate=self.__translate,
        )

    # This is used for image resampling, not for point clouds.
    def back_transform_points(
        self,
        points: PointsTensor,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> PointsTensor:
        print('performing translate back transform points')

        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t

    # Defines the forward/backward transforms.
    def __create_transforms(self) -> None:
        self.__matrix = create_translation(self.__translate)
        self.__backward_matrix = create_translation(-self.__translate)

    def get_affine_back_transform(
        self,
        device: torch.device,
        **kwargs) -> torch.Tensor:

        print('getting translation back transform')

        # Get homogeneous matrix.
        return self.__backward_matrix.to(device)

    def get_affine_transform(
        self,
        device: torch.device,
        **kwargs) -> torch.Tensor:

        print('getting translation forward transform')

        # Get homogeneous matrix.
        return self.__matrix.to(device)

    def __str__(self) -> str:
        params = dict(
            translate=to_tuple(self.__translate),
        )
        return super().__str__(self.__class__.__name__, params)
