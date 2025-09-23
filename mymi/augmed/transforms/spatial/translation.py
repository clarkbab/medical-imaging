import torch
from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..mixins import AffineMixin, RandomAffineMixin, TransformImageMixin
from ..random import RandomTransform
from ..transform import Transform
from .affine import Affine
from .identity import Identity

# translation:
# - t=5 -> t=((-5, 5), (-5, 5)) for 2D, t=((-5, 5), (-5, 5), (-5, 5)) for 3D.
# - t=(-3, 5) -> t=(-3, 5, -3, 5) for 2D, t=(-3, 5, -3, 5, -3, 5) for 3D.
class RandomTranslation(RandomAffineMixin, RandomTransform):
    def __init__(
        self, 
        translation: Union[Number, Tuple[Number, ...]] = 100.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        trans_range = expand_range_arg(translation, negate_lower=True, vals_per_dim=2)
        assert len(trans_range) == 2 * self._dim, f"Expected 'translation' of length {2 * self._dim}, got {len(trans_range)}."
        self.__trans_range = to_tensor(trans_range).reshape(self._dim, 2)
        self._params = dict(
            dim=self._dim,
            p=self._p,
            translation_range=self.__trans_range,
        )

    def freeze(self) -> 'Translation':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        trans_draw = draw * (self.__trans_range[:, 1] - self.__trans_range[:, 0]) + self.__trans_range[:, 0]
        return Translation(translation=trans_draw, dim=self._dim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__trans_range.flatten())}, dim={self._dim}, p={self._p})"

class Translation(AffineMixin, TransformImageMixin, Affine):
    def __init__(
        self,
        translation: Union[Number, Tuple[Number], np.ndarray, torch.Tensor],
        **kwargs) -> None:
        super().__init__(**kwargs)
        translation = arg_to_list(translation, (int, float), broadcast=self._dim)
        assert len(translation) == self._dim, f"Expected 'translation' of length {self._dim} for dim={self._dim}, got {len(translation)}."
        self.__translation = to_tensor(translation)
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            dim=self._dim,
            matrix=self.__matrix,
            translation=self.__translation,
        )

    # This is used for image resampling, not for point clouds.
    def back_transform_points(
        self,
        points: PointsTensor,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> PointsTensor:
        print('performing translation back transform points')

        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t

    # Defines the forward/backward transforms.
    def __create_transforms(self) -> None:
        self.__matrix = create_translation(self.__translation)
        self.__backward_matrix = create_translation(-self.__translation)

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
        return f"{self.__class__.__name__}({to_tuple(self.__translation)}, dim={self._dim})"

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
        origin = to_tensor(origin, device=points.device)
        size = to_tensor(size, device=points.device, dtype=torch.int)
        spacing = to_tensor(spacing, device=points.device)

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
                points_t, indices = points_t.numpy(), indices.numpy()
            if return_filtered:
                return points_t, indices
            else:
                return points_t
        else:
            if return_type == 'numpy':
                points_t = points_t.numpy()
            return points_t
