import torch
from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..mixins import AffineMixin, RandomAffineMixin, TransformImageMixin
from ..random import RandomTransform
from .affine import Affine
from .identity import Identity

# rotation:
# - r=5 -> r=(-5, 5, -5, 5) for 2D, r=(-5, 5, -5, 5, -5, 5) for 3D.
# - r=(-3, 5) -> r=(-3, 5, -3, 5) for 2D, r=(-3, 5, -3, 5, -3, 5) for 3D.
# Mixins: 
# RandomAffineMixin calls super().__init__() which means it continues up the 
# chain looking for __init__ methods. RandomAffineMixin parent is 'object', so
# the MRO (method resolution order) moves on to RandomTransform.
# So 'RandomAffineMixin' stuff after super() will actually get called last.
class RandomRotation(RandomAffineMixin, RandomTransform):
    def __init__(
        self, 
        rotation: Union[Number, Tuple[Number, ...]] = 15.0,
        centre: Union[Literal['centre'], PointArray, PointTensor] = 'centre',
        **kwargs) -> None:
        super().__init__(**kwargs)
        print('init random rotation transform')
        rot_range = expand_range_arg(rotation, dim=self._dim, negate_lower=True)
        assert len(rot_range) == 2 * self._dim, f"Expected 'rotation' of length {2 * self._dim}, got {len(rot_range)}."
        if isinstance(centre, tuple):
            assert len(centre) == self._dim, f"Rotation centre must have {self._dim} dimensions."
        self.__rot_range = to_tensor(rot_range).reshape(self._dim, 2)
        self.__centre = to_tensor(centre) if not centre == 'centre' else centre
        self._params = dict(
            centre=self.__centre,
            dim=self._dim,
            p=self._p,
            rotation_range=self.__rot_range,
        )

    def freeze(self) -> 'Rotation':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(self._dim))
        rot_draw = draw * (self.__rot_range[:, 1] - self.__rot_range[:, 0]) + self.__rot_range[:, 0]
        t_frozen = Rotation(rotation=rot_draw, centre=self.__centre)
        super().freeze(t_frozen)
        return t_frozen

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__rot_range.flatten())}, dim={self._dim}, p={self._p})"

class Rotation(AffineMixin, TransformImageMixin, Affine):
    def __init__(
        self,
        rotation: Union[Number, Tuple[Number], np.ndarray, torch.Tensor],
        centre: Union[Literal['centre'], Point, PointArray, PointTensor] = 'centre',
        **kwargs) -> None:
        super().__init__(**kwargs)
        rotation = arg_to_list(rotation, (int, float), broadcast=self._dim)
        assert len(rotation) == self._dim, f"Expected 'rotation' of length {self._dim} for dim={self._dim}, got {len(rotation)}."
        self.__rotation = to_tensor(rotation)
        self.__centre = 'centre' if centre == 'centre' else to_tensor(centre)
        self.__rotation_rad = torch.deg2rad(self.__rotation)
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            centre=self.__centre,
            dim=self._dim,
            matrix=self.__matrix,
            rotation=self.__rotation,
            rotation_rad=self.__rotation_rad,
        )

    # This is used for image resampling, not for point clouds.
    def back_transform_points(
        self,
        points: PointsTensor,
        size: Optional[SizeTensor] = None,
        spacing: Optional[SpacingTensor] = None,
        origin: Optional[PointTensor] = None,
        **kwargs) -> PointsTensor:
        print('performing rotation back transform points')

        # Get homogeneous matrix.
        matrix_a = self.get_affine_back_transform(points.device, size=size, spacing=spacing, origin=origin)

        # Transform points.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]
        return points_t

    # Defines the forward/backward transforms.
    def __create_transforms(self) -> None:
        self.__backward_matrix = create_rotation(self.__rotation_rad)
        self.__matrix = self.__backward_matrix.T     # Rotation matrix inverse is just transpose.

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
        size = to_tensor(size, device=device, dtype=torch.int32)
        spacing = to_tensor(spacing, device=device)
        origin = to_tensor(origin, device=device)

        print('getting rotation back transform')

        # Get centre of rotation.
        if self.__centre == 'centre':
            assert size is not None
            assert spacing is not None
            assert origin is not None
            size = to_tensor(size, device=device, dtype=torch.int32)
            spacing = to_tensor(spacing, device=device)
            origin = to_tensor(origin, device=device)
            grid = torch.stack([origin, origin + size * spacing]).to(device)
            rot_centre = grid.sum(axis=0) / 2
        else:
            rot_centre = self.__centre.to(device)

        # Get homogeneous matrix.
        trans_matrix = create_translation(-rot_centre, device=device)
        inv_trans_matrix = create_translation(rot_centre, device=device)
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
        size = to_tensor(size, device=device, dtype=torch.int32)
        spacing = to_tensor(spacing, device=device)
        origin = to_tensor(origin, device=device)

        print('getting rotation forward transform')

        # Get centre of rotation.
        if self.__centre == 'centre':
            assert size is not None
            assert spacing is not None
            assert origin is not None
            size = to_tensor(size, device=device, dtype=torch.int32)
            spacing = to_tensor(spacing, device=device)
            origin = to_tensor(origin, device=device)
            grid = torch.stack([origin, origin + size * spacing]).to(device)
            rot_centre = grid.sum(axis=0) / 2
        else:
            rot_centre = self.__centre.to(device)

        # Get homogeneous matrix.
        trans_matrix = create_translation(-rot_centre, device=device)
        inv_trans_matrix = create_translation(rot_centre, device=device)
        matrix_a = torch.linalg.multi_dot([inv_trans_matrix, self.__matrix.to(device), trans_matrix])
        return matrix_a

    def __str__(self) -> str:
        centre = "\"centre\"" if self.__centre == 'centre' else self.__centre
        return f"{self.__class__.__name__}({to_tuple(self.__rotation)}, centre={centre}, dim={self._dim})"

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
        origin = to_tensor(origin, device=points.device)
        size = to_tensor(size, device=points.device, dtype=torch.int32)
        spacing = to_tensor(spacing, device=points.device)

        # Get homogeneous matrix.
        matrix_a = self.get_affine_transform(points.device, size=size, spacing=spacing, origin=origin)

        # Perform forward transform.
        points_h = torch.hstack([points, create_ones((points.shape[0], 1), device=points.device)])  # Homogeneous coordinates.
        points_t_h = torch.linalg.multi_dot([matrix_a, points_h.T]).T
        points_t = points_t_h[:, :-1]

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        if filter_offgrid:
            assert origin is not None
            assert size is not None
            assert spacing is not None
            grid = torch.stack([origin, origin + size * spacing]).to(points.device)
            to_keep = (points_t >= grid[0]) & (points_t < grid[1])
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
