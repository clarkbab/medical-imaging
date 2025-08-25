import torch
from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ..transform import DetTransform, RandomTransform
from .identity import IdentityTransform
from .spatial import SpatialTransform

class RandomRotation(RandomTransform):
    def __init__(
        self, 
        rotation: Tuple[AngleDegree] = (-5, 5),
        centre: Union[Point, Literal['centre']] = 'centre',
        dim: int = 3,   # Setting this allows us to apply transform to 2-5D arrays/tensors and determine batch/channel/spatial dimensions.
        p: float = 1.0,  # What proportion of the time is a random rotation applied?
        ) -> None:
        assert dim in [2, 3], "Only 2D and 3D rotations are supported."
        if dim == 2:
            assert len(rotation) == 2, "2D rotation requires two angles."
        else:
            assert len(rotation) == 2 or len(rotation) == 6, "3D rotation requires either two or six angles."
        if isinstance(centre, tuple):
            assert len(centre) == dim, f"Rotation centre must have {dim} dimensions."
        if dim == 3 and len(rotation) == 2:
            rotation = rotation * 3
        self._dim = dim
        self.__rot_range = torch.Tensor(rotation).reshape(3, 2)
        self.__centre = centre
        self.__p = p

    def get_det_transform(
        self,
        random_seed: Optional[int] = None) -> SpatialTransform:
        if random_seed is not None:
            torch.manual_seed(random_seed)
        should_apply = torch.rand(1) < self.__p
        if not should_apply:
            return IdentityTransform()
        draw = torch.rand(self._dim)
        rot_draw = draw * (self.__rot_range[:, 1] - self.__rot_range[:, 0]) + self.__rot_range[:, 0]
        return RotationTransform(self._dim, rot_draw, centre=self.__centre)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({tuple(self.__rot_range.flatten().cpu().numpy())}, dim={self._dim}, p={self.__p})"

class RotationTransform(DetTransform, SpatialTransform):
    def __init__(
        self,
        dim: int,
        rotation: Union[torch.Tensor, Tuple[AngleDegree]],
        centre: Union[Point, Literal['centre']] = 'centre') -> None:
        self._dim = dim
        self.__rotation = torch.Tensor(rotation)
        self.__centre = centre
        self.__rotation_rad = torch.deg2rad(rotation)
        self.__create_transforms()
        self._params = dict(
            backward_matrix=self.__backward_matrix,
            centre=self.__centre,
            dim=self._dim,
            matrix=self.__matrix,
            matrix_complete=False,      # Do matrices define the full transform?
            requires=['centre'],        # What information is needed at transform time?
            rotation=self.__rotation,
            rotation_rad=self.__rotation_rad,
        )

    # This is used for image resampling, not for point clouds.
    def back_transform_points(
        self,
        points: Points,
        centre: Point,
        **kwargs) -> Points:
        if isinstance(points, np.ndarray):
            points = torch.Tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        logging.info(f"Back-rotating points around centre={centre}")
        centre = torch.Tensor(centre)
        trans_matrix = torch.eye(self._dim + 1)
        trans_matrix[:self._dim, self._dim] = -centre  # Translate to origin.
        inv_trans_matrix = torch.eye(self._dim + 1)
        inv_trans_matrix[:self._dim, self._dim] = centre  # Translate back to centre.
        points_h = torch.hstack([points, torch.ones((points.shape[0], 1))])  # Homogeneous coordinates.
        points_h_t = torch.linalg.multi_dot([inv_trans_matrix, self.__backward_matrix, trans_matrix, points_h.T]).T
        points_t = points_h_t[:, :-1]
        if return_type == 'numpy':
            points_t = points_t.numpy()
        return points_t

    # Defines the forward/backward transforms.
    def __create_transforms(self) -> None:
        # Standard rotation matrices are 'backward' as we want the forward transform
        # to be clockwise. Do we?
        # We use homogeneous coordinates for rotation matrices so they can be easily
        # multiplied with transforms that require homogeneous coords, e.g. translations.
        if self._dim == 2:
            # 2D rotation matrix.
            self.__backward_matrix = torch.Tensor([
                [torch.cos(self.__rotation_rad[0]), -torch.sin(self.__rotation_rad[0]), 0],
                [torch.sin(self.__rotation_rad[0]), torch.cos(self.__rotation_rad[0]), 0],
                [0, 0, 1]
            ])
        else:
            # 3D rotation matrix.
            self.__backward_x = torch.Tensor([
                [1, 0, 0, 0],
                [0, torch.cos(self.__rotation_rad[0]), -torch.sin(self.__rotation_rad[0]), 0],
                [0, torch.sin(self.__rotation_rad[0]), torch.cos(self.__rotation_rad[0]), 0],
                [0, 0, 0, 1]
            ])
            self.__backward_y = torch.Tensor([
                [torch.cos(self.__rotation_rad[1]), 0, torch.sin(self.__rotation_rad[1]), 0],
                [0, 1, 0, 0],
                [-torch.sin(self.__rotation_rad[1]), 0, torch.cos(self.__rotation_rad[1]), 0],
                [0, 0, 0, 1]
            ])
            self.__backward_z = torch.Tensor([
                [torch.cos(self.__rotation_rad[2]), -torch.sin(self.__rotation_rad[2]), 0, 0],
                [torch.sin(self.__rotation_rad[2]), torch.cos(self.__rotation_rad[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            self.__backward_matrix = torch.linalg.multi_dot([self.__backward_z, self.__backward_y, self.__backward_x])

        # Rotation matrix inverse is just the transpose.
        self.__matrix = self.__backward_matrix.T

    def __str__(self) -> str:
        centre = "\"centre\"" if self.__centre == 'centre' else self.__centre
        return f"{self.__class__.__name__}({tuple(self.__rotation.cpu().numpy())}, centre={centre}, dim={self._dim})"

    # We should let transformation parameters be passed in patient coords (mm) or voxel coords.
    # When passed in voxel coords, we'll need to know the image spacing/offset.
    @alias_kwargs([
        ('s', 'spacing'),
    ])
    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        spacing: Optional[Spacing] = None) -> Union[ImageArray, ImageTensor]:
        if isinstance(image, np.ndarray):
            image = torch.Tensor(image)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        image_shape = image.shape
        image_dim = len(image_shape)
        if self._dim == 2:
            assert image_dim in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {image_dim}D."
        elif self._dim == 3:
            assert image_dim in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {image_dim}D."
        if spacing is None:
            spacing = (1,) * self._dim

        # Get points in fixed image (voxel coords for now)
        image_spatial_shape = image.shape[-self._dim:]
        grids = torch.meshgrid([torch.arange(s) for s in image_spatial_shape], indexing='ij')
        points_vox = torch.stack(grids, dim=-1).reshape(-1, self._dim)
        points = points_vox * torch.Tensor(spacing)

        # Perform rotation.
        centre = torch.Tensor(fov_centre(image, offset=(0, 0, 0), spacing=spacing, use_patient_coords=True)) if self.__centre == 'centre' else self.__centre
        points_t = self.back_transform_points(points, centre)

        # Resample the image at the transformed points.
        image_fw_mm = fov_width(image, offset=(0, 0, 0), spacing=spacing, use_patient_coords=True)
        points_t = 2 * points_t / torch.Tensor(image_fw_mm) - 1      # Points in range [-1, 1].
        image_dims_to_add = self._dim + 2 - image_dim
        spatial_dims = list(range(-self._dim, 0))
        image = torch.moveaxis(image, spatial_dims, list(reversed(spatial_dims)))    # Transpose spatial axes for 'grid_sample'.
        image = image.reshape(*(1,) * image_dims_to_add, *image.shape) if image_dims_to_add > 0 else image    # Add missing channels for 'grid_sample'.
        point_dims_to_add = self._dim
        points_t = points_t.reshape(*(1,) * point_dims_to_add, *points_t.shape)   # Add missing channels for 'grid_sample'.
        image_t = torch.nn.functional.grid_sample(image, points_t, align_corners=True)
        image_t = image_t.reshape(*image_shape)
        image = torch.moveaxis(image, spatial_dims, list(reversed(spatial_dims)))    # Transpose spatial axes for 'grid_sample'.

        # Convert to return format.
        image_dims_to_remove = self._dim + 2 - image_dim
        if image_dims_to_remove > 0:
            image_t = image_t.squeeze(axis=tuple(range(image_dims_to_remove)))
        if return_type == 'numpy':
            image_t = image_t.numpy()

        return image_t

    # This is for point clouds, not for image resampling. Note that this
    # requires invertibility of the back point transform, which may not be
    # be available for some transforms (e.g. folded elastic).
    def transform_points(
        self,
        points: Points,
        centre: Point,  # Required for centre of rotation.
        **kwargs) -> Points:
        if isinstance(points, np.ndarray):
            points = torch.Tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        logging.info(f"Rotating points around centre={centre}")
        centre = torch.Tensor(centre)
        trans_matrix = torch.eye(self._dim + 1)
        trans_matrix[:self._dim, self._dim] = -centre  # Translate to origin.
        inv_trans_matrix = torch.eye(self._dim + 1)
        inv_trans_matrix[:self._dim, self._dim] = centre  # Translate back to centre.
        points_h = torch.hstack([points, torch.ones((points.shape[0], 1))])  # Homogeneous coordinates.
        points_h_t = torch.linalg.multi_dot([inv_trans_matrix, self.__matrix, trans_matrix, points_h.T]).T
        points_t = points_h_t[:, :-1]
        if return_type == 'numpy':
            points_t = points_t.numpy()
        return points_t
