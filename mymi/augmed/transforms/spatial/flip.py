from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ..transform import DetTransform, RandomTransform
from .identity import IdentityTransform
from .spatial import SpatialTransform

class RandomFlip(RandomTransform):
    def __init__(
        self,
        p_flip: Tuple[float] = (0.5,),
        dim: int = 3,
        p: float = 1.0) -> None:
        assert dim in [2, 3], "Only 2D and 3D flips are supported."
        if dim == 2:
            assert len(p_flip) == 1 or len(p_flip) == 2, "2D rotation requires 'p_flip' of length 1 or 2."
        else:
            assert len(p_flip) == 1 or len(p_flip) == 3, "3D rotation requires 'p_flip' of length 1 or 3."
        if len(p_flip) == 1:
            p_flip = p_flip * dim
        self._dim = dim
        self.__p_flip = torch.Tensor(p_flip)
        self.__p = p

    def get_det_transform(
        self,
        random_seed: Optional[int] = None) -> SpatialTransform:
        if random_seed is not None:
            torch.manual_seed(random_seed)
        should_apply = torch.rand(1) < self.__p
        if not should_apply:
            return IdentityTransform()
        should_flip = torch.rand(self._dim) < self.__p_flip
        return FlipTransform(self._dim, should_flip)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({tuple(self.__p_flip.cpu().numpy())}, dim={self._dim}, p={self.__p})"
        
class FlipTransform(DetTransform, SpatialTransform):
    def __init__(
        self,
        dim: int,
        flips: Tuple[bool]) -> None:
        assert len(flips) == dim
        self._dim = dim
        self.__flips = flips
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
        points: Points,
        centre: Point,
        **kwargs) -> Points:
        if isinstance(points, np.ndarray):
            points = torch.Tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        logging.info(f"Back-flipping around centre={centre}")
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

    def __create_transforms(self) -> None:
        if self._dim == 2:
            # 2D flip matrix.
            self.__matrix = torch.Tensor([
                [-1 if self.__flips[0] else 1, 0, 0],
                [0, self.__flips[1], 0],
                [0, 0, 1],
            ])
        elif self._dim == 3:
            # 3D flip matrix.
            self.__matrix = torch.Tensor([
                [-1 if self.__flips[0] else 1, 0, 0, 0],
                [0, -1 if self.__flips[1] else 1, 0, 0],
                [0, 0, -1 if self.__flips[2] else 1, 0],
                [0, 0, 0, 1],
            ])

        # Flip matrix is it's own inverse.
        self.__backward_matrix = self.__matrix

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({tuple(self.__flips.cpu().numpy())}, dim={self._dim})"

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

        # Get points in mm.
        image_spatial_shape = image.shape[-self._dim:]
        grids = torch.meshgrid([torch.arange(s) for s in image_spatial_shape], indexing='ij')
        points_vox = torch.stack(grids, dim=-1).reshape(-1, self._dim)
        points = points_vox * torch.Tensor(spacing)

        # Perform rotation.
        centre = torch.Tensor(fov_centre(image, offset=(0, 0, 0), spacing=spacing, use_patient_coords=True))
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
        logging.info(f"Flipping points around centre={centre}")
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
