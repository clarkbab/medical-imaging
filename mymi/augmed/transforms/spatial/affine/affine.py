import torch
from typing import *

from mymi.typing import *

from ....utils import *
from ..random import RandomSpatialTransform
from ..spatial import SpatialTransform

class RandomAffine(RandomSpatialTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def get_affine_back_transform(
        self,
        device: torch.device,
        **kwargs) -> torch.Tensor:
        return self.freeze().get_affine_back_transform(device, **kwargs)

    def get_affine_transform(
        self,
        device: torch.device,
        **kwargs) -> torch.Tensor:
        return self.freeze().get_affine_transform(device, **kwargs)

# Flip, Rotation, Translation (and others) should probably subclass this.
class Affine(SpatialTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)

    def get_affine_back_transform(
        self,
        # These are required because the function may need to multiply matrices (e.g. rotation around
        # image centre). Probably pretty quick on CPU, but anyway...
        device: torch.device,
        **kwargs) -> torch.Tensor:
        raise ValueError(f"Affine transforms must implement 'get_affine_back_transform' method.")

    def get_affine_transform(
        self,
        # These are required because the function may need to multiply matrices (e.g. rotation around
        # image centre). Probably pretty quick on CPU, but anyway...
        device: torch.device,
        **kwargs) -> torch.Tensor:
        raise ValueError(f"Affine transforms must implement 'get_affine_transform' method.")

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
        origin = to_tensor(origin, device=points.device, dtype=points.dtype)
        size = to_tensor(size, device=points.device, dtype=points.dtype)
        spacing = to_tensor(spacing, device=points.device, dtype=points.dtype)

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
