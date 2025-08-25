import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

from ..transform import DetTransform, RandomTransform
from .flip import FlipTransform
from .rotation import RotationTransform
from .spatial import SpatialTransform

class Pipeline:
    def __init__(
        self,
        transforms: List[Union[DetTransform, RandomTransform]]) -> None:
        dim = transforms[0].dim
        for t in transforms:
            assert t.dim == dim, "All transforms must have same 'dim'."
        # Can be a combination of random and deterministic transforms.
        # For example, we may always want to centre crop first, or extract patches last,
        # with random augmentations in the middle.
        self._dim = dim
        self.__transforms = transforms

    def back_transform_points(
        self,
        points: Points,
        centre: Point) -> Points:
        # Call each transform's 'back_transform_points' in reverse order.
        # This is not the most efficient way to handle the matrix multiplication.
        # We could condense down 4x4 homogeneous matrices before applying to the 
        # Nx4 point matrix. Transforms that consist of homogeneous matrix multiplications
        # should have the option to return these transforms for 'pipeline'.
        points_t = points
        for t in reversed(self.__transforms):
            # Pass transform-specific parameters.
            params = {}
            if isinstance(t, FlipTransform):
                # Transform uses image centre.
                params['centre'] = centre
            elif isinstance(t, RotationTransform):
                # Transform can define own centre or use image centre.
                t_centre = centre if t.params['centre'] == 'centre' else t.params['centre']
                params['centre'] = t_centre
            points_t = t.back_transform_points(points_t, **params)
        return points_t

    def get_det_pipeline(
        self,
        random_seed: Optional[Union[int, List[int]]] = None) -> 'Pipeline':
        # Returns a new pipeline with the deterministic versions of all transforms.
        random_seeds = arg_to_list(random_seed, (int, None), broadcast=len(self.__transforms))
        assert len(random_seeds) == len(self.__transforms), f"Expected number of seeds to match number of transforms ({len(self.__transforms)})."
        transforms = [t.get_det_transform(random_seed=s) if isinstance(t, RandomTransform) else t for t, s in zip(self.__transforms, random_seeds)]
        det_pipeline = Pipeline(transforms)
        return det_pipeline

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__transforms})"

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

        # Perform back transform of resampling points.
        fov_w = np.array(image.shape) * spacing
        fov_c = fov_w / 2
        points_t = self.back_transform_points(points, fov_c)

        # Resample the image at the transformed points.
        points_t = 2 * points_t / torch.Tensor(fov_w) - 1      # Points in range [-1, 1].
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
        size: Size,     # Required for fov centre calc and filtering offscreen landmarks.
        spacing: Spacing,
        ) -> Points:
        # Get FOV and centre.
        fov = ((0, 0, 0), tuple(np.array(spacing) * size))
        fov_c = np.array(fov[1]) / 2

        # Call each transform's 'transform_points'.
        points_t = points
        for t in self.__transforms:
            # Pass transform-specific parameters.
            params = {}
            if isinstance(t, FlipTransform):
                # Transform uses image centre.
                params['centre'] = fov_c
            elif isinstance(t, RotationTransform):
                # Transform can define own centre or use image centre.
                t_centre = fov_c if t.params['centre'] == 'centre' else t.params['centre']
                params['centre'] = t_centre
            points_t = t.transform_points(points_t, **params)

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        print(fov)
        to_keep = (points_t >= fov[0]) & (points_t < fov[1])
        to_keep = to_keep.all(axis=1)
        points_t = points_t[to_keep]
        indices = np.where(to_keep)[0]
        return points_t, indices

    @property
    def transforms(self) -> List[Union[DetTransform, RandomTransform]]:
        return self.__transforms
