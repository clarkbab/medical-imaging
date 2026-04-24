from dicomset.typing import *
from dicomset.utils import affine_spacing
from dicomset.utils.logging import logger
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt
from scipy.spatial import KDTree
from skimage.segmentation import find_boundaries

def register_linear(
    fixed_label: LabelImage3D,
    moving_label: LabelImage3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    t: float,
    ) -> LabelImage3D:
    logger.log_method()
    assert np.all(fixed_affine == moving_affine), "Fixed/moving affines must match."
    spacing = affine_spacing(fixed_affine)
    fixed_distances = distance_transform_edt(~fixed_label, sampling=spacing) - distance_transform_edt(fixed_label, sampling=spacing)
    moving_distances = distance_transform_edt(~moving_label, sampling=spacing) - distance_transform_edt(moving_label, sampling=spacing)
    return ((1 - t) * fixed_distances + t * moving_distances) <= 0

def register_linear_v2(
    fixed_label: LabelImage3D,
    moving_label: LabelImage3D,
    fixed_affine: AffineMatrix3D | None = None,
    moving_affine: AffineMatrix3D | None = None,
    t: float = 0.5,
    ) -> LabelImage3D:
    """Surface-correspondence interpolation: for each fixed surface voxel, find the
    nearest moving surface voxel, interpolate between the two, then reconstruct a
    filled mask from the resulting surface point cloud."""
    logger.log_method()
    if fixed_affine is not None and moving_affine is not None:
        assert np.all(fixed_affine == moving_affine), "Fixed/moving affines must match."
        spacing = np.array(affine_spacing(fixed_affine))
    else:
        spacing = np.ones(3)

    # Extract surface voxel coordinates.
    fixed_surface = np.argwhere(find_boundaries(fixed_label.astype(bool), mode='inner'))
    moving_surface = np.argwhere(find_boundaries(moving_label.astype(bool), mode='inner'))

    # Convert to physical space for distance-correct nearest-neighbour search.
    fixed_phys = fixed_surface * spacing
    moving_phys = moving_surface * spacing

    # For each fixed surface voxel, find the closest moving surface voxel.
    tree = KDTree(moving_phys)
    _, nn_idx = tree.query(fixed_phys)

    # Interpolate in physical space and convert back to voxel grid.
    interp_phys = (1 - t) * fixed_phys + t * moving_phys[nn_idx]
    interp_vox = np.round(interp_phys / spacing).astype(int)
    shape = fixed_label.shape
    interp_vox = np.clip(interp_vox, 0, np.array(shape) - 1)

    # Place interpolated surface voxels into a grid.
    surface_grid = np.zeros(shape, dtype=bool)
    surface_grid[interp_vox[:, 0], interp_vox[:, 1], interp_vox[:, 2]] = True

    # Dilate to close any gaps caused by rounding, fill the interior, then erode back.
    n_iter = 3
    filled = binary_fill_holes(binary_dilation(surface_grid, iterations=n_iter))
    result = binary_erosion(filled, iterations=n_iter)
    # Ensure interpolated surface voxels are always included.
    return result | surface_grid
