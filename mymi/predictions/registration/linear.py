from dicomset.typing import *
from dicomset.utils import affine_spacing
from dicomset.utils.logging import logger
import numpy as np
from scipy.ndimage import distance_transform_edt

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
