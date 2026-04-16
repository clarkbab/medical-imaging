from dicomset.typing import *
from dicomset.utils.logging import logger
from typing import Literal, Optional, Tuple
import numpy as np
import SimpleITK as sitk

from mymi.utils.sitk import from_sitk_image, to_sitk_image

def register_rigid(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    fixed_mask: Optional[LabelImage3D] = None,
    moving_mask: Optional[LabelImage3D] = None,
    disable_x_axis_rotation: bool = False,
    disable_y_axis_rotation: bool = False,
    metric: Literal['mi', 'mse'] = 'mi',
    sampling_percentage: float = 0.1,
    show_progress: bool = False,
    ) -> sitk.Transform:
    logger.log_method()

    fixed_sitk = to_sitk_image(fixed_ct, affine=fixed_affine)
    moving_sitk = to_sitk_image(moving_ct, affine=moving_affine)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Set registration masks.
    if fixed_mask is not None:
        fixed_mask_sitk = to_sitk_image(fixed_mask, affine=fixed_affine)
        registration_method.SetMetricFixedMask(fixed_mask_sitk)
    if moving_mask is not None:
        moving_mask_sitk = to_sitk_image(moving_mask, affine=moving_affine)
        registration_method.SetMetricMovingMask(moving_mask_sitk)

    # Similarity metric settings.
    if metric == 'mi':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == 'mse':
        registration_method.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Metric '{metric}' not recognised.")

    # Scale the sampling percentage if a mask is passed - the samples are drawn from the whole
    # fixed image, so we might miss the mask for small masks.
    effective_p = sampling_percentage
    if fixed_mask is not None:
        mask_ratio = np.sum(fixed_mask.astype(bool)) / fixed_mask.size
        if mask_ratio > 0:
            effective_p = min(1.0, sampling_percentage / mask_ratio)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(effective_p)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.5, numberOfIterations=1000,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    weights = [1.0] * 6
    if disable_x_axis_rotation:
        weights[0] = 0.0
    if disable_y_axis_rotation:
        weights[1] = 0.0
    registration_method.SetOptimizerWeights(weights)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    if show_progress:
        def print_progress(method):
            logger.info(f"Step: {method.GetOptimizerIteration()}, Metric: {method.GetMetricValue()}")
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print_progress(registration_method))

    transform = registration_method.Execute(fixed_sitk, moving_sitk)

    if show_progress:
        logger.info(f"Final metric value: {registration_method.GetMetricValue()}")
        logger.info(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")

    return transform
