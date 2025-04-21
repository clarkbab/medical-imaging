import SimpleITK as sitk
from typing import *

from mymi import logging
from mymi.typing import *
from mymi.utils import *

def rigid_image_registration(
    moving_image: Image,
    moving_spacing: ImageSpacing3D,
    moving_offset: PointMM3D,
    fixed_image: Image,
    fixed_spacing: ImageSpacing3D,
    fixed_offset: PointMM3D,
    disable_x_axis_rotation: bool = False,
    disable_y_axis_rotation: bool = False,
    fill: Union[float, Literal['min']] = 'min',
    mask_window: Optional[Tuple[float]] = None,     # Only calculate registration metrics over values in this range.
    metric: Literal['mi', 'mse'] = 'mi',
    show_progress: bool = False) -> Tuple[Image, sitk.Transform]:
    # Convert to sitk images.
    fixed_size = fixed_image.shape
    fixed_sitk = to_sitk_image(fixed_image, fixed_spacing, fixed_offset)
    moving_sitk = to_sitk_image(moving_image, moving_spacing, moving_offset)

    initial_transform = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Set registration masks.
    if mask_window is not None:
        logging.info(f"Using registration mask window: {mask_window}")
        fixed_mask = np.ones(fixed_image.shape, dtype=np.uint8)
        moving_mask = np.ones(moving_image.shape, dtype=np.uint8)
        if mask_window[0] is not None:
            fixed_mask[fixed_image < mask_window[0]] = 0
            moving_mask[moving_image < mask_window[0]] = 0
        if mask_window[1] is not None:
            fixed_mask[fixed_image >= mask_window[1]] = 0
            moving_mask[moving_image >= mask_window[1]] = 0
        fixed_mask_sitk = to_sitk_image(fixed_mask, fixed_spacing, fixed_offset)
        moving_mask_sitk = to_sitk_image(moving_mask, moving_spacing, moving_offset)
        registration_method.SetMetricFixedMask(fixed_mask_sitk)
        registration_method.SetMetricMovingMask(moving_mask_sitk)

    # Similarity metric settings.
    if metric == 'mi':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == 'mse':
        registration_method.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Metric {metric} not recognised.")
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)       # Percentage of voxels sampled (for each iteration?).
    # registration_method.SetMetricAsMeanSquares

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=1000, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    weights = [1.0] * 6
    if disable_x_axis_rotation:
        weights[0] = 0.0
    if disable_y_axis_rotation:
        weights[1] = 0.0
    registration_method.SetOptimizerWeights(weights)

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)

    if show_progress:
        def print_progress(method):
            logging.info(f"Step: {method.GetOptimizerIteration()}, Metric: {method.GetMetricValue()}")
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print_progress(registration_method))

    transform = registration_method.Execute(fixed_sitk, moving_sitk)

    # Always check the reason optimization terminated.
    if show_progress:
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    # Apply transform to moving image.
    if fill == 'min':
        fill = moving_image.min()
    moved_sitk = sitk.Resample(moving_sitk, fixed_sitk, transform, sitk.sitkLinear, fill, moving_sitk.GetPixelID())
    moved, moved_spacing, moved_offset = from_sitk_image(moved_sitk)
    assert moved.shape == fixed_size
    assert moved_spacing == fixed_spacing
    assert moved_offset == fixed_offset

    return moved, transform
    