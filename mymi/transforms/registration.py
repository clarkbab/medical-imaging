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
    show_progress: bool = False) -> Tuple[Image, sitk.Transform]:
    # Convert to sitk images.
    fixed_size = fixed_image.shape
    moving_sitk = to_sitk_image(moving_image, moving_spacing, moving_offset)
    fixed_sitk = to_sitk_image(fixed_image, fixed_spacing, fixed_offset)

    initial_transform = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    # registration_method.SetMetricAsMeanSquares

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

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
    moved_sitk = sitk.Resample(moving_sitk, fixed_sitk, transform, sitk.sitkLinear, moving_image.min(), moving_sitk.GetPixelID())
    moved, moved_spacing, moved_offset = from_sitk(moved_sitk)
    assert moved.shape == fixed_size
    assert moved_spacing == fixed_spacing
    assert moved_offset == fixed_offset

    return moved, transform
    