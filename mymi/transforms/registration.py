import numpy as np
import SimpleITK as sitk
from typing import Tuple, Union

from mymi import logging
from mymi.types import ImageSize3D, ImageSpacing3D

def register_image(
    moving_image: np.ndarray,
    moving_spacing: ImageSpacing3D,
    fixed_image: np.ndarray,
    fixed_spacing: ImageSpacing3D,
    return_transform: bool = False,
    show_progress: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, sitk.Transform]]:
    # Convert to SimpleITK ordering (z, y, x).
    fixed_spacing = tuple(reversed(fixed_spacing))
    moving_spacing = tuple(reversed(moving_spacing))

    # Convert to SimpleITK images.
    fixed_shape = fixed_image.shape
    fixed_image = sitk.GetImageFromArray(fixed_image)
    fixed_image.SetSpacing(fixed_spacing)
    moving_min = moving_image.min()
    moving_image = sitk.GetImageFromArray(moving_image)
    moving_image.SetSpacing(moving_spacing)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

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

    registration_transform = registration_method.Execute(fixed_image, moving_image)

    # Always check the reason optimization terminated.
    if show_progress:
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    # Apply transform to moving image.
    moving_resampled = sitk.Resample(moving_image, fixed_image, registration_transform, sitk.sitkLinear, moving_min, moving_image.GetPixelID())
    moving_resampled = sitk.GetArrayFromImage(moving_resampled)
    assert moving_resampled.shape == fixed_shape

    if return_transform:
        return moving_resampled, registration_transform
    else:
        return moving_resampled 

def register_label(
    moving_label: np.ndarray,
    fixed_spacing: ImageSpacing3D, 
    moving_spacing: ImageSpacing3D,
    fixed_size: ImageSize3D,
    transform: sitk.Transform) -> np.ndarray:
    # Convert to SimpleITK ordering (z, y, x).
    fixed_spacing = tuple(reversed(fixed_spacing))
    moving_spacing = tuple(reversed(moving_spacing))
    fixed_size = tuple(reversed(fixed_size))

    # Convert to SimpleITK images.
    moving_label = moving_label.astype('uint8')     # Convert to sitk-friendly type.
    moving_label = sitk.GetImageFromArray(moving_label)
    moving_label.SetSpacing(moving_spacing)

    # Apply transform.
    moving_label_resampled = sitk.Resample(moving_label, fixed_size, transform=transform, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0, outputPixelType=moving_label.GetPixelID(), outputSpacing=fixed_spacing)
    moving_label_resampled = sitk.GetArrayFromImage(moving_label_resampled)
    moving_label_resampled = moving_label_resampled.astype(bool)

    return moving_label_resampled
    