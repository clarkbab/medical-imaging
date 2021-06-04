import numpy as np
import SimpleITK as sitk
from typing import Tuple

def resample(
    input: np.ndarray,
    spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float]) -> sitk.Image:
    """
    returns: a resampled tensor.
    args:
        image: the SimpleITK image.
        spacing: the old spacing.
        new_spacing: the new spacing.
    kwargs:
        nearest_neighbour: use nearest neighbour interpolation.
    """
    # Convert boolean data to sitk-friendly type.
    boolean = input.dtype == bool
    if boolean:
        input = input.astype('uint8') 

    # Create sitk image.
    image = sitk.GetImageFromArray(input)
    image.SetSpacing(tuple(reversed(spacing)))

    # Create resample filter.
    resample = sitk.ResampleImageFilter()
    if boolean:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(tuple(reversed(new_spacing)))
    image_size = np.array(image.GetSize())
    image_spacing = np.array(image.GetSpacing())
    new_size = np.ceil(image_size * (image_spacing / tuple(reversed(new_spacing))))
    new_size = tuple(int(s) for s in new_size)
    resample.SetSize(new_size)

    # Perform resampling.
    image = resample.Execute(image)
    
    # Get output data.
    output = sitk.GetArrayFromImage(image)

    # Convert back to boolean.
    if boolean:
        output = output.astype(bool)

    return output
