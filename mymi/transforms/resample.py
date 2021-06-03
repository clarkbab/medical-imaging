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
    """
    # Create sitk image.
    image = sitk.GetImageFromArray(input)
    image.SetSpacing(tuple(reversed(spacing)))

    # Create resample filter.
    resample = sitk.ResampleImageFilter()
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

    return output
