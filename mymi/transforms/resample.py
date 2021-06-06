import numpy as np
import SimpleITK as sitk

from mymi import types

def resample_3D(
    input: np.ndarray,
    spacing: types.Spacing3D,
    new_spacing: types.Spacing3D) -> sitk.Image:
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

def resample_box_3D(
    bounding_box: types.Box3D,
    spacing: types.Spacing3D,
    new_spacing: types.Spacing3D) -> types.Box3D:
    """
    returns: a bounding box in resampled coordinates.
    args:
        bounding_box: the bounding box.
        spacing: the current voxel spacing.
        new_spacing: the new voxel spacing.
    """
    # Convert bounding box to label array.
    min, max = bounding_box
    bbox_label = np.zeros(max, dtype=bool)
    slices = tuple(slice(min, max) for min, max in zip(min, max))
    bbox_label[slices] = 1

    # Resample label array.
    bbox_label = resample_3D(bbox_label, spacing, new_spacing)

    # Extract new bounding box.
    non_zero = np.argwhere(bbox_label != 0).astype(int)
    min = tuple(non_zero.min(axis=0))
    max = tuple(non_zero.max(axis=0))
    bounding_box = (min, max)

    return bounding_box
