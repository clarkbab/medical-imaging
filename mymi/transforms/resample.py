import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
from typing import Any, Dict, Optional, Tuple, Union

from mymi import logging
from mymi.types import Box3D, PointMM3D, ImageOffset3D, ImageSize3D, ImageSpacing3D
from mymi.utils import to_sitk_image

def resample_list(
    data: np.ndarray,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    ds = []
    for d in data:
        d = resample(d, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def resample_4D(
    data: np.ndarray,
    **kwargs: Dict[str, Any]) -> np.ndarray:
    assert len(data.shape) == 4, "Data must be 4D."
    ds = []
    for d in data:
        d = resample_3D(d, **kwargs)
        ds.append(d)
    output = np.stack(ds, axis=0)
    return output

def resample_3D_zoom(
    data: np.ndarray,
    output_spacing: Optional[ImageSpacing3D] = None,
    spacing: Optional[ImageSpacing3D] = None) -> np.ndarray:
    scaling = np.array(output_spacing) / spacing
    data = zoom(data, scaling, order=1)
    return data

    moving_resampled = sitk.Resample(moving_image, fixed_image, registration_transform, sitk.sitkLinear, moving_min, moving_image.GetPixelID())

def resample(
    data: np.ndarray,
    fill: float = 'min',
    offset: Optional[PointMM3D] = None,
    output_offset: Optional[PointMM3D] = None,
    output_size: Optional[ImageSize3D] = None,
    output_spacing: Optional[ImageSpacing3D] = None,
    return_transform: bool = False,
    spacing: Optional[ImageSpacing3D] = None) -> Union[np.ndarray, Tuple[np.ndarray, sitk.Transform]]:
    """
    output_offset: 
        - if None, will take on value of 'offset'.
        - if specified, will result in translation of the resulting image (cropping/padding).
    output_size:
        - if None, will take on dimensions of 'data'.
        - if None, will be calculated as a scaling of the 'data' dimensions, where the scaling is determined
            by the ratio of 'spacing' to 'output_spacing'. This ensures, that all image information is preserved
            when doing a spatial resampling.
    output_spacing:
        - if None, will take on value of 'spacing'.
        - if specified, will change the spatial resolution of the image.
    """
    # Convert to SimpleITK ordering (z, y, x).
    if offset is not None:
        offset = tuple(reversed(offset))
    if output_offset is not None:
        output_offset = tuple(reversed(output_offset))
    if output_spacing is not None:
        output_spacing = tuple(reversed(output_spacing))
    if output_size is not None:
        output_size = tuple(reversed(output_size))
    if spacing is not None:
        spacing = tuple(reversed(spacing))

    # Convert datatypes.
    if spacing is not None:
        spacing = tuple(float(s) for s in spacing)
    if output_spacing is not None:
        output_spacing = tuple(float(s) for s in output_spacing)

    # Convert boolean data to sitk-friendly type.
    boolean = data.dtype == bool
    if boolean:
        data = data.astype('uint8') 

    # Create 'sitk' image and set parameters.
    image = sitk.GetImageFromArray(data)
    if offset is not None:
        image.SetOrigin(offset)
    if spacing is not None:
        image.SetSpacing(spacing)

    # Get default pixel value.
    if isinstance(fill, str):
        if fill == 'min':
            fill = float(data.min())
        elif fill == 'max':
            fill = float(data.max())
        else:
            raise ValueError(f"Unknown fill value '{fill}'.")
    else:
        fill = float(fill)

    # Create resample filter.
    resample = sitk.ResampleImageFilter()
    resample.SetDefaultPixelValue(fill)
    if boolean:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    if output_offset is not None:
        resample.SetOutputOrigin(output_offset)
    else:
        resample.SetOutputOrigin(image.GetOrigin())
    if output_spacing is not None:
        resample.SetOutputSpacing(output_spacing)
    else:
        resample.SetOutputSpacing(image.GetSpacing())
    if output_size is not None:
        resample.SetSize(output_size)
    else:
        scaling = np.array(image.GetSpacing()) / resample.GetOutputSpacing()

        # Magic formula is: n_new = f * (n - 1) + 1
        size = tuple(int(np.ceil(f * (n - 1) + 1)) for f, n in zip(scaling, image.GetSize()))
        resample.SetSize(size)

    # Perform resampling.
    image = resample.Execute(image)

    # Get output data.
    output = sitk.GetArrayFromImage(image)

    # Convert back to boolean.
    if boolean:
        output = output.astype(bool)

    if return_transform:
        return output, resample.GetTransform()
    else:
        return output

def resample_3D(
    data: np.ndarray,
    offset: Optional[PointMM3D] = None,
    output_offset: Optional[PointMM3D] = None,
    output_size: Optional[ImageSize3D] = None,
    output_spacing: Optional[ImageSpacing3D] = None,
    return_transform: bool = False,
    spacing: Optional[ImageSpacing3D] = None) -> Union[np.ndarray, Tuple[np.ndarray, sitk.Transform]]:
    """
    output_offset: 
        - if None, will take on value of 'offset'.
        - if specified, will result in translation of the resulting image (cropping/padding).
    output_size:
        - if None, will take on dimensions of 'data'.
        - if None, will be calculated as a scaling of the 'data' dimensions, where the scaling is determined
            by the ratio of 'spacing' to 'output_spacing'. This ensures, that all image information is preserved
            when doing a spatial resampling.
    output_spacing:
        - if None, will take on value of 'spacing'.
        - if specified, will change the spatial resolution of the image.
    """
    assert len(data.shape) == 3, "Data must be 3D."

    # Convert to SimpleITK ordering (z, y, x).
    if offset is not None:
        offset = tuple(reversed(offset))
    if output_offset is not None:
        output_offset = tuple(reversed(output_offset))
    if output_spacing is not None:
        output_spacing = tuple(reversed(output_spacing))
    if output_size is not None:
        output_size = tuple(reversed(output_size))
    if spacing is not None:
        spacing = tuple(reversed(spacing))

    # Convert datatypes.
    if spacing is not None:
        spacing = tuple(float(s) for s in spacing)
    if output_spacing is not None:
        output_spacing = tuple(float(s) for s in output_spacing)

    # Convert boolean data to sitk-friendly type.
    boolean = data.dtype == bool
    if boolean:
        data = data.astype('uint8') 

    # Create 'sitk' image and set parameters.
    image = sitk.GetImageFromArray(data)
    if offset is not None:
        image.SetOrigin(offset)
    if spacing is not None:
        image.SetSpacing(spacing)

    # Create resample filter.
    resample = sitk.ResampleImageFilter()
    resample.SetDefaultPixelValue(float(data.min()))
    if boolean:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    if output_offset is not None:
        resample.SetOutputOrigin(output_offset)
    else:
        resample.SetOutputOrigin(image.GetOrigin())
    if output_spacing is not None:
        resample.SetOutputSpacing(output_spacing)
    else:
        resample.SetOutputSpacing(image.GetSpacing())
    if output_size is not None:
        resample.SetSize(output_size)
    else:
        scaling = np.array(image.GetSpacing()) / resample.GetOutputSpacing()

        # Magic formula is: n_new = f * (n - 1) + 1
        size = tuple(int(np.ceil(f * (n - 1) + 1)) for f, n in zip(scaling, image.GetSize()))
        resample.SetSize(size)

    # Perform resampling.
    image = resample.Execute(image)

    # Get output data.
    output = sitk.GetArrayFromImage(image)

    # Convert back to boolean.
    if boolean:
        output = output.astype(bool)

    if return_transform:
        return output, resample.GetTransform()
    else:
        return output

def resample_box_3D(
    bounding_box: Box3D,
    spacing: ImageSpacing3D,
    new_spacing: ImageSpacing3D) -> Box3D:
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
