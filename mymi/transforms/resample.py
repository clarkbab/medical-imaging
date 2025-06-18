import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import *

from mymi import logging
from mymi.typing import *
from mymi.utils import *

from .shared import handle_non_spatial_dims

def __spatial_resample(
    data: Image,
    fill: Union[float, Literal['min']] = 'min',
    offset: Optional[Point3D] = None,
    output_offset: Optional[Point3D] = None,
    output_size: Optional[Size3D] = None,
    output_spacing: Optional[Spacing3D] = None,
    return_transform: bool = False,
    spacing: Optional[Spacing3D] = None,
    transform: Optional[sitk.Transform] = None,     # This transforms points not intensities. I.e. positive transform will move image in negative direction.
    ) -> Union[Image, Tuple[Image, sitk.Transform]]:

    # Convert to sitk datatypes.
    is_boolean = data.dtype == bool
    if is_boolean:
        data = data.astype(np.uint8) 
    if output_offset is not None:
        output_offset = tuple(float(o) for o in output_offset)
    if spacing is not None:
        spacing = tuple(float(s) for s in spacing)
    if output_spacing is not None:
        output_spacing = tuple(float(s) for s in output_spacing)

    # Create 'sitk' image and set parameters.
    kwargs = {}
    if spacing is not None:
        kwargs['spacing'] = spacing
    if offset is not None:
        kwargs['offset'] = offset
    img = to_sitk_image(data, **kwargs)

    # Create resample filter.
    filter = sitk.ResampleImageFilter()
    if fill == 'min':
        fill = float(data.min())
    filter.SetDefaultPixelValue(fill)
    if is_boolean:
        filter.SetInterpolator(sitk.sitkNearestNeighbor)
    if output_offset is not None:
        filter.SetOutputOrigin(output_offset)
    else:
        filter.SetOutputOrigin(img.GetOrigin())
    if output_spacing is not None:
        filter.SetOutputSpacing(output_spacing)
    else:
        filter.SetOutputSpacing(img.GetSpacing())
    if output_size is not None:
        filter.SetSize(output_size)
    else:
        # Choose output size that maintains the image field-of-view.
        size_factor = np.array(img.GetSpacing()) / filter.GetOutputSpacing()

        # Magic formula is: n_new = f * (n - 1) + 1
        # I think I worked this out by trial and error, but what's going on here is:
        # (n - 1) is the number of intervals between voxels (voxel fov), multiplied by the size
        # factor gives the "voxel fov" of the new image. Plus one to get number of voxels.
        # E.g. downsampling by a factor of 2, from 5 voxels: f = 0.5, 0.5 * (5 - 1) + 1 = 3.
        size = tuple(int(np.ceil(f * (s - 1) + 1)) for f, s in zip(size_factor, img.GetSize()))
        filter.SetSize(size)
    if transform is not None:
        filter.SetTransform(transform)

    # Perform resampling.
    img = filter.Execute(img)

    # Get output data.
    image, _, _ = from_sitk_image(img)

    # Convert back to boolean.
    if is_boolean:
        image = image.astype(bool)

    if return_transform:
        return image, filter.GetTransform()
    else:
        return image

@delegates(__spatial_resample)
def resample(
    data: Images,
    *args, **kwargs) -> Images:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_resample, data, *args, **kwargs)

def resample_box_3D(
    bounding_box: VoxelBox,
    spacing: Spacing3D,
    new_spacing: Spacing3D) -> VoxelBox:
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
    bbox_label = resample(bbox_label, spacing, new_spacing)

    # Extract new bounding box.
    non_zero = np.argwhere(bbox_label != 0).astype(int)
    min = tuple(non_zero.min(axis=0))
    max = tuple(non_zero.max(axis=0))
    bounding_box = (min, max)

    return bounding_box

def resample_landmarks(
    landmarks: pd.DataFrame,
    offset: Point3D = (0, 0, 0),
    output_offset: Point3D = (0, 0, 0),
    output_spacing: Spacing3D = (1, 1, 1),
    spacing: Spacing3D = (1, 1, 1)) -> pd.DataFrame:
    raise MemoryError("You forgot that this function makes no sense. Landmarks are not images, they're points in physical space.")

    # Transform landmarks.
    mult = np.array(spacing) / np.array(output_spacing)
    landmarks[list(range(3))] = (landmarks[list(range(3))] - offset) * mult + output_offset

    return landmarks
