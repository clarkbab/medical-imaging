import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import *

from mymi.typing import *
from mymi.utils import *

from .transforms import assert_box_width

def __spatial_resample(
    data: Optional[ImageData3D] = None,
    fill: Union[float, Literal['min']] = 'min',
    image: Optional[Union['DicomSeries', 'NiftiImageSeries']] = None,
    offset: Point3D = (0, 0, 0),
    output_image: Optional[Union['DicomSeries', 'NiftiImageSeries']] = None,
    output_offset: Optional[Point3D] = None,    # Defaults to input image offset.
    output_size: Optional[Size3D] = None,       # Unless specified, is set to maintain image FOV.
    output_spacing: Optional[Spacing3D] = None,     # Defaults to input image spacing.
    return_transform: bool = False,
    spacing: Spacing3D = (1, 1, 1),
    transform: Optional[sitk.Transform] = None,     # This transforms points not intensities. I.e. positive transform will move image in negative direction.
    ) -> Union[ImageData3D, Tuple[ImageData3D, sitk.Transform]]:
    # Use 'image' and 'output_image' to get data/spacing/offset if provided.
    if data is None:
        if image is None:
            raise ValueError("Either 'data' or 'image' must be provided.")
        else:
            data = image.data
            spacing = image.spacing
            offset = image.offset
    if output_image is not None:
        output_size = output_image.size
        output_spacing = output_image.spacing
        output_offset = output_image.offset

    # Convert to sitk datatypes.
    is_boolean = data.dtype == bool
    if is_boolean:
        data = data.astype(np.uint8) 
    offset = tuple(float(o) for o in offset)
    spacing = tuple(float(s) for s in spacing)
    if output_offset is not None:
        output_offset = tuple(float(o) for o in output_offset)
    if output_spacing is not None:
        output_spacing = tuple(float(s) for s in output_spacing)

    # Create 'sitk' image.
    img = to_sitk_image(data, offset=offset, spacing=spacing)

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
    data: Optional[ImageData] = None,
    **kwargs) -> ImageData:
    return handle_non_spatial_dims(__spatial_resample, data, **kwargs) if data is not None else __spatial_resample(**kwargs)

def resample_box_3D(
    bounding_box: Box3D,
    spacing: Spacing3D,
    new_spacing: Spacing3D) -> Box3D:
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

def __spatial_sample(
    data: ImageData3D,
    points: Union[LandmarksData, Point3D, Points3D],
    fill: Union[float, Literal['min']] = 'min',
    landmarks_col: str = 'sample',
    offset: Optional[Point3D] = None,
    sample_size: SizeMM3D = (0, 0, 0),  # Defaults to point sample.
    sample_spacing: Spacing3D = (1, 1, 1),
    spacing: Optional[Spacing3D] = None,
    transform: Optional[sitk.Transform] = None,     # This transforms points not intensities. I.e. positive transform will move image in negative direction.
    **kwargs) -> Union[ImageData3D, Tuple[ImageData3D, sitk.Transform]]:

    # Convert to sitk datatypes.
    is_boolean = data.dtype == bool
    if is_boolean:
        data = data.astype(np.uint8) 
    if spacing is not None:
        spacing = tuple(float(s) for s in spacing)

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
    if sample_size == (0, 0, 0):
        output_size = (1, 1, 1)    # Sample a single point.
    else:
        output_size = tuple(int(np.ceil(f / s)) for f, s in zip(sample_size, sample_spacing))
    filter.SetSize(output_size)
    if transform is not None:
        filter.SetTransform(transform)

    # Convert points to list.
    if isinstance(points, LandmarksData):
        return_type = 'landmarks'
        lm_df = points.copy()
        points = landmarks_to_data(points)
    elif isinstance(points, Points3D):
        return_type = 'list'
    else:
        return_type = 'single'
        points = [points]

    # Perform resampling.
    res = []
    for p in points:
        origin = tuple(np.array(p) - (np.array(sample_size) / 2))
        filter.SetOutputOrigin(origin)
        rimg = filter.Execute(img)
        r, _, _ = from_sitk_image(rimg)
        r = r.mean()    # Take average of sample grid.
        # r = bool(r) if is_boolean else float(r)
        res.append(r)

    # Convert based on input type.
    if return_type == 'landmarks':
        lm_df[landmarks_col] = res
        res = lm_df
    elif return_type == 'single':
        res = res[0]

    return res

@delegates(__spatial_sample)
def sample(
    data: ImageData,
    *args, **kwargs) -> ImageData:
    assert_image(data)
    return handle_non_spatial_dims(__spatial_sample, data, *args, **kwargs)
