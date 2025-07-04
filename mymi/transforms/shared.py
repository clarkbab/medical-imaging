import numpy as np
from typing import *

from mymi import logging
from mymi.typing import *
from mymi.utils import *

def assert_box_width(box: Union[PixelBox, VoxelBox]) -> None:
    # Check box width.
    min, max = box
    for min_i, max_i in zip(min, max):
        width = max_i - min_i
        if width <= 0:
            raise ValueError(f"Box width must be positive, got '{box}'.")

def handle_non_spatial_dims(
    spatial_fn: Callable,
    data: ImageData,
    *args, **kwargs) -> ImageData:
    datas = arg_to_list(data, ImageData)
    outputs = []
    for data in datas:
        n_dims = len(data.shape)
        if n_dims in (2, 3):
            output = spatial_fn(data, *args, **kwargs)
        elif n_dims == 4:
            size = data.shape
            if size[0] > size[-1]:
                logging.warning(f"Channels dimension should come first when resampling 4D. Got shape {size}, is this right?")
            os = []
            for d in data:
                d = spatial_fn(d, *args, **kwargs)
                os.append(d)
            output = stack(os, axis=0)
        elif n_dims == 5:
            size = data.shape
            if size[0] > size[-1]:
                logging.warning(f"Batch dimension should come first when resampling 5D. Got shape {size}, is this right?")
            if size[1] > size[-1]:
                logging.warning(f"Channel dimension should come second when resampling 5D. Got shape {size}, is this right?")
            bs = []
            for batch_item in data:
                ocs = []
                for channel_data in batch_item:
                    oc = spatial_fn(channel_data, *args, **kwargs)
                    ocs.append(oc)
                ocs = stack(ocs, axis=0)
                bs.append(ocs)
            output = stack(bs, axis=0)
        else:
            raise ValueError(f"Data should have (2, 3, 4, 5) dims, got {n_dims}.")

        outputs.append(output)

    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs

def stack(
    data: List[ImageData],
    axis: int = 0) -> ImageData:
    if isinstance(data[0], np.ndarray):
        data = np.stack(data, axis=axis)
    elif isinstance(data[0], torch.Tensor):
        data = torch.stack(data, dim=axis)
    return data

def replace_box_none(
    bounding_box: Union[PixelBox, VoxelBox],
    size: Union[Size2D, Size3D],
    offset: Optional[Union[Point2D, Point3D]] = None,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None,
    use_patient_coords: bool = True) -> Tuple[PixelBox, VoxelBox]:
    if use_patient_coords:
        assert spacing is not None
        assert offset is not None

    # Replace 'None' values.
    n_dims = len(size)
    min, max = bounding_box
    min, max = list(min), list(max)
    for i in range(n_dims):
        if min[i] is None:
            if use_patient_coords:
                min[i] = offset[i]
            else:
                min[i] = 0
        if max[i] is None:
            if use_patient_coords:
                max[i] = size[i] * spacing[i] + offset[i]
            else:
                max[i] = size[i]
    min, max = tuple(min), tuple(max)
    return min, max
