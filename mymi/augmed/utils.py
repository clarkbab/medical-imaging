import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

def create_eye(
    dim: int,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    scaling: Optional[Union[Tuple, np.ndarray, torch.Tensor]] = None) -> torch.Tensor:
    matrix = torch.eye(dim, device=device, dtype=dtype)
    if scaling is not None:
        scalings = to_tensor(scaling, device=device)
        assert len(scaling) == dim
        for i, s in enumerate(scalings):
            matrix[i, i] = s
    return matrix
    
def create_ones(
    size: Tuple[int, ...],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.ones(size, device=device, dtype=dtype)

def create_rotation(
    rotation: Union[Number, Tuple[Number], np.ndarray, torch.Tensor],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dim = len(rotation)
    if dim == 2:
        # 2D rotation matrix.
        matrix = to_tensor([
            [torch.cos(rotation[0]), -torch.sin(rotation[0]), 0],
            [torch.sin(rotation[0]), torch.cos(rotation[0]), 0],
            [0, 0, 1]
        ], dtype=dtype)
    else:
        # 3D rotation matrix.
        rotation_x = to_tensor([
            [1, 0, 0, 0],
            [0, torch.cos(rotation[0]), -torch.sin(rotation[0]), 0],
            [0, torch.sin(rotation[0]), torch.cos(rotation[0]), 0],
            [0, 0, 0, 1]
        ], dtype=dtype)
        rotation_y = to_tensor([
            [torch.cos(rotation[1]), 0, torch.sin(rotation[1]), 0],
            [0, 1, 0, 0],
            [-torch.sin(rotation[1]), 0, torch.cos(rotation[1]), 0],
            [0, 0, 0, 1]
        ], dtype=dtype)
        rotation_z = to_tensor([
            [torch.cos(rotation[2]), -torch.sin(rotation[2]), 0, 0],
            [torch.sin(rotation[2]), torch.cos(rotation[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=dtype)
        matrix = torch.linalg.multi_dot([rotation_z, rotation_y, rotation_x])

    return matrix

def create_translation(
    translation: Union[Point, PointArray, PointTensor],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dim = len(translation)
    matrix = create_eye(dim + 1, device=device, dtype=dtype)
    matrix[:dim, dim] = translation
    return matrix

def expand_range_arg(
    a: Union[Number, Tuple[Number, ...]],
    dim: SpatialDim = 3,
    negate_lower: bool = False,
    vals_per_dim: int = 2) -> Tuple[Number, ...]:
    if isinstance(a, (int, float)):
        ranges = (-a if negate_lower else a, a) * (vals_per_dim // 2) * dim
    elif len(a) == vals_per_dim // 2:
        ranges = a * 2 * dim
    elif len(a) == vals_per_dim:
        ranges = a * dim
    else:
        ranges = a
    return ranges

# 'grid_sample' can be used for interpolating at points (Nx3) or on image grids (3xXxYxZ).
# We don't need to know the spatial coordinates of the resampling grid for the interpolation,
# this information can be added back in after resampling to create the moved image.
# For 'grid_sample' we just need to know the coordinates of each sample (points, mm) in the moving
# image (image) and the coordinates of the moving image grid (spacing, origin).
def grid_sample(
    image: ImageTensor,
    spacing: SpacingTensor,
    origin: PointTensor,
    points: Union[Points, ImageTensor],
    mode: Literal['bicubic', 'bilinear', 'nearest'] = 'bilinear',
    padding: Union[Number, Literal['border', 'max', 'min', 'reflection', 'zeros']] = 'min',
    dim: SpatialDim = 3,
    **kwargs) -> Union[ImageArray, ImageTensor]:
    if points.shape[-1] == 2 or points.shape[-1] == 3:
        points_type = 'points'
    else:
        points_type = 'image'

    # We use 'float32' for resample points and maintain the original dtype of 'image'.
    points = to_tensor(points, device=image.device, dtype=torch.float32)
    spatial_size = to_tensor(image.shape[-dim:], device=image.device, dtype=torch.float32)
    origin = to_tensor(origin, device=image.device, dtype=torch.float32)
    spacing = to_tensor(spacing, device=image.device, dtype=torch.float32)

    # Normalise to range [-1, 1] expected by 'torch.grid_sample'.
    if points_type == 'image':
        points = points.moveaxis(0, -1)     # Move channels to end - expected by 'torch.grid_sample'.
    points = 2 * (points - origin) / ((spatial_size - 1) * spacing) - 1      

    # Add image channels expected by 'torch.grid_sample'.
    image_dims_to_add = dim + 2 - len(image.shape)
    image = image.reshape(*(1,) * image_dims_to_add, *image.shape) if image_dims_to_add > 0 else image

    # Add points channels expected by 'torch.grid_sample'.
    points_dims_to_add = dim + 2 - len(points.shape)
    points = points.reshape(*(1,) * points_dims_to_add, *points.shape)

    # Transpose image spatial axes as expected by 'torch.grid_sample'.
    image_src_dims = list(range(-dim, 0))    # Image should have channels first anyway.
    image_dest_dims = list(reversed(image_src_dims))
    image = torch.moveaxis(image, image_src_dims, image_dest_dims)

    # Convert bool types to float as required.
    return_dtype = image.dtype
    if return_dtype is not torch.float32:
        image = image.type(torch.float32)

    # Convert padding to float.
    if isinstance(padding, str):
        if padding == 'min':
            padding = float(image.min())
        elif padding == 'max':
            padding = float(image.max())
        else:
            padding_mode = padding      # Pass values such as 'border' directly to 'grid_sample'.

    # For number padding, translate intensities as 'grid_sample' only provides zero-padding.
    if isinstance(padding, (int, float)):
        image = image - padding
        padding_mode = 'zeros'

    # Determine interpolation mode.
    mode = 'nearest' if return_dtype is torch.bool else mode

    # Resample image.
    image_t = torch.nn.functional.grid_sample(image, points, align_corners=True, mode=mode, padding_mode=padding_mode, **kwargs)

    # Convert to return types.
    if return_dtype is not torch.float32:
        image_t = image_t.type(return_dtype)

    # Reverse intensity translation for padding.
    if isinstance(padding, (int, float)):
        image_t = image_t + padding

    # Remove channels that were added for 'grid_sample'.
    image_t = image_t.squeeze(axis=tuple(range(image_dims_to_add))) if image_dims_to_add > 0 else image_t

    return image_t

def grid_points(
    size: SizeTensor,
    spacing: SpacingTensor,
    origin: PointTensor,
    return_superset: bool = False) -> PointsTensor:
    sizes, size_was_single = arg_to_list(size, (tuple, np.ndarray, torch.Tensor), return_matched=True)
    spacings = arg_to_list(spacing, (tuple, np.ndarray, torch.Tensor), broadcast=len(sizes))
    origins = arg_to_list(origin, (tuple, np.ndarray, torch.Tensor), broadcast=len(sizes))
    sizes = [to_tensor(s) for s in sizes]
    devices = [s.device for s in sizes]
    spacings = [to_tensor(sp, device=s.device) for sp, s in zip(spacings, sizes)]
    origins = [to_tensor(o, device=s.device) for o, s in zip(origins, sizes)]
    assert len(spacings) == len(sizes)
    assert len(origins) == len(sizes)

    # Get grid points.
    pointses = []
    for si, o, sp, d in zip(sizes, origins, spacings, devices):
        dim = len(si)
        grids = torch.meshgrid([torch.arange(s) for s in si], indexing='ij')
        points_vox = torch.stack(grids, dim=-1).reshape(-1, dim).to(d)
        points = points_vox * sp + o
        pointses.append(points)

    # Create superset.
    if return_superset:
        # Move points to device for concatenation.
        # Which device should this go on? Use first GPU if available, because
        # we're going to have to calculate 'back_transform_points' on these points.
        device_types = [d.type for d in devices]
        super_device = devices[device_types.index('cuda')] if 'cuda' in device_types else devices[0]
        points = [p.to(super_device) for p in pointses]

        # Get superset of points.
        # While it might seem like a good idea to create a superset of points to 
        # reduce transform processing, in practice the act of getting unique points
        # takes much longer than just transforming them all.
        # !!! This 'unique' op, over millions of points, takes a looooong time.
        # This kind of makes the superset idea non-viable.
        # Apparently it sorts the array first, which might be the slow part.
        # Our stacked array is not sorted by default.
        super_points = torch.vstack(points).unique(dim=0)

        # For each image, get the indices of it's points within the superset.
        # This is required for creating subsets later on after 'back_transform_points'.
        indices = []
        for p, d in zip(pointses, devices):
            matches = (p[:, None, :].to(super_device) == super_points[None, :, :])
            matches = matches.all(dim=-1)
            index = matches.float().argmax(dim=1)
            index = index.to(d)
            indices.append(index)

        return super_points, indices

    if size_was_single:
        return pointses[0]
    else:
        return pointses

def to_array(
    data: Optional[Union[Tuple[Union[Number, bool, str]], np.ndarray, torch.Tensor, torch.Size]],
    broadcast: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None) -> Optional[np.ndarray]:
    if data is None:
        return None

    # Convert data to array.
    if isinstance(data, (bool, float, int, str)):
        data = np.array([data])
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    elif isinstance(data, torch.Size):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Set data type.
    if dtype is not None:
        data = data.astype(dtype)

    # Broadcast if required.
    if broadcast is not None and len(data) == 1:
        data = np.repeat(data, broadcast)

    return data

def to_tensor(
    data: Optional[Union[Tuple[Union[bool, Number]], np.ndarray, torch.Size, torch.Tensor]],
    broadcast: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if data is None:
        return None

    # Convert to tensor.
    if isinstance(data, (bool, float, int, str)):
        device = torch.device('cpu') if device is None else device  
        data = torch.tensor([data], device=device, dtype=dtype)
    elif isinstance(data, (list, tuple, np.ndarray, torch.Size)):
        device = torch.device('cpu') if device is None else device  
        data = torch.tensor(data, device=device, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        device = data.device if device is None else device
        dtype = data.dtype if dtype is None else dtype
        data = data.to(device=device, dtype=dtype)

    # Broadcast if required.
    if broadcast is not None and len(data) == 1:
        data = data.repeat(broadcast)

    return data

@delegates(to_array)
def to_tuple(
    data: Optional[Union[Tuple[Union[Number, bool, str]], np.ndarray, torch.Tensor, torch.Size]],
    **kwargs) -> Optional[torch.Tensor]:
    if data is None:
        return None 
    return tuple(to_array(data, **kwargs).tolist())
