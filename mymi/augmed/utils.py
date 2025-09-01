import numpy as np
import torch
from typing import *

from mymi.typing import *
from mymi.utils import *

def create_eye(
    dim: int,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.eye(dim, device=device, dtype=dtype)
    
def create_ones(
    size: Tuple[int, ...],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.ones(size, device=device, dtype=dtype)

def create_translation(
    t: Union[Point, PointArray, PointTensor],
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dim = len(t)
    matrix = create_eye(dim + 1, device=device, dtype=dtype)
    matrix[:dim, dim] = t  # Translate to origin.
    return matrix

def image_points(
    image: Union[ImageArray, ImageTensor, List[Union[ImageArray, ImageTensor]]],
    spacing: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
    origin: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
    return_superset: bool = False) -> Union[PointsArray, PointsTensor, List[Union[PointsArray, PointsTensor]], Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
    images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_matched=True)
    return_types = ['torch' if isinstance(i, torch.Tensor) else 'numpy' for i in images]
    spacings = arg_to_list(spacing, (tuple, np.ndarray, torch.Tensor), broadcast=len(images))
    origins = arg_to_list(origin, (tuple, np.ndarray, torch.Tensor), broadcast=len(images))
    images = [to_tensor(i) for i in images]
    devices = [i.device for i in images]
    sizes = [to_tensor(i.shape, device=i.device, dtype=torch.int) for i in images]
    spacings = [to_tensor(s, device=i.device) for s, i in zip(spacings, images)]
    origins = [to_tensor(o, device=i.device) for o, i in zip(origins, images)]
    assert len(spacings) == len(sizes)
    assert len(origins) == len(sizes)

    # Get grid points.
    points_mms = []
    for si, o, sp, r, d in zip(sizes, origins, spacings, return_types, devices):
        dim = len(si)
        grids = torch.meshgrid([torch.arange(s) for s in si], indexing='ij')
        points_vox = torch.stack(grids, dim=-1).reshape(-1, dim).to(d)
        points_mm = points_vox * sp + o
        points_mms.append(points_mm)

    # Create superset.
    if return_superset:
        # Move points to device for concatenation.
        # Which device should this go on? Use first GPU if available, because
        # we're going to have to calculate 'back_transform_points' on these points.
        device_types = [d.type for d in devices]
        super_device = devices[device_types.index('cuda')] if 'cuda' in device_types else devices[0]
        points_mms = [p.to(super_device) for p in points_mms]

        # Get superset of points.
        # While it might seem like a good idea to create a superset of points to 
        # reduce transform processing, in practice the act of getting unique points
        # takes much longer than just transforming them all.
        # !!! This 'unique' op, over millions of points, takes a looooong time.
        # This kind of makes the superset idea non-viable.
        # Apparently it sorts the array first, which might be the slow part.
        # Our stacked array is not sorted by default.
        super_points_mm = torch.vstack(points_mms).unique(dim=0)

        # For each image, get the indices of it's points within the superset.
        # This is required for creating subsets later on after 'back_transform_points'.
        indices = []
        for p, d in zip(points_mms, devices):
            matches = (p[:, None, :].to(super_device) == super_points_mm[None, :, :])
            matches = matches.all(dim=-1)
            index = matches.float().argmax(dim=1)
            index = index.to(d)
            indices.append(index)

        if 'torch' in return_types:
            return super_points_mm, indices
        else:
            indices = [to_array(i) for i in indices]
            return to_array(super_points_mm), indices 
    
    # Set return types.
    for i, r in enumerate(return_types):
        if r == 'numpy':
            points_mms[i] = to_array(points_mms[i])

    if image_was_single:
        return points_mms[0]
    else:
        return points_mms

def to_array(
    data: Optional[Union[Tuple[Union[bool, Number]], np.ndarray, torch.Tensor, torch.Size]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None) -> Optional[np.ndarray]:
    # Gives no-op for None.
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        # Use data dtype default.
        dtype = data.dtype if dtype is None else dtype
        return data.astype(dtype)
    if isinstance(data, torch.Size):
        return np.array(data)
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    dtype = np.float32 if dtype is None else dtype
    return np.array(data, dtype=dtype)

def to_tensor(
    data: Optional[Union[Tuple[Union[bool, Number]], np.ndarray, torch.Size, torch.Tensor]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    # Gives no-op for None.
    if data is None:
        return None
    if isinstance(data, torch.Size):
        return torch.tensor(data, device=device)
    if isinstance(data, torch.Tensor):
        # Use data device/dtype defaults.
        device = data.device if device is None else device
        dtype = data.dtype if dtype is None else dtype
        return data.to(device=device, dtype=dtype)
    # Use CPU for default device.
    device = torch.device('cpu') if device is None else device  
    dtype = torch.float32 if dtype is None else dtype
    return torch.tensor(data, device=device, dtype=dtype)

@delegates(to_array)
def to_tuple(
    *args,
    **kwargs) -> Optional[torch.Tensor]:
    return tuple(to_array(*args, **kwargs))
