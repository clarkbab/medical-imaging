from mymi.transforms.crop_or_pad import crop_or_pad_3D
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Tuple, Union

from mymi import dataset
from mymi.postprocessing import get_largest_cc
from mymi.transforms import centre_crop_or_pad_3D, resample_3D
from mymi import types

def get_patient_localisation(
    id: types.PatientID,
    localiser: nn.Module,
    localiser_size: types.ImageSize3D,
    localiser_spacing: types.ImageSpacing3D,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_seg: bool = False,
    use_postprocessing: bool = True) -> Union[types.Box3D, Tuple[types.Box3D, np.ndarray]]:
    """
    returns: the bounding box (min, max) pair in voxel coordinates. Optionally returns the localiser
        segmentation prediction.
    args:
        id: the patient ID.
        localiser: the localiser network.
        localiser_size: the input size expected by the localiser network.
        localiser_spacing: the input spacing expected by the localiser network.
    kwargs:
        clear_cache: force the cache to clear.
        device: the device to run network calcs on.
        return_seg: return the network's segmentation prediction.
        use_postprocessing: apply postprocessing steps.
    """
    # Get the patient CT data and spacing.
    patient = dataset.patient(id)
    input = patient.ct_data(clear_cache=clear_cache)
    spacing = patient.ct_spacing(clear_cache=clear_cache)
    input_size = input.shape

    # Create downsampled input.
    input = resample_3D(input, spacing, localiser_spacing)
    downsampled_size = input.shape

    # Shape the image so it'll fit the network.
    input = centre_crop_or_pad_3D(input, localiser_size, fill=input.min())

    # Get localiser result.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    autocast_enabled = device.type == 'cuda'
    with autocast(enabled=autocast_enabled):
        pred = localiser(input)
    pred = pred.cpu()

    # Get binary mask.
    pred = pred.argmax(axis=1)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Reverse the crop/pad.
    pred = centre_crop_or_pad_3D(pred, downsampled_size)

    # Upsample to full resolution.
    pred = resample_3D(pred, localiser_spacing, spacing)
    
    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    # Run CCA.
    if use_postprocessing:
        pred = get_largest_cc(pred)

    # Get OAR extent.
    non_zero = np.argwhere(pred != 0).astype(int)
    min = tuple(non_zero.min(axis=0))
    max = tuple(non_zero.max(axis=0))
    bounding_box = (min, max)

    # Create result.
    if return_seg:
        return (bounding_box, pred)
    else:
        return bounding_box
