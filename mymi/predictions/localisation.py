import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Tuple, Union

from mymi import dataset
from mymi.transforms import centre_crop_or_pad, resample

def get_patient_bounding_box(
    id: str,
    localiser: nn.Module,
    localiser_size: Tuple[int, int, int],
    localiser_spacing: Tuple[float, float, float],
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_prediction: bool = False) -> Union[Tuple[Tuple[float, float, float], Tuple[float, float, float]], Tuple[Tuple[float, float, float], Tuple[float, float, float], np.ndarray]]:
    """
    returns: the bounding box, (mins, widths) pair in voxel coordinates.
    args:
        id: the patient ID.
        localiser: the localiser network.
    kwargs:
        clear_cache: force the cache to clear.
        device: the device to run network calcs on.
        return_prediction: return the network's segmentation prediction.
    """
    # Get the patient CT data and spacing.
    patient = dataset.patient(id)
    input = patient.ct_data(clear_cache=clear_cache)
    spacing = patient.spacing(clear_cache=clear_cache)

    # Create downsampled input.
    input = resample(input, spacing, localiser_spacing)
    downsampled_size = input.shape

    # Shape the image so it'll fit the network.
    input = centre_crop_or_pad(input, localiser_size, fill=input.min())

    # Get localiser result.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with autocast(enabled=True):
        pred = localiser(input)
    pred = pred.cpu()

    # Get binary mask.
    pred = pred.argmax(axis=1)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Reverse the crop/pad.
    pred = centre_crop_or_pad(pred, downsampled_size)

    # Upsample to full resolution.
    pred = resample(pred, localiser_spacing, spacing)

    # Get OAR extent.
    non_zero = np.argwhere(pred != 0).astype(int)
    mins = non_zero.min(axis=0)
    maxs = non_zero.max(axis=0)
    widths = maxs - mins

    # Create result.
    result = (tuple(mins), tuple(widths))
    if return_prediction:
        result = (*result, pred)

    return result
