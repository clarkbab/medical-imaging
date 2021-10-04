from mymi.transforms.crop_or_pad import crop_or_pad_3D
import numpy as np
import torch
from typing import Optional, Tuple, Union

from mymi.dataset import Dataset
from mymi.models.systems import Localiser
from mymi.postprocessing import get_largest_cc
from mymi.transforms import centre_crop_or_pad_3D, resample_3D
from mymi import types

def get_localiser_prediction(
    dataset: Dataset,
    pat_id: types.PatientID,
    localiser: types.Model,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_seg: bool = False) -> Union[Optional[types.Box3D], Tuple[Optional[types.Box3D], np.ndarray]]:
    """
    returns: the bounding box (min, max) pair in voxel coordinates. Optionally returns the localiser
        segmentation prediction.
    args:
        localiser: the localiser model.
        dataset: the dataset.
        pat_id: the patient ID.
    kwargs:
        clear_cache: force the cache to clear.
        device: the device to run network calcs on.
        return_seg: return the network's segmentation prediction.
    """
    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    localiser.eval()
    localiser.to(device)
    localiser_size = (128, 128, 96)
    localiser_spacing = (4, 4, 6.625)

    # Get the patient CT data and spacing.
    patient = dataset.patient(pat_id)
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
    with torch.no_grad():
        pred = localiser(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Reverse the crop/pad.
    pred = centre_crop_or_pad_3D(pred, downsampled_size)

    # Upsample to full resolution.
    pred = resample_3D(pred, localiser_spacing, spacing)
    
    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    # Get OAR extent.
    if pred.sum() > 0:
        non_zero = np.argwhere(pred != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)
    else:
        box = None

    # Create result.
    if return_seg:
        return (box, pred)
    else:
        return box
