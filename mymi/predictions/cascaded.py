import numpy as np
import pydicom as dcm
import torch
from torch import nn
from typing import *

from mymi.predictions import get_patient_bounding_box
from mymi import types

def get_patient_segmentation(
    id: Union[str, int],
    localiser: nn.Module,
    localiser_size: types.Size3D,
    localiser_spacing: types.Spacing3D,
    segmenter: nn.Module,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu')) -> dcm.FileDataset:
    """
    returns: an RTSTRUCT dicom file containing the predictions made by the model.
    args:
        ds: the dataset name.
        patient: the patient ID.
        localiser: the localiser model.
        segmenter: the segmenter model.
    kwargs:
        clear_cache: force the cache to clear.
    """
    # Get the OAR bounding box.
    mins, widths = get_patient_bounding_box(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device)

    # Get segmentation prediction.
    pred = _get_segmentation(input, (mins, widths), segmenter, device=device)
    print('segmenter pred shape:', pred.shape)

    return pred

def _get_segmentation(
    input: np.ndarray,
    bounding_box: types.Box3D,
    segmenter: nn.Module,
    device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    returns: the full result of segmentation at (1, 1, 3) spacing.
    args:
        input: the input array at (1, 1, 3) spacing.
        bounding_box: a (mins, widths) pair describing the voxel locations of the bounding box.
        pred: the location prediction at (1, 1, 3) spacing.
        segmenter: the segmenter model.
    kwargs:
        device: the device to use for network calcs.
    """
    # Extract patch around bounding box.
    print('segmentation input shape:', input.shape)
    input, crop_or_padding = _extract_patch(input, pred, (128, 128, 96))
    print('patch shape:', input.shape)
    print('crop or padding:', crop_or_padding)

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    pred = segmenter(input)
    pred = pred.cpu()
    print('segmentation shape:', pred.shape)

    # Get binary mask.
    pred = pred.argmax(axis=1)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.
    print('pred shape:',  pred.shape)

    # Pad segmentation prediction.
    crop_or_padding = tuple((-d[0], -d[1]) for d in crop_or_padding)    # Reverse crop/padding amounts.
    print('reverse crop or padding:', crop_or_padding)
    pred = _asymmetric_crop_or_pad(pred, crop_or_padding)
    print('asymmetric crop or pad:', pred.shape)

    return pred

def _extract_patch(
    input: np.ndarray,
    pred: torch.Tensor,
    size: types.Size3D) -> Tuple[np.ndarray, types.Box3D]:
    """
    returns: a (patch, crop_or_padding) tuple where the patch is the extracted patch centred around the OAR,
        and the crop_or_padding tells us how much was added/removed at each end of each dimension.
    args:
        input: the input data.
        pred: the label data.
        size: the patch will be this size with the OAR in the centre. Must be larger than OAR extent.
    raises:
        ValueError: if the OAR extent is larger than the patch size.
    """
    # Find OAR extent.
    non_zero = np.argwhere(pred != 0).int()
    mins, _ = non_zero.min(axis=1)
    maxs, _ = non_zero.max(axis=1)
    oar_size = maxs - mins

    # Check oar size.        
    size = torch.Tensor(size)
    size = size.type(torch.int)
    if (oar_size > size).any():
        raise ValueError(f"OAR size '{oar_size}' larger than requested patch size '{size}'.")

    # Determine min/max indices of the patch.
    size_diff = size - oar_size
    lower_add = np.ceil(size_diff / 2).int()
    mins = mins - lower_add
    maxs = mins + size

    # Check for negative indices, and record padding.
    lower_pad = (-mins).clip(0) 
    mins = mins.clip(0)

    # Check for max indices larger than input size, and record padding.
    input_size = torch.Tensor(input.shape)
    input_size = input_size.type(torch.int)
    upper_pad = (maxs - input_size).clip(0)
    maxs = maxs - upper_pad

    # Perform crop.
    slices = tuple(slice(min.item(), max.item()) for min, max in zip(mins, maxs))
    input = input[slices]

    # Perform padding.
    padding = tuple(zip(lower_pad, upper_pad))
    input = np.pad(input, padding, mode='minimum')

    # Get crop or padding information.
    info = tuple((min.item(), max.item()) for min, max in zip(mins, maxs - input_size))

    return input, info

