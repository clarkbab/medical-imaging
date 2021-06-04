import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Tuple

from mymi import dataset
from mymi.transforms import crop_or_pad, resample

def get_patient_segmentation(
    id: str,
    bounding_box: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    segmenter: nn.Module,
    segmenter_size: Tuple[int, int, int],
    segmenter_spacing: Tuple[float, float, float],
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    returns: the segmentation for the patient.
    args:
        segmenter: the segmentation network.
    kwargs:
        clear_cache: forces the cache to clear.
        device: the device to use for network calcs.
    """
    # Load patient CT data and spacing.
    patient = dataset.patient(id)
    input = patient.ct_data(clear_cache=clear_cache)
    spacing = patient.spacing(clear_cache=clear_cache)

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample(input, spacing, segmenter_spacing) 

    # Resample bounding box to segmenter spacing.
    mins, widths = bounding_box
    bbox_label = np.zeros(input_size, dtype=bool)
    indices = tuple(slice(m, m + w) for m, w in zip(mins, widths))
    bbox_label[indices] = 1
    bbox_label = resample(bbox_label, spacing, segmenter_spacing)

    # Get new bounding box.
    non_zero = np.argwhere(bbox_label != 0)
    mins = non_zero.min(axis=0)
    maxs = non_zero.max(axis=0)
    widths = maxs - mins
    bounding_box = (tuple(mins), tuple(widths))

    # Extract patch around bounding box.
    input, crop_or_padding = _extract_patch(input, bounding_box, size=segmenter_size)

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with autocast(enabled=True):
        pred = segmenter(input)
    pred = pred.cpu()

    # Get binary mask.
    pred = pred.argmax(axis=1)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Crop or pad to pre-patch-extraction.
    crop_or_padding = tuple((-d[0], -d[1]) for d in crop_or_padding)    # Reverse crop/padding amounts.
    pred = crop_or_pad(pred, crop_or_padding)

    # Resample to original spacing.
    pred = resample(pred, segmenter_spacing, spacing)

    return pred

def _extract_patch(
    input: np.ndarray,
    bounding_box: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    size: Tuple[int, int, int]) -> np.ndarray:
    """
    returns: a patch of the input data that is centered on the bounding box.
    args:
        input: the input data.
        pred: the label data.
        size: the patch will be this size with the OAR in the centre. Must be larger than OAR extent.
    raises:
        ValueError: if the OAR extent is larger than the patch size.
    """
    # Check bounding box size.
    size = np.array(size, dtype=int)
    mins = np.array(bounding_box[0], dtype=int)
    widths = np.array(bounding_box[1], dtype=int)
    if (widths > size).any():
        raise ValueError(f"Bounding box size '{widths}' larger than patch size '{size}'.")

    # Determine min/max indices of the patch.
    size_diff = size - widths
    lower_add = np.ceil(size_diff / 2).astype(int)
    mins = mins - lower_add
    maxs = mins + size

    # Check for negative indices, and record padding.
    lower_pad = (-mins).clip(0) 
    mins = mins.clip(0)

    # Check for max indices larger than input size, and record padding.
    input_size = np.array(input.shape, dtype=int)
    upper_pad = (maxs - input_size).clip(0)
    maxs = maxs - upper_pad

    # Perform crop.
    slices = tuple(slice(min.item(), max.item()) for min, max in zip(mins, maxs))
    input = input[slices]

    # Perform padding.
    padding = tuple(zip(lower_pad, upper_pad))
    input = np.pad(input, padding, mode='minimum')

    # Get crop or padding information.
    info = tuple((-min, max) for min, max in zip(mins, maxs - input_size))

    return input, info
