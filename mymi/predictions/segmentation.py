import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Tuple, Union

from mymi import dataset
from mymi.transforms import crop_or_pad_3D, resample_box_3D, resample_3D
from mymi import types

def get_patient_segmentation(
    id: str,
    bounding_box: types.Box3D,
    segmenter: nn.Module,
    segmenter_size: types.Size3D,
    segmenter_spacing: types.Spacing3D,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_patch: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, types.Box3D]]:
    """
    returns: the segmentation for the patient.
    args:
        id: the patient ID.
        bounding_box: the box from localisation.
        segmenter: the segmentation network.
        segmenter_size: the input size expected by the segmenter.
        segmenter_spacing: the voxel spacing expected by the segmenter.
    kwargs:
        clear_cache: forces the cache to clear.
        device: the device to use for network calcs.
        return_patch: returns the box used for the segmentation.
    """
    # Load patient CT data and spacing.
    patient = dataset.patient(id)
    input = patient.ct_data(clear_cache=clear_cache)
    spacing = patient.spacing(clear_cache=clear_cache)

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample_3D(input, spacing, segmenter_spacing) 

    # Resample the localisation bounding box.
    bounding_box = resample_box_3D(bounding_box, spacing, segmenter_spacing)

    # Extract patch around bounding box.
    pre_extract_size = input.shape
    input, patch_box = _extract_patch(input, bounding_box, size=segmenter_size)

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

    # Crop/pad to size before patch extraction.
    bbox_min, bbox_max = bounding_box
    bbox_min = -np.array(bbox_min)
    bbox_max = np.array(pre_extract_size) - bbox_max
    bounding_box = (bbox_min, bbox_max)
    pred = crop_or_pad_3D(pred, bounding_box)

    # Resample to original spacing.
    pred = resample_3D(pred, segmenter_spacing, spacing)

    # Resample patch box to original spacing.
    patch_box = resample_box_3D(patch_box, segmenter_spacing, spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    # Get result.
    if return_patch:
        return (pred, patch_box)
    else:
        return pred

def _extract_patch(
    input: np.ndarray,
    bounding_box: types.Box3D,
    size: types.Size3D) -> Tuple[np.ndarray, types.Box3D]:
    """
    returns: a patch of size 'size' centred on the bounding box. Also returns the bounding
        box that was used to extract the patch, relative to the input size.
    args:
        input: the input data.
        bounding_box: the bounding box of the OAR.
        size: the patch will be this size with the OAR in the centre. Must be larger than OAR extent.
    raises:
        ValueError: if the OAR extent is larger than the patch size.
    """
    # Check bounding box size.
    size = np.array(size)
    min, max = bounding_box
    min = np.array(min)
    max = np.array(max)
    width = max - min
    if (width > size).any():
        raise ValueError(f"Bounding box size '{width}' larger than patch size '{size}'.")

    # Determine min/max indices of the patch.
    size_diff = size - width
    lower_add = np.ceil(size_diff / 2).astype(int)
    min = min - lower_add
    max = min + size
    
    # Perform the crop or pad.
    input_size = input.shape
    patch_box = (tuple(min), tuple(max))
    input = crop_or_pad_3D(input, patch_box, fill=input.min())

    return input, patch_box
