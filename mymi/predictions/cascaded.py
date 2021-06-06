import numpy as np
import pydicom as dcm
import torch
from torch import nn
from typing import *

from mymi import types

from .localisation import get_patient_bounding_box
from .segmentation import get_patient_patch_segmentation

def get_patient_segmentation(
    id: Union[str, int],
    localiser: nn.Module,
    localiser_size: types.Size3D,
    localiser_spacing: types.Spacing3D,
    segmenter: nn.Module,
    segmenter_size: types.Size3D,
    segmenter_spacing: types.Spacing3D,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    returns: an RTSTRUCT dicom file containing the predictions made by the model.
    args:
        ds: the dataset name.
        patient: the patient ID.
        localiser: the localiser model.
        localiser_size: the input size of the localiser network.
        localiser_spacing: the voxel spacing of the localiser network input layer.
        segmenter: the segmenter model.
        segmenter_size: the input size of the segmenter network.
        segmenter_spacing: the voxel spacing of the segmenter network input layer.
    kwargs:
        clear_cache: force the cache to clear.
        device: the device to perform network calcs on.
    """
    # Get the OAR bounding box.
    bounding_box = get_patient_bounding_box(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device)

    # Get segmentation prediction.
    seg = get_patient_patch_segmentation(id, bounding_box, segmenter, segmenter_size, segmenter_spacing, clear_cache=clear_cache, device=device)

    return seg
