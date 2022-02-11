import numpy as np
import torch
from typing import Optional

from mymi import logging
from mymi.geometry import get_extent, get_extent_width_mm
from mymi.models.systems import Localiser
from mymi.regions import RegionLimits
from mymi.transforms import crop_foreground_3D, crop_or_pad_3D, resample_3D, top_crop_or_pad_3D
from mymi import types

def get_localiser_prediction(
    localiser: types.Model,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    input: np.ndarray,
    spacing: types.ImageSpacing3D,
    device: Optional[torch.device] = None,
    truncate: bool = False) -> np.ndarray:
    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    localiser.eval()
    localiser.to(device)

    # Save input size.
    input_size = input.shape

    # Resample/crop data for network.
    resample = True if spacing != loc_spacing else False
    if resample:
        input = resample_3D(input, spacing, loc_spacing)

    # Crop the image so it won't overflow network memory. Perform 'top' crop
    # as we're interested in the cranial end of z-axis.
    pre_crop_size = input.shape
    input = top_crop_or_pad_3D(input, loc_size, fill=input.min())

    # Get localiser result.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = localiser(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Reverse the crop.
    pred = top_crop_or_pad_3D(pred, pre_crop_size)

    # Reverse the resample.
    if resample:
        pred = resample_3D(pred, loc_spacing, spacing)
    
    # Crop to input size to clean up any resampling rounding errors.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    # Perform truncation to 'SpinalCord' z-extent.
    if truncate:
        ext_width = get_extent_width_mm(pred, spacing)
        if ext_width and ext_width[2] > RegionLimits.SpinalCord[2]:
            # Crop caudal end of spine.
            logging.warning(f"Cropping bottom end of 'SpinalCord'. Got z-extent width '{ext_width[2]}mm', maximum '{RegionLimits.SpinalCord[2]}mm'.")
            top_z = get_extent(pred)[1][2]
            bottom_z = int(np.ceil(top_z - RegionLimits.SpinalCord[2] / spacing[2]))
            crop = ((0, 0, bottom_z), tuple(np.array(pred.shape) - 1))
            pred = crop_foreground_3D(pred, crop)

    return pred
