import numpy as np
import pytorch_lightning as pl
import torch
from typing import Optional

from mymi import logging
from mymi.geometry import extent, extent_width_mm
from mymi.models.lightning_modules import Localiser
from mymi.regions import RegionLimits
from mymi.transforms import crop_foreground, crop_or_pad, resample, top_crop_or_pad
from mymi import typing

def get_localiser_prediction(
    input: np.ndarray,
    spacing: typing.ImageSpacing3D,
    localiser: pl.LightningModule,
    loc_size: typing.ImageSize3D = (128, 128, 150),
    loc_spacing: typing.ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
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
        localiser = Localiser.load(*localiser, check_epochs=False, map_location=device)
    localiser.eval()
    localiser.to(device)

    # Save input size.
    input_size = input.shape

    # Resample/crop data for network.
    use_resample = True if spacing != loc_spacing else False
    if use_resample:
        input = resample(input, spacing=spacing, output_spacing=loc_spacing)

    # Crop the image so it won't overflow network memory. Perform 'top' crop
    # as we're interested in the cranial end of z-axis.
    pre_crop_size = input.shape
    input = top_crop_or_pad(input, loc_size, fill=input.min())

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
    pred = top_crop_or_pad(pred, pre_crop_size)

    # Reverse the resample.
    if use_resample:
        pred = resample(pred, spacing=loc_spacing, output_spacing=spacing)
    
    # Crop to input size to clean up any resampling rounding errors.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad(pred, crop_box)

    return pred

def get_localiser_prediction_at_training_resolution(
    input: np.ndarray,
    spacing: typing.ImageSpacing3D,
    localiser: pl.LightningModule,
    loc_size: typing.ImageSize3D = (128, 128, 150),
    loc_spacing: typing.ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
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
    use_resample = True if spacing != loc_spacing else False
    if use_resample:
        input = resample(input, spacing=spacing, output_spacing=loc_spacing)

    # Crop the image so it won't overflow network memory. Perform 'top' crop
    # as we're interested in the cranial end of z-axis.
    pre_crop_size = input.shape
    input = top_crop_or_pad(input, loc_size, fill=input.min())

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
    pred = top_crop_or_pad(pred, pre_crop_size)

    return pred
