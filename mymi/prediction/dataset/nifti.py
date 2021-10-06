import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Union

from mymi import dataset as ds
from mymi import logging
from mymi.models.systems import Localiser
from mymi.transforms import centre_crop_or_pad_3D, crop_or_pad_3D, resample_3D
from mymi import types

def create_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.Model,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_seg: bool = False) -> Union[Optional[types.Box3D], Tuple[Optional[types.Box3D], np.ndarray]]:
    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    localiser.eval()
    localiser.to(device)
    localiser_size = (128, 128, 96)
    localiser_spacing = (4, 4, 6.625)

    # Load the patient data.
    set = ds.get(dataset, 'nifti')
    input = set.patient(pat_id).ct_data()
    input_size = input.shape
    spacing = set.patient(pat_id).ct_spacing()

    # Resample/crop data for network.
    input = resample_3D(input, spacing, localiser_spacing)
    pre_crop_size = input.shape

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

    # Reverse the resample/crop.
    pred = centre_crop_or_pad_3D(pred, pre_crop_size)
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

def create_localiser_predictions(
    dataset: str,
    localiser: Tuple[str, str, str],
    region: str,
    clear_cache: bool = False) -> None:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=region)

    # Load models.
    localiser_args = localiser
    localiser = Localiser.load(*localiser)

    for pat in tqdm(pats):
        # Make prediction.
        _, data = create_localiser_prediction(dataset, pat, localiser, device=device, return_seg=True)

        # Save in folder.
        filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser_args, f"{pat}.npz") 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, data=data)

def get_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser, f"{pat_id}.npz") 
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction for dataset '{set}', localiser '{localiser}' not found.")
    data = np.load(filepath)['data']
    return data
