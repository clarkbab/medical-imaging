import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Union

from mymi import dataset as ds
from mymi import logging
from mymi.models.systems import Localiser
from mymi.postprocessing import get_extent_centre
from mymi.transforms import centre_crop_or_pad_3D, crop_or_pad_3D, resample_3D
from mymi import types

def get_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.Model,
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    raise_fov_error: bool = True) -> np.ndarray:
    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    localiser.eval()
    localiser.to(device)

    # Load the patient data.
    set = ds.get(dataset, 'nifti')
    input = set.patient(pat_id).ct_data()
    input_size = input.shape
    spacing = set.patient(pat_id).ct_spacing()

    # Check patient FOV.
    fov = np.array(input_size) * spacing
    loc_fov = np.array(loc_size) * loc_spacing
    for axis in len(fov):
        if fov[axis] > loc_fov[axis]:
            error_message = f"Patient FOV '{fov}', larger than localiser FOV '{loc_fov}'."
            if raise_fov_error:
                raise ValueError(error_message)
            else:
                logging.error(error_message)

    # Resample/crop data for network.
    input = resample_3D(input, spacing, loc_spacing)
    pre_crop_size = input.shape

    # Shape the image so it'll fit the network.
    input = centre_crop_or_pad_3D(input, loc_size, fill=input.min())

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
    pred = resample_3D(pred, loc_spacing, spacing)
    
    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    return pred

def create_localiser_predictions(
    dataset: str,
    localiser: Tuple[str, str, str],
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    region: str) -> None:
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
    localiser_args = Localiser.replace_best(*localiser)
    localiser = Localiser.load(*localiser)

    for pat in tqdm(pats):
        # Make prediction.
        seg = get_localiser_prediction(dataset, pat, localiser, loc_size, loc_spacing, device=device)

        # Save segmentation.
        filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser_args, f'{pat}.npz') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, data=seg)

def load_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    localiser = Localiser.replace_best(*localiser)
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction not found for dataset '{set}', localiser '{localiser}'.")
    seg = np.load(filepath)['data']
    return seg

def get_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    segmenter: types.Model,
    segmenter_size: types.ImageSize3D,
    centre: types.Point3D,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_patch: bool = False) -> np.ndarray:
    """
    returns: the segmentation for the patient.
    args:
        dataset: the dataset.
        pat_id: the patient ID.
        segmenter: the segmentation network.
        centre: the centre of the segmentation network in voxels at native resolution.
    kwargs:
        clear_cache: forces the cache to clear.
        device: the device to use for network calcs.
        return_patch: returns the box used for the segmentation.
    """
    # Load model if not already loaded.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)
    segmenter.eval()
    segmenter.to(device)
    segmenter_spacing = (1, 1, 2)

    # Load patient CT data and spacing.
    patient = dataset.patient(pat_id)
    input = patient.ct_data(clear_cache=clear_cache)
    spacing = patient.ct_spacing(clear_cache=clear_cache)

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample_3D(input, spacing, segmenter_spacing) 

    # Get centre on resampled image.
    scale = np.array(segmenter_spacing) / np.array(spacing)
    centre = tuple(np.floor(scale * centre).astype(int)) 

    # Extract patch around centre.
    pre_extract_size = input.shape
    input, patch_box = _extract_patch(input, segmenter_size, centre)

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = segmenter(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Pad (or crop) to size before patch extraction.
    rev_patch_box_min, rev_patch_box_max = patch_box
    rev_patch_box_max = tuple(np.array(pre_extract_size) - rev_patch_box_min)
    rev_patch_box_min = tuple(-np.array(rev_patch_box_min))
    rev_patch_box = (rev_patch_box_min, rev_patch_box_max)
    pred = crop_or_pad_3D(pred, rev_patch_box)

    # Resample to original spacing.
    pred = resample_3D(pred, segmenter_spacing, spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    return pred

def get_two_stage_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.Model,
    segmenter: types.Model,
    segmenter_size: types.ImageSize3D,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    # Get localiser segmentation.
    seg = get_localiser_prediction(dataset, pat_id, localiser, clear_cache=clear_cache, device=device)

    # Find seg centre.
    centre = get_extent_centre(seg)

    # Get segmentation prediction.
    seg = get_segmenter_prediction(dataset, pat_id, segmenter, segmenter_size, centre, clear_cache=clear_cache, device=device)

    return seg

def _extract_patch(
    input: np.ndarray,
    size: types.ImageSize3D,
    box: types.Box3D) -> Tuple[np.ndarray, types.Box3D]:
    """
    returns: a patch of size 'size' centred on the bounding box. Also returns the bounding
        box that was used to extract the patch, relative to the input size.
    args:
        input: the input data.
        size: the extent of the patch.
        centre: the patch centre.
    raises:
        ValueError: if the OAR extent is larger than the patch size.
    """
    # Check bounding box size.
    size = np.array(size)
    min, max = box
    min = np.array(min)
    max = np.array(max)
    width = max - min
    if (width > size).any():
        raise ValueError(f"Bounding box size '{width}' larger than patch size '{size}'.")

    # Determine min/max indices of the patch.
    size = np.array(size)
    lower_sub = np.ceil(size_diff / 2).astype(int)
    min = tuple(centre - lower_sub)
    max = tuple(min + size)
    
    # Perform the crop or pad.
    input_size = input.shape
    box = (min, max)
    patch = crop_or_pad_3D(input, box, fill=input.min())

    return patch, box
