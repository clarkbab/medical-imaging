import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, Union

from ..prediction import get_localiser_prediction
from mymi import dataset as ds
from mymi.geometry import get_box, get_extent, get_extent_centre, get_extent_width_mm
from mymi import logging
from mymi.loaders import Loader
from mymi.models.systems import Localiser, Segmenter
from mymi.transforms import crop_foreground_3D
from mymi.regions import RegionLimits, get_patch_size
from mymi.transforms import top_crop_or_pad_3D, crop_or_pad_3D, resample_3D
from mymi import types

def get_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.Model,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    device: Optional[torch.device] = None,
    truncate: bool = False) -> None:
    # Load data.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction(localiser, loc_size, loc_spacing, input, spacing, device=device, truncate=truncate)

    return pred

def create_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.Model,
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    device: Optional[torch.device] = None,
    truncate: bool = False) -> None:
    # Load dataset.
    set = ds.get(dataset, 'nifti')

    # Load localiser.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Make prediction - don't truncate saved predictions.
    pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device, truncate=truncate)

    # Save segmentation.
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser.name, f'{pat_id}.npz') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, data=pred)

def create_localiser_predictions(
    dataset: str,
    region: str,
    localiser: Tuple[str, str, str],
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float]) -> None:
    logging.info(f"Making localiser predictions for NIFTI dataset '{dataset}', region '{region}', localiser '{localiser}'.")

    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)

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

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    for pat in tqdm(pats):
        create_patient_localiser_prediction(dataset, pat, localiser, loc_size, loc_spacing, device=device, truncate=truncate)

def create_localiser_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: Tuple[str, str, str],
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    num_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {num_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Perform for specified folds
    sets = [ds.get(d, 'training') for d in datasets]
    if test_folds == 'all':
        test_folds = list(range(num_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(sets, region, num_folds=num_folds, test_fold=test_fold)

        # Make predictions.
        for datasets, pat_ids in tqdm(iter(test_loader), leave=False):
            if type(pat_ids) == torch.Tensor:
                pat_ids = pat_ids.tolist()
            for dataset, pat_id in zip(datasets, pat_ids):
                create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device, truncate=truncate)

def load_patient_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName) -> np.ndarray:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)

    # Load prediction.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', localiser '{localiser}'.")
    pred = np.load(filepath)['data']

    return pred

def load_patient_localiser_centre(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName) -> types.Point3D:
    seg = load_patient_localiser_prediction(dataset, pat_id, localiser)
    ext_centre = get_extent_centre(seg)
    return ext_centre

def get_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    region: str,
    loc_centre: types.Point3D,
    segmenter: types.Model,
    seg_spacing: types.ImageSpacing3D,
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    # Load model.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)
    segmenter.eval()
    segmenter.to(device)

    # Load patient CT data and spacing.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample_3D(input, spacing, seg_spacing) 

    # Get localiser centre on downsampled image.
    scale_factor = np.array(spacing) / seg_spacing
    loc_centre = np.round(tuple(np.array(scale_factor) * loc_centre)).astype(int)

    # Extract segmentation patch.
    resampled_size = input.shape
    patch_size = get_patch_size(region, seg_spacing)
    patch = get_box(loc_centre, patch_size)
    input = crop_or_pad_3D(input, patch, fill=input.min())

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = segmenter(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Crop/pad to the resampled size, i.e. before patch extraction.
    rev_patch_min, rev_patch_max = patch
    rev_patch_min = tuple(-np.array(rev_patch_min))
    rev_patch_max = tuple(np.array(rev_patch_min) + resampled_size)
    rev_patch_box = (rev_patch_min, rev_patch_max)
    pred = crop_or_pad_3D(pred, rev_patch_box)

    # Resample to original spacing.
    pred = resample_3D(pred, seg_spacing, spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    return pred

def create_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    segmenter: types.Model,
    seg_spacing: types.ImageSpacing3D,
    device: Optional[torch.device] = None) -> None:
    # Load dataset.
    set = ds.get(dataset, 'nifti')

    # Load localiser/segmenter.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Get segmenter prediction.
    loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
    if loc_centre is None:
        # Create empty pred.
        ct_data = set.patient(pat_id).ct_data
        pred = np.zeros_like(ct_data, dtype=bool) 
    else:
        pred = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, seg_spacing, device=device)

    # Save segmentation.
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter.name, f'{pat_id}.npz') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, data=pred)

def create_segmenter_predictions(
    dataset: str,
    region: str,
    localiser: Tuple[str, str, str],
    segmenter: Tuple[str, str, str],
    seg_spacing: Tuple[float, float, float]) -> None:
    logging.info(f"Making segmenter predictions for NIFTI dataset '{dataset}', region '{region}', segmenter '{segmenter}'.")

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

    # Load segmenter.
    segmenter = Segmenter.load(*segmenter)

    for pat in tqdm(pats):
        create_patient_segmenter_prediction(dataset, pat, region, localiser, segmenter, seg_spacing, device=device)

def create_segmenter_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    seg_spacing: types.ImageSpacing3D,
    num_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter.name}', with {num_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Perform for specified folds
    sets = [ds.get(d, 'training') for d in datasets]
    if test_folds == 'all':
        test_folds = list(range(num_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(sets, region, num_folds=num_folds, test_fold=test_fold)

        # Make predictions.
        for datasets, pat_ids in tqdm(iter(test_loader), leave=False):
            if type(pat_ids) == torch.Tensor:
                pat_ids = pat_ids.tolist()
            for dataset, pat_id in zip(datasets, pat_ids):
                create_patient_segmenter_prediction(dataset, pat_id, region, localiser, segmenter, seg_spacing, device=device)

def load_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName) -> np.ndarray:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'.")
    npz_file = np.load(filepath)
    seg = npz_file['data']
    
    return seg

def save_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    data: np.ndarray) -> None:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    np.savez_compressed(filepath, data=data)

def create_two_stage_predictions(
    dataset: str,
    region: str,
    localiser: types.ModelName,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    segmenter: types.ModelName,
    seg_spacing: types.ImageSpacing3D) -> None:
    logging.info(f"Making two-stage predictions for NIFTI dataset '{dataset}', region '{region}', segmenter '{segmenter}'.")

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

    # Load segmenter.
    segmenter = Segmenter.load(*segmenter)

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    for pat in tqdm(pats):
        create_patient_localiser_prediction(dataset, pat, localiser, loc_size, loc_spacing, device=device, truncate=truncate)
        create_patient_segmenter_prediction(dataset, pat, region, localiser.name, segmenter, seg_spacing, device=device)

def create_two_stage_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    segmenter: types.ModelName,
    seg_spacing: types.ImageSpacing3D,
    num_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', segmenter '{segmenter.name}', with {num_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Perform for specified folds
    sets = [ds.get(d, 'training') for d in datasets]
    if test_folds == 'all':
        test_folds = list(range(num_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(sets, region, num_folds=num_folds, test_fold=test_fold)

        # Make predictions.
        for datasets, pat_ids in tqdm(iter(test_loader), leave=False):
            if type(pat_ids) == torch.Tensor:
                pat_ids = pat_ids.tolist()
            for dataset, pat_id in zip(datasets, pat_ids):
                # Create predictions.
                create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device, truncate=truncate)
                create_patient_segmenter_prediction(dataset, pat_id, region, localiser.name, segmenter, seg_spacing, device=device)
