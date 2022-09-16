import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Literal, Optional, Tuple, Union

from ..prediction import get_localiser_prediction
from mymi import config
from mymi import dataset as ds
from mymi.geometry import get_box, get_extent, get_extent_centre, get_extent_width_mm
from mymi import logging
from mymi.loaders import Loader
from mymi.models import replace_checkpoint_alias
from mymi.models.systems import Localiser, Segmenter
from mymi.transforms import crop_foreground_3D
from mymi.regions import RegionLimits, get_region_patch_size
from mymi.transforms import top_crop_or_pad_3D, crop_or_pad_3D, resample_3D
from mymi import types

def get_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.Model,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None,
    truncate: bool = False) -> np.ndarray:
    # Load data.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device, truncate=truncate)

    return pred

def create_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: Union[types.Model, types.ModelName],
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
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
    localiser: types.Model,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4)) -> None:
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
    localiser: types.Model,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device, truncate=truncate)

def load_patient_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    raise_error: bool = True) -> Optional[np.ndarray]:
    localiser = replace_checkpoint_alias(*localiser)

    # Load prediction.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', localiser '{localiser}'.")
        else:
            return None
    pred = np.load(filepath)['data']

    return pred

def load_patient_localiser_centre(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    raise_error: bool = True) -> types.Point3D:
    seg = load_patient_localiser_prediction(dataset, pat_id, localiser, raise_error=raise_error)
    if not raise_error and seg is None:
        return None
    ext_centre = get_extent_centre(seg)
    return ext_centre

def get_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    region: str,
    loc_centre: types.Point3D,
    segmenter: Union[types.Model, types.ModelName],
    probs: bool = False,
    seg_spacing: types.ImageSpacing3D = (1, 1, 2),
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
    input = resample_3D(input, spacing=spacing, output_spacing=seg_spacing) 

    # Get localiser centre on downsampled image.
    scaling = np.array(spacing) / seg_spacing
    loc_centre = tuple(int(el) for el in scaling * loc_centre)

    # Extract segmentation patch.
    resampled_size = input.shape
    patch_size = get_region_patch_size(region, seg_spacing)
    patch = get_box(loc_centre, patch_size)
    input = crop_or_pad_3D(input, patch, fill=input.min())

    # Pass patch to segmenter.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = segmenter(input, probs=probs)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Crop/pad to the resampled size, i.e. before patch extraction.
    rev_patch_min, rev_patch_max = patch
    rev_patch_min = tuple(-np.array(rev_patch_min))
    rev_patch_max = tuple(np.array(rev_patch_min) + resampled_size)
    rev_patch_box = (rev_patch_min, rev_patch_max)
    pred = crop_or_pad_3D(pred, rev_patch_box)

    # Resample to original spacing.
    pred = resample_3D(pred, spacing=seg_spacing, output_spacing=spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_3D(pred, crop_box)

    return pred

def create_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    segmenter: Union[types.Model, types.ModelName],
    probs: bool = False,
    raise_error: bool = False,
    seg_spacing: types.ImageSpacing3D = (1, 1, 2),
    device: Optional[torch.device] = None) -> None:
    localiser = replace_checkpoint_alias(*localiser)

    # Load dataset.
    set = ds.get(dataset, 'nifti')

    # Load segmenter.
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
        if raise_error:
            raise ValueError(f"No 'loc_centre' returned from localiser.")
        else:
            ct_data = set.patient(pat_id).ct_data
            pred = np.zeros_like(ct_data, dtype=bool) 
    else:
        pred = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, seg_spacing=seg_spacing, device=device)

    # Save segmentation.
    if probs:
        filename = f'{pat_id}-prob.npz'
    else:
        filename = f'{pat_id}.npz'
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter.name, filename) 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, data=pred)

def create_segmenter_predictions(
    dataset: str,
    region: str,
    localiser: types.Model,
    segmenter: types.Model,
    seg_spacing: types.ImageSpacing3D = (1, 1, 2)) -> None:
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
    seg_spacing: types.ImageSpacing3D = (1, 1, 2),
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            create_patient_segmenter_prediction(dataset, pat_id, region, localiser, segmenter, seg_spacing=seg_spacing, device=device)

def load_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    raise_error: bool = True) -> Optional[np.ndarray]:
    localiser = replace_checkpoint_alias(*localiser)
    segmenter = replace_checkpoint_alias(*segmenter)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    if config.environ('PETER_MAC_HACK') == 'True':
        if dataset == 'PMCC-HN-TEST':
            pred_path = 'S:\\ImageStore\\AtlasSegmentation\\BC_HN\\nifti\\test'
        elif dataset == 'PMCC-HN-TRAIN':
            pred_path = 'S:\\ImageStore\\AtlasSegmentation\\BC_HN\\nifti\\train'
    else:
        pred_path = os.path.join(set.path, 'predictions')
    filepath = os.path.join(pred_path, 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'. Path: {filepath}")
        else:
            return None
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
    segmenter: types.ModelName,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    seg_spacing: types.ImageSpacing3D = (1, 1, 2)) -> None:
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
    segmenter: types.ModelName,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    n_folds: Optional[int] = None,
    seg_spacing: types.ImageSpacing3D = (1, 1, 2),
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(n_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

        # Make predictions.
        for dataset_b, pat_id_b in tqdm(iter(test_loader)):
            if type(pat_id_b) == torch.Tensor:
                pat_id_b = pat_id_b.tolist()
            for dataset, pat_id in zip(dataset_b, pat_id_b):
                create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device, truncate=truncate)
                create_patient_segmenter_prediction(dataset, pat_id, region, localiser.name, segmenter, seg_spacing, device=device)
