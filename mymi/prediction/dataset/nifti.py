import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from mymi import dataset as ds
from mymi.geometry import get_box, get_extent, get_extent_centre, get_extent_width_mm
from mymi import logging
from mymi.loaders import Loader
from mymi.models.systems import Localiser, Segmenter
from mymi.transforms import crop_foreground_3D
from mymi.regions import RegionLimits, get_patch_size
from mymi.transforms import centre_crop_or_pad_3D, crop_or_pad_3D, resample_3D
from mymi import types

def get_patient_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.Model,
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    device: torch.device = torch.device('cpu'),
    raise_fov_error: bool = False) -> np.ndarray:
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
    if np.minimum(loc_fov - fov, 0).sum() != 0:
        error_message = f"Patient '{pat_id}' FOV '{fov}', larger than localiser FOV '{loc_fov}'."
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

def create_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.Model,
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    device: Optional[torch.device] = None) -> None:
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

    # Make prediction.
    seg = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device)

    # Save segmentation.
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser.name, f'{pat_id}.npz') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, data=seg)

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

    for pat in tqdm(pats):
        create_patient_localiser_prediction(dataset, pat, localiser, loc_size, loc_spacing, device=device)

def create_localiser_predictions_from_loader(
    datasets: Union[str, List[str]],
    localiser: Tuple[str, str, str],
    loc_size: Tuple[int, int, int],
    loc_spacing: Tuple[float, float, float],
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', localiser '{localiser.name}'...")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create test loader.
    sets = [ds.get(d, 'training') for d in datasets]
    _, _, test_loader = Loader.build_loaders(sets, num_folds=num_folds, test_fold=test_fold)

    # Make predictions.
    for datasets, pat_ids in iter(test_loader):
        if type(pat_ids) == torch.Tensor:
            pat_ids = pat_ids.tolist()
        for dataset, pat_id in zip(datasets, pat_ids):
            create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device)

def load_patient_localiser_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    truncate_spine: bool = False) -> np.ndarray:
    localiser = Localiser.replace_best(*localiser)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'localiser', *localiser, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', localiser '{localiser}'.")
    seg = np.load(filepath)['data']

    # Perform truncation of 'SpinalCord'.
    if truncate_spine:
        spacing = ds.get(dataset, 'nifti').patient(pat_id).ct_spacing()
        ext_width = get_extent_width_mm(seg, spacing)
        if ext_width[2] > RegionLimits.SpinalCord[2]:
            # Crop caudal end of spine.
            logging.error(f"Cropping bottom end of 'SpinalCord' for patient '{pat_id}'. Got z-extent width '{ext_width[2]}mm', maximum '{RegionLimits.SpinalCord[2]}mm'.")
            top_z = get_extent(seg)[1][2]
            bottom_z = int(np.ceil(top_z - RegionLimits.SpinalCord[2] / spacing[2]))
            crop = ((0, 0, bottom_z), tuple(np.array(seg.shape) - 1))
            seg = crop_foreground_3D(seg, crop)

    return seg

def load_patient_localiser_centre(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    truncate_spine: bool = False) -> types.Point3D:
    seg = load_patient_localiser_prediction(dataset, pat_id, localiser, truncate_spine=truncate_spine)
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
    input = patient.ct_data()
    spacing = patient.ct_spacing()

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
    device: Optional[torch.device] = None,
    truncate_spine: bool = False) -> None:
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
    if region != 'SpinalCord':
        truncate_spine = False
    loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser, truncate_spine=truncate_spine)
    seg = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, seg_spacing, device=device)

    # Save segmentation.
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter.name, f'{pat_id}.npz') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, data=seg)

def create_segmenter_predictions(
    dataset: str,
    region: str,
    localiser: Tuple[str, str, str],
    segmenter: Tuple[str, str, str],
    seg_spacing: Tuple[float, float, float],
    truncate_spine: bool = False) -> None:
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
        if region != 'SpinalCord':
            truncate_spine = False
        create_patient_segmenter_prediction(dataset, pat, region, localiser, segmenter, seg_spacing, device=device, truncate_spine=truncate_spine)

def create_segmenter_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: Tuple[str, str, str],
    segmenter: Tuple[str, str, str],
    seg_spacing: types.ImageSpacing3D,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    truncate_spine: bool = False) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    logging.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', localiser '{localiser.name}', segmenter '{segmenter.name}'...")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create test loader.
    sets = [ds.get(d, 'training') for d in datasets]
    _, _, test_loader = Loader.build_loaders(sets, num_folds=num_folds, test_fold=test_fold)

    # Make predictions.
    for dataset, pat_id in iter(test_loader):
        create_patient_segmenter_prediction(dataset, pat_id, region, localiser, segmenter, seg_spacing, device=device, truncate_spine=truncate_spine)

def load_patient_segmenter_prediction(
    dataset: str,
    pat_id: types.PatientID,
    localiser: types.ModelName,
    segmenter: types.ModelName) -> np.ndarray:
    localiser = Localiser.replace_best(*localiser)
    segmenter = Segmenter.replace_best(*segmenter)

    # Load segmentation.
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'.")
    npz_file = np.load(filepath)
    seg = npz_file['data']
    
    return seg

def create_patient_two_stage_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.Model,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    segmenter: types.Model,
    seg_spacing: types.ImageSpacing3D,
    device: Optional[torch.device] = None,
    truncate_spine: bool = False) -> None:
    # Load dataset.
    set = ds.get(dataset, 'nifti')

    # Load localiser/segmenter.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
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

    # Create predictions.
    create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, device=device)
    create_patient_segmenter_prediction(dataset, pat_id, region, localiser.name, segmenter, seg_spacing, device=device, truncate_spine=truncate_spine)

def create_two_stage_predictions(
    dataset: str,
    region: str,
    localiser: types.ModelName,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    segmenter: types.ModelName,
    seg_spacing: types.ImageSpacing3D,
    truncate_spine: bool = False) -> None:
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

    for pat in tqdm(pats):
        create_patient_two_stage_prediction(dataset, pat, region, localiser, loc_size, loc_spacing, segmenter, seg_spacing, device=device, truncate_spine=truncate_spine)

def create_two_stage_predictions_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    segmenter: types.ModelName,
    seg_spacing: types.ImageSpacing3D,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    truncate_spine: bool = False) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making two-stage predictions for NIFTI datasets '{datasets}', localiser '{localiser.name}', segmenter '{segmenter.name}'...")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create test loader.
    sets = [ds.get(d, 'training') for d in datasets]
    _, _, test_loader = Loader.build_loaders(sets, region, num_folds=num_folds, test_fold=test_fold)

    # Make predictions.
    for datasets, pat_ids in tqdm(iter(test_loader)):
        if type(pat_ids) == torch.Tensor:
            pat_ids = pat_ids.tolist()
        for dataset, pat_id in zip(datasets, pat_ids):
            create_patient_two_stage_prediction(dataset, pat_id, region, localiser, loc_size, loc_spacing, segmenter, seg_spacing, device=device, truncate_spine=truncate_spine)
