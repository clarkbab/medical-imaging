from itertools import chain
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from typing import *

from mymi import config
from mymi.datasets import NiftiDataset, TrainingDataset
from mymi.geometry import get_box, extent, centre_of_extent
from mymi.loaders import Loader, MultiLoader
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.lightning_modules import Segmenter
from mymi.postprocessing import largest_cc_4D
from mymi.regions import RegionNames, get_region_patch_size, regions_to_list, truncate_spine
from mymi.transforms import centre_crop_or_pad, crop_or_pad
from mymi.typing import *
from mymi.utils import *

def get_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: pl.LightningModule,
    loc_size: ImageSize3D = (128, 128, 150),
    loc_spacing: ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
    # Load data.
    set = NiftiDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction_base(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device)

    return pred

def get_localiser_prediction_at_training_resolution(
    dataset: str,
    pat_id: str,
    localiser: pl.LightningModule,
    loc_size: ImageSize3D = (128, 128, 150),
    loc_spacing: ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
    # Load data.
    set = NiftiDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction_at_training_resolution_base(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device)

    return pred

def create_localiser_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[PatientID, List[PatientID]],
    localiser: Union[ModelName, pl.LightningModule],
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, (int, str), out_type=str)
    datasets = arg_broadcast(datasets, pat_ids)
    assert len(datasets) == len(pat_ids)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load localiser.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser, map_location=device, **kwargs)

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NiftiDataset(dataset)
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser.name}'.")

        # Make prediction.
        pred = get_localiser_prediction(dataset, pat_id, localiser, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, pat_id, *localiser.name, 'pred.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_localiser_prediction_at_training_resolution(
    dataset: Union[str, List[str]],
    pat_id: Union[PatientID, List[PatientID]],
    localiser: Union[ModelName, pl.LightningModule],
    check_epochs: bool = True,
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, (int, str), out_type=str)
    datasets = arg_broadcast(datasets, pat_ids)
    assert len(datasets) == len(pat_ids)

    # Load localiser.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser, check_epochs=check_epochs)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NiftiDataset(dataset)
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser.name}'.")

        # Make prediction.
        pred = get_localiser_prediction_at_training_resolution(dataset, pat_id, localiser, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, pat_id, *localiser.name, 'pred-at-training-resolution.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_localiser_predictions_for_first_n_pats(
    n_pats: int,
    region: str,
    localiser: ModelName,
    savepath: Optional[str] = None) -> None:
    localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NIFTI datasets for region '{region}', first '{n_pats}' patients in 'all-patients.csv'.")

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Get dataset/patient IDs.
    create_localiser_prediction(*df, localiser, device=device, savepath=savepath)
    
def create_localiser_predictions_v2(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    n_epochs: int = np.inf,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser, n_epochs=n_epochs)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Get patient IDs from original evaluation.
    # We have to evaluate the segmenter using the original evaluation patient IDs
    # as our 'Loader' now returns different patients per fold.
    orig_localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')
    orig_localiser = replace_ckpt_alias(orig_localiser)
    segmenter = (f'segmenter-{region}-v2', localiser.name[1], 'best')
    segmenter = replace_ckpt_alias(segmenter)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *orig_localiser, *segmenter, encode(datasets), f'{filename}.csv')
    orig_df = pd.read_csv(filepath, dtype={'patient-id': str})
    orig_df = orig_df[['dataset', 'patient-id']].drop_duplicates()

    for i, row in tqdm(orig_df.iterrows()):
        dataset, pat_id = row['dataset'], row['patient-id']

        # Timing table data.
        data = {
            'fold': test_fold,
            'dataset': dataset,
            'patient-id': pat_id,
            'region': region,
            'device': device.type
        }

        with timer.record(data, enabled=timing):
            create_localiser_prediction(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_all_localiser_predictions(
    dataset: Union[str, List[str]],
    localiser: ModelName,
    check_epochs: bool = True,
    n_epochs: int = np.inf,
    timing: bool = True) -> None:
    logging.arg_log('Making localiser predictions', ('dataset', 'localiser'), (dataset, localiser))
    datasets = arg_to_list(dataset, str)
    localiser = Localiser.load(*localiser, check_epochs=check_epochs, n_epochs=n_epochs)

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'localiser': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patients.
    for dataset in datasets:
        set = NiftiDataset(dataset)
        pat_ids = set.list_patients()

        for pat_id in tqdm(pat_ids):
            # Timing table data.
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'localiser': str(localiser),
                'device': device.type
            }

            with timer.record(data, enabled=timing):
                create_localiser_prediction(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), *localiser.name, f'timing-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_all_localiser_predictions_at_training_resolution(
    dataset: Union[str, List[str]],
    localiser: ModelName,
    check_epochs: bool = True,
    n_epochs: int = np.inf,
    timing: bool = True) -> None:
    logging.arg_log('Making localiser predictions at training resolution', ('dataset', 'localiser'), (dataset, localiser))
    datasets = arg_to_list(dataset, str)
    localiser = Localiser.load(*localiser, check_epochs=check_epochs, n_epochs=n_epochs)

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'localiser': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patients.
    for dataset in datasets:
        set = NiftiDataset(dataset)
        pat_ids = set.list_patients()

        for pat_id in tqdm(pat_ids):
            # Timing table data.
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'localiser': str(localiser),
                'device': device.type
            }

            with timer.record(data, enabled=timing):
                create_localiser_prediction_at_training_resolution(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), *localiser.name, f'timing-at-training-resolution-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_localiser_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    n_epochs: int = np.inf,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser, n_epochs=n_epochs)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Timing table data.
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'device': device.type
            }

            with timer.record(data, enabled=timing):
                create_localiser_prediction(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_localiser_predictions_at_training_resolution(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    n_epochs: int = np.inf,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser, n_epochs=n_epochs)
    logging.info(f"Making localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Timing table data.
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'device': device.type
            }

            with timer.record(data, enabled=timing):
                create_localiser_prediction_at_training_resolution(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-at-training-resolution-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def get_multi_segmenter_prediction_nnunet_bootstrap(
    dataset: str,
    pat_id: PatientID,
    model: Union[ModelName, pl.LightningModule],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    device: torch.device = torch.device('cpu'),
    crop_mm: Optional[Box3D] = None,
    crop_type: str = 'brain',
    **kwargs) -> np.ndarray:
    model_regions = arg_to_list(model_region, str)

    # Load model.
    if type(model) == tuple:
        model = MultiSegmenter.load(*model, **kwargs)
        model.eval()
        model.to(device)

    # Load registered (!) patient CT data and spacing.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    pat_id_pt = pat_id
    pat_id_mt = pat_id.replace('-0', '-1')
    # input, _ = load_patient_registration(dataset, pat_id_mt, pat_id, region=model_regions, regions_ignore_missing=True)
    # input = pat.ct_data
    input_spacing = pat.ct_spacing

    # Resample input to model spacing.
    input_size = input.shape
    input = resample(input, spacing=input_spacing, output_spacing=model_spacing) 
    input_size_after_resample = input.shape

    # Apply 'naive' cropping.
    if crop_type == 'naive':
        if crop_mm is None:
            raise ValueError(f"Must provide 'crop_mm' for 'naive' cropping.")
        crop = tuple(np.round(np.array(crop_mm) / model_spacing).astype(int))
        input = centre_crop_or_pad(input, crop)
    elif crop_type == 'brain':
        if crop_mm is None:
            raise ValueError(f"Must provide 'crop_mm' for 'brain' cropping.")
        crop_voxels = tuple((np.array(crop_mm) / np.array(model_spacing)).astype(np.int32))

        # Get brain extent.
        localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
        check_epochs = True
        n_epochs = 150
        brain_pred_exists = load_localiser_prediction(dataset, pat_id, localiser, exists_only=True)
        if not brain_pred_exists:
            create_localiser_prediction(dataset, pat_id, localiser, check_epochs=check_epochs, device=device, n_epochs=n_epochs)
        brain_label = load_localiser_prediction(dataset, pat_id, localiser)
        brain_label = resample(brain_label, spacing=input_spacing, output_spacing=model_spacing)
        brain_extent = extent(brain_label)

        # Use image extent if brain isn't present.
        if brain_extent is None:
            brain_extent = ((0, 0, 0), input.shape)

        # Get crop coordinates.
        # Crop origin is centre-of-extent in x/y, and max-extent in z.
        # Cropping boundary extends from origin equally in +/- directions for x/y, and extends
        # in - direction for z.
        p_above_brain = 0.04
        crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, brain_extent[1][2])
        crop = (
            (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - int(crop_voxels[2] * (1 - p_above_brain)))),
            (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2] + int(crop_voxels[2] * p_above_brain)))
        )

        # Crop input.
        input = crop(input, crop)

    else:
        raise ValueError(f"Unknown 'crop_type' value '{crop_type}'.")

    # Pass image to model.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = model(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Apply thresholding/one-hot-encoding.
    pred = pred.argmax(dim=0)
    pred = one_hot(pred, num_classes=len(model_regions) + 1)
    pred = pred.moveaxis(-1, 0)
    pred = pred.cpu().numpy().astype(np.bool_)
    
    # Apply postprocessing.
    pred = largest_cc_4D(pred)

    # Reverse the 'naive' or 'brain' cropping.
    if crop_type == 'naive':
        pred = centre_crop_or_pad(pred, input_size_after_resample)
    elif crop_type == 'brain':
        pad_min = tuple(-np.array(crop[0]))
        pad_max = tuple(np.array(pad_min) + np.array(input_size_after_resample))
        pad_box = (pad_min, pad_max)
        pred = pad(pred, pad_box)

    # Resample to original spacing.
    pred = resample(pred, spacing=model_spacing, output_spacing=input_spacing)

    # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad(pred, crop_box)

    return pred

def get_multi_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    model: Union[ModelName, pl.LightningModule],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    device: torch.device = torch.device('cpu'),
    crop_mm: Optional[Box3D] = None,
    crop_type: str = 'brain',
    **kwargs) -> np.ndarray:
    model_regions = arg_to_list(model_region, str)

    # Load model.
    if type(model) == tuple:
        model = MultiSegmenter.load(*model, **kwargs)
        model.eval()
        model.to(device)

    # Load patient CT data and spacing.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    input = pat.ct_data
    input_spacing = pat.ct_spacing

    # Resample input to model spacing.
    input_size = input.shape
    input = resample(input, spacing=input_spacing, output_spacing=model_spacing) 
    input_size_after_resample = input.shape

    # Apply 'naive' cropping.
    if crop_type == 'naive':
        if crop_mm is None:
            raise ValueError(f"Must provide 'crop_mm' for 'naive' cropping.")
        crop = tuple(np.round(np.array(crop_mm) / model_spacing).astype(int))
        input = centre_crop_or_pad(input, crop)
    elif crop_type == 'brain':
        if crop_mm is None:
            raise ValueError(f"Must provide 'crop_mm' for 'brain' cropping.")
        crop_voxels = tuple((np.array(crop_mm) / np.array(model_spacing)).astype(np.int32))

        # Get brain extent.
        localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
        check_epochs = True
        n_epochs = 150
        brain_pred_exists = load_localiser_prediction(dataset, pat_id, localiser, exists_only=True)
        if not brain_pred_exists:
            create_localiser_prediction(dataset, pat_id, localiser, check_epochs=check_epochs, device=device, n_epochs=n_epochs)
        brain_label = load_localiser_prediction(dataset, pat_id, localiser)
        brain_label = resample(brain_label, spacing=input_spacing, output_spacing=model_spacing)
        brain_extent = extent(brain_label)

        # Use image extent if brain isn't present.
        if brain_extent is None:
            brain_extent = ((0, 0, 0), input.shape)

        # Get crop coordinates.
        # Crop origin is centre-of-extent in x/y, and max-extent in z.
        # Cropping boundary extends from origin equally in +/- directions for x/y, and extends
        # in - direction for z.
        p_above_brain = 0.04
        crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, brain_extent[1][2])
        crop = (
            (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - int(crop_voxels[2] * (1 - p_above_brain)))),
            (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2] + int(crop_voxels[2] * p_above_brain)))
        )

        # Threshold crop values.
        min, max = crop
        min = tuple((np.max((m, 0)) for m in min))
        max = tuple((np.min((m, s)) for m, s in zip(max, input_size_after_resample)))
        crop = (min, max)

        # Crop input.
        input = crop(input, crop)

    else:
        raise ValueError(f"Unknown 'crop_type' value '{crop_type}'.")

    # Pass image to model.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = model(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Apply thresholding/one-hot-encoding.
    pred = pred.argmax(dim=0)
    pred = one_hot(pred, num_classes=len(model_regions) + 1)
    pred = pred.moveaxis(-1, 0)
    pred = pred.cpu().numpy().astype(np.bool_)
    
    # Apply postprocessing.
    pred = largest_cc_4D(pred)

    # Reverse the 'naive' or 'brain' cropping.
    if crop_type == 'naive':
        pred = centre_crop_or_pad(pred, input_size_after_resample)
    elif crop_type == 'brain':
        pad_min = tuple(-np.array(crop[0]))
        pad_max = tuple(np.array(pad_min) + np.array(input_size_after_resample))
        pad_box = (pad_min, pad_max)
        pred = pad(pred, pad_box)

    # Resample to original spacing.
    pred = resample(pred, spacing=model_spacing, output_spacing=input_spacing)

    # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad(pred, crop_box)

    return pred

def get_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    region: str,
    loc_centre: Point3D,
    segmenter: Union[pl.LightningModule, ModelName],
    probs: bool = False,
    seg_spacing: ImageSpacing3D = (1, 1, 2),
    device: torch.device = torch.device('cpu')) -> np.ndarray:

    # Load model.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)
    segmenter.eval()
    segmenter.to(device)

    # Load patient CT data and spacing.
    set = NiftiDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Resample input to segmenter spacing.
    input_size = input.shape
    input = resample(input, spacing=spacing, output_spacing=seg_spacing) 

    # Get localiser centre on downsampled image.
    scaling = np.array(spacing) / seg_spacing
    loc_centre = tuple(int(el) for el in scaling * loc_centre)

    # Extract segmentation patch.
    resampled_size = input.shape
    patch_size = get_region_patch_size(region, seg_spacing)
    patch = get_box(loc_centre, patch_size)
    input = crop_or_pad(input, patch, fill=input.min())

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
    pred = crop_or_pad(pred, rev_patch_box)

    # Resample to original spacing.
    pred = resample(pred, spacing=seg_spacing, output_spacing=spacing)

    # Resampling will round up to the nearest number of voxels, so cropping may be necessary.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad(pred, crop_box)

    return pred

def create_multi_segmenter_prediction_nnunet_bootstrap(
    dataset: Union[str, List[str]],
    pat_id: Union[str, List[str]],
    model: Union[ModelName, pl.LightningModule],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    crop_type: str = 'brain',
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None,
    **kwargs: Dict[str, Any]) -> None:
    model_name = model if isinstance(model, tuple) else model.name
    logging.arg_log('Creating multi-segmenter prediction', ('dataset', 'pat_id', 'model', 'model_region', 'model_spacing', 'crop_type', 'device', 'savepath'), (dataset, pat_id, model_name, model_region, model_spacing, crop_type, device, savepath))
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, str)
    assert len(datasets) == len(pat_ids)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, map_location=device, n_gpus=n_gpus, region=model_region, **kwargs)

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NiftiDataset(dataset)
        pat = set.patient(pat_id)

        # Make prediction.
        pred = get_multi_segmenter_prediction(dataset, pat_id, model, model_region, model_spacing, crop_type=crop_type, device=device, **kwargs)

        # Save segmentation.
        if savepath is None:
            crop_type_str = f'-{crop_type}' if crop_type != 'brain' else ''
            savepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model_name, f'pred{crop_type_str}.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_multi_segmenter_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[str, List[str]],
    model: Union[ModelName, pl.LightningModule],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    crop_type: str = 'brain',
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None,
    **kwargs: Dict[str, Any]) -> None:
    model_name = model if isinstance(model, tuple) else model.name
    logging.arg_log('Creating multi-segmenter prediction', ('dataset', 'pat_id', 'model', 'model_region', 'model_spacing', 'crop_type', 'device', 'savepath'), (dataset, pat_id, model_name, model_region, model_spacing, crop_type, device, savepath))
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, str)
    assert len(datasets) == len(pat_ids)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, map_location=device, n_gpus=n_gpus, region=model_region, **kwargs)

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NiftiDataset(dataset)
        pat = set.patient(pat_id)

        # Make prediction.
        pred = get_multi_segmenter_prediction(dataset, pat_id, model, model_region, model_spacing, crop_type=crop_type, device=device, **kwargs)

        # Save segmentation.
        if savepath is None:
            crop_type_str = f'-{crop_type}' if crop_type != 'brain' else ''
            savepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model_name, f'pred{crop_type_str}.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_segmenter_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[PatientID, List[PatientID]],
    region: str,
    localiser: ModelName,
    segmenter: Union[pl.LightningModule, ModelName],
    device: Optional[torch.device] = None,
    probs: bool = False,
    raise_error: bool = False,
    savepath: Optional[str] = None) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, (int, str), out_type=str)
    datasets = arg_broadcast(dataset, pat_ids, arg_type=str)
    localiser = replace_ckpt_alias(localiser)
    assert len(datasets) == len(pat_ids)

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

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NiftiDataset(dataset)
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser}', segmenter '{segmenter.name}'.")

        # Load localiser centre.
        loc_centre = load_localiser_centre(dataset, pat_id, localiser)

        # Get segmenter prediction.
        if loc_centre is None:
            # Create empty pred.
            if raise_error:
                raise ValueError(f"No 'loc_centre' returned from localiser.")
            else:
                ct_data = set.patient(pat_id).ct_data
                pred = np.zeros_like(ct_data, dtype=bool) 
        else:
            pred = get_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, device=device)

        # Save segmentation.
        if probs:
            filename = 'pred-prob.npz'
        else:
            filename = 'pred.npz'
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'segmenter', dataset, pat_id, *localiser, *segmenter.name, filename)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_all_multi_segmenter_predictions(
    dataset: Union[str, List[str]],
    model: ModelName,
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    timing: bool = True,
    **kwargs) -> None:
    logging.arg_log('Making multi-segmenter predictions', ('dataset', 'localiser'), (dataset, model))
    datasets = arg_to_list(dataset, str)

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load model.
    model = MultiSegmenter.load(model, map_location=device, **kwargs)

    # Create timing table.
    if timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'model': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patients.
    for dataset in datasets:
        set = NiftiDataset(dataset)
        pat_ids = set.list_patients()

        for pat_id in tqdm(pat_ids):
            # Timing table data.
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'model': str(model),
                'device': device.type
            }

            with timer.record(data, enabled=timing):
                create_multi_segmenter_prediction(dataset, pat_id, model, model_region, model_spacing, device=device, **kwargs)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), *model.name, f'timing-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_multi_segmenter_predictions_nnunet_bootstrap(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: Union[ModelName, pl.LightningModule],
    exclude_like: Optional[str] = None,
    use_timing: bool = True,
    use_test_loader: bool = False,
    **kwargs: Dict[str, Any]) -> None:
    logging.arg_log('Making multi-segmenter predictions (nnU-Net bootstrap)', ('dataset', 'region', 'model'), (dataset, region, model))
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    test_fold = kwargs.get('test_fold', None)
    model_spacing = TrainingDataset(datasets[0]).params['spacing']     # Consistency is checked when building loaders in 'MultiLoader'.

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if use_timing:
        cols = {
            'fold': float,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create loaders.
    train_loader, val_loader, test_loader = MultiLoader.build_loaders(datasets, load_data=False, load_train_origin=True, region=regions, **kwargs) 

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, n_gpus=n_gpus, region=regions, **kwargs)

    # Make predictions.
    loader = test_loader if use_test_loader else chain(train_loader, val_loader)
    for pat_desc_b in tqdm(iter(loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            logging.info(f"Predicting '{dataset}:{pat_id}'.")

            if exclude_like is not None:
                if exclude_like in pat_id:
                    logging.info(f"Skipping '{dataset}:{pat_id}', matched 'exclude_like={exclude_like}'.")
                    continue

            # Timing table data.
            data = {
                'fold': test_fold if test_fold is not None else np.nan,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': str(regions),
                'device': device.type
            }

            with timer.record(data, enabled=use_timing):
                if '-0' in pat_id:
                    create_multi_segmenter_prediction_nnunet_bootstrap(dataset, pat_id, model, regions, model_spacing, device=device, **kwargs)
                else:
                    create_multi_segmenter_prediction(dataset, pat_id, model, regions, model_spacing, device=device, **kwargs)

    # Save timing data.
    if use_timing:
        model_name = replace_ckpt_alias(model) if type(model) == tuple else model.name
        params = {
            'device': device.type,
            'load_all_samples': kwargs.get('load_all_samples', False),
            'n_folds': kwargs.get('n_folds', None),
            'shuffle_samples': kwargs.get('shuffle_samples', True),
            'use_grouping': kwargs.get('use_grouping', False),
            'use_split_file': kwargs.get('use_split_file', False),
        }
        filepath = os.path.join(config.directories.predictions, 'timing', 'multi-segmenter', encode(datasets), encode(regions), *model_name, encode(params), 'timing.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_multi_segmenter_predictions(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: Union[ModelName, pl.LightningModule],
    exclude_like: Optional[str] = None,
    use_timing: bool = True,
    use_test_loader: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    logging.arg_log('Making multi-segmenter predictions', ('dataset', 'region', 'model'), (dataset, region, model))
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    test_fold = kwargs.get('test_fold', None)
    model_spacing = TrainingDataset(datasets[0]).params['spacing']     # Consistency is checked when building loaders in 'MultiLoader'.

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if use_timing:
        cols = {
            'fold': float,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create test loader.
    train_loader, val_loader, test_loader = MultiLoader.build_loaders(datasets, region=regions, **kwargs) 

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, n_gpus=n_gpus, region=regions, **kwargs)

    # Make predictions.
    loader = test_loader if use_test_loader else chain(train_loader, val_loader)
    for pat_desc_b in tqdm(iter(loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            logging.info(f"Predicting '{dataset}:{pat_id}'.")

            if exclude_like is not None:
                if exclude_like in pat_id:
                    logging.info(f"Skipping '{dataset}:{pat_id}', matched 'exclude_like={exclude_like}'.")
                    continue

            # Timing table data.
            data = {
                'fold': test_fold if test_fold is not None else np.nan,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': str(regions),
                'device': device.type
            }

            with timer.record(data, enabled=use_timing):
                create_multi_segmenter_prediction(dataset, pat_id, model, regions, model_spacing, device=device, **kwargs)

    # Save timing data.
    if use_timing:
        model_name = replace_ckpt_alias(model) if type(model) == tuple else model.name
        params = {
            'device': device.type,
            'load_all_samples': kwargs.get('load_all_samples', False),
            'n_folds': kwargs.get('n_folds', None),
            'shuffle_samples': kwargs.get('shuffle_samples', True),
            'use_grouping': kwargs.get('use_grouping', False),
            'use_split_file': kwargs.get('use_split_file', False),
        }
        filepath = os.path.join(config.directories.predictions, 'timing', 'multi-segmenter', encode(datasets), encode(regions), *model_name, encode(params), 'timing.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def get_institutional_localiser(
    datasets: Union[str, List[str]],
    dataset: str,
    pat_id: str,
    region: str,
    n_train: float) -> Optional[ModelName]:
    n_folds = 5
    test_folds = list(range(5))
    for test_fold in test_folds:
        localiser = (f'localiser-{region}', f'clinical-fold-{test_fold}-samples-{n_train}', 'best')
        localiser = replace_ckpt_alias(localiser)
        filename = f'eval-folds-{n_folds}-test-{test_fold}'
        filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
        df = pd.read_csv(filepath, dtype={'patient-id': str})
        pdf = df[['dataset', 'patient-id']].drop_duplicates()
        pdf = pdf[(pdf['dataset'] == dataset) & (pdf['patient-id'] == str(pat_id))]
        if len(pdf) == 1:
            return localiser

    return None

def create_segmenter_predictions_v2(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_train: int,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser 'TBD', segmenter '{segmenter.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Get patient IDs from original evaluation.
    # We have to evaluate the segmenter using the original evaluation patient IDs
    # as our 'Loader' now returns different patients per fold.
    orig_localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')
    orig_localiser = replace_ckpt_alias(orig_localiser)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *orig_localiser, *segmenter.name, encode(datasets), f'{filename}.csv')
    orig_df = pd.read_csv(filepath, dtype={'patient-id': str})
    orig_df = orig_df[['dataset', 'patient-id']].drop_duplicates()

    for i, row in tqdm(list(orig_df.iterrows())):
        dataset, pat_id = row['dataset'], row['patient-id']

        # Timing table data.
        data = {
            'fold': test_fold,
            'dataset': dataset,
            'patient-id': pat_id,
            'region': region,
            'device': device.type
        }

        with timer.record(data, enabled=timing):
            create_segmenter_prediction(dataset, pat_id, localiser, segmenter, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser, *segmenter.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_segmenter_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = replace_ckpt_alias(localiser)
    segmenter = Segmenter.load(*segmenter)
    logging.info(f"Making segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Timing table data.
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'device': device.type
            }

            with timer.record(data, enabled=timing):
                create_segmenter_prediction(dataset, pat_id, localiser, segmenter, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser, *segmenter.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def load_multi_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    model: ModelName,
    crop_type: str = 'brain',
    exists_only: bool = False,
    use_model_manifest: bool = False,
    **kwargs) -> Union[np.ndarray, bool]:
    pat_id = str(pat_id)
    model = replace_ckpt_alias(model, use_manifest=use_model_manifest)

    # Load prediction.
    crop_type_str = f'-{crop_type}' if crop_type != 'brain' else ''
    filepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model, f'pred{crop_type_str}.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{dataset}', patient '{pat_id}', model '{model}'. Path: {filepath}")
    pred = np.load(filepath)['data']

    return pred

def load_multi_segmenter_prediction_dict(
    dataset: str,
    pat_id: PatientID,
    model: ModelName,
    model_region: PatientRegions,
    region: Optional[PatientRegions] = None,
    **kwargs) -> Union[Dict[str, np.ndarray], bool]:
    model_regions = regions_to_list(model_region)
    regions = regions_to_list(region)

    # Load prediction.
    pred = load_multi_segmenter_prediction(dataset, pat_id, model, **kwargs)
    if pred.shape[0] != len(model_regions) + 1:
        logging.error(f"Error when processing patient '{dataset}:{pat_id}'.")
        raise ValueError(f"Number of channels in prediction ({pred.shape[0]}) should be one more than number of 'model_regions' ({len(model_regions)}).")

    # Convert to dict.
    data = {}
    for i, r in enumerate(model_regions):
        # Filter based on 'region'.
        if regions is not None and r not in regions:
            continue
        region_pred = pred[i + 1]
        data[r] = region_pred


    return data

def load_multi_segmenter_prediction_timings(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    model = replace_ckpt_alias(model)

    # Load prediction.
    params = {
        'device': kwargs.get('device', 'cuda'),
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.predictions, 'timing', 'multi-segmenter', encode(datasets), encode(regions), *model, encode(params), 'timing.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Multi-segmenter prediction timings not found for dataset '{dataset}', region '{region}', model '{model}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath)

    return df

def load_segmenter_predictions(
    dataset: str,
    pat_id: PatientID,
    model: str,
    exists_only: bool = False,
    regions: PatientRegions = 'all',
    series_id: str = 'series_1',
    study_id: str = 'study_0') -> Union[np.ndarray, bool]:

    # Load predictions.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })
    region_data = {}
    for r in regions:
        filepath = os.path.join(set.path, 'data', 'predictions', pat_id, study_id, 'regions', series_id, r, f'{model}.nii.gz')
        if not os.path.exists(filepath):
            if exists_only:
                return False
            else:
                raise ValueError(f"Prediction not found for dataset '{dataset}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'. Path: {filepath}")
        data, _, _ = load_nifti(filepath)
        region_data[r] = data
    
    if exists_only:
        return True

    return region_data

def load_segmenter_predictions_timings(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    device: str = 'cuda',
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser, *segmenter, f'timing-folds-{n_folds}-test-{test_fold}-device-{device}.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction timings not found for datasets '{datasets}', region '{region}', localiser '{localiser}' and segmenter '{segmenter}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath)

    return df

def save_patient_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    localiser: ModelName,
    segmenter: ModelName,
    data: np.ndarray) -> None:
    localiser = Localiser.replace_ckpt_aliases(*localiser)
    segmenter = Segmenter.replace_ckpt_aliases(*segmenter)

    # Load segmentation.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    np.savez_compressed(filepath, data=data)

def create_two_stage_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[Union[int, List[int], Literal['all']]] = None,
    timing: bool = True,
    **kwargs) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser, **kwargs)
    segmenter = Segmenter.load(*segmenter, **kwargs)
    if test_fold == 'all':
        test_folds = list(range(n_folds))
    elif type(test_fold) == int:
        test_folds = [test_fold]
    else:
        test_folds = test_fold
    logging.info(f"Making two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser.name}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test folds '{test_folds}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if timing:
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        loc_timer = Timer(cols)
        seg_timer = Timer(cols)

    for test_fold in tqdm(test_folds):
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

        # Make predictions.
        for pat_desc_b in tqdm(iter(test_loader)):
            if type(pat_desc_b) == torch.Tensor:
                pat_desc_b = pat_desc_b.tolist()
            for pat_desc in pat_desc_b:
                dataset, pat_id = pat_desc.split(':')

                # Timing table data.
                data = {
                    'fold': test_fold,
                    'dataset': dataset,
                    'patient-id': pat_id,
                    'region': region,
                    'device': device.type
                }

                with loc_timer.record(data, enabled=timing):
                    create_localiser_prediction(dataset, pat_id, localiser, device=device)

                with seg_timer.record(data, enabled=timing):
                    create_segmenter_prediction(dataset, pat_id, region, localiser.name, segmenter, device=device)

        # Save timing data.
        if timing:
            filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            loc_timer.save(filepath)
            filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser.name, *segmenter.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            seg_timer.save(filepath)

def load_moved_data(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    model: str,
    regions: Optional[PatientRegions] = 'all') -> Tuple[CtImage, RegionLabels]:
    # Load moved CT.
    set = NiftiDataset(dataset)
    basepath = os.path.join(set.path, 'data', 'predictions', 'registration', moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model)
    filepath = os.path.join(basepath, 'ct.nii.gz')
    moved_ct, _, _ = load_nifti(filepath)

    # Load moved labels.
    if regions is not None:
        moving_study = set.patient(moving_pat_id).study(moving_study_id)
        regions = regions_to_list(regions, literals={ 'all': moving_study.list_regions })
        moved_regions = {}
        for r in regions:
            if not moving_study.has_regions(r):
                continue

            filepath = os.path.join(basepath, 'rtstruct', f'{r}.nii.gz')
            rdata, _, _ = load_nifti(filepath)
            moved_regions[r] = rdata

    return moved_ct, moved_regions
