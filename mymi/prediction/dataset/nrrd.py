import numpy as np
import os
import pandas as pd
from pandas import DataFrame
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from mymi import config
from mymi.dataset import NRRDDataset, TrainingDataset
from mymi.geometry import get_box, get_extent_centre
from mymi.loaders import Loader, MultiLoader
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.systems import Localiser, MultiSegmenter, Segmenter
from mymi.postprocessing import largest_cc_4D
from mymi.regions import RegionNames, get_region_patch_size, truncate_spine
from mymi.transforms import centre_crop_3D, centre_pad_4D, crop_or_pad_3D, crop_or_pad_4D, resample_3D, resample_4D
from mymi.types import ImageSize3D, ImageSpacing3D, Model, ModelName, PatientID, PatientRegions, Point3D
from mymi.utils import Timer, arg_broadcast, arg_to_list, encode, load_csv

from ..prediction import get_localiser_prediction as get_localiser_prediction_base

def get_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: Model,
    loc_size: ImageSize3D = (128, 128, 150),
    loc_spacing: ImageSpacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
    # Load data.
    set = NRRDDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction_base(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device)

    return pred

def create_localiser_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[Union[int, str], List[Union[int, str]]],
    localiser: Union[ModelName, Model],
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, [int, str], out_type=str)
    datasets = arg_broadcast(datasets, pat_ids)
    assert len(datasets) == len(pat_ids)

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

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NRRDDataset(dataset)
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser.name}'.")

        # Make prediction.
        pred = get_localiser_prediction(dataset, pat_id, localiser, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, pat_id, *localiser.name, 'pred.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_localiser_predictions_for_first_n_pats(
    n_pats: int,
    region: str,
    localiser: ModelName,
    savepath: Optional[str] = None) -> None:
    localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NRRD datasets for region '{region}', first '{n_pats}' patients in 'all-patients.csv'.")

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

def create_localiser_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    logging.info(f"Making localiser predictions for NRRD datasets '{datasets}', region '{region}', localiser '{localiser.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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

            with timer.record(timing, data):
                create_localiser_prediction(dataset, pat_id, localiser, device=device)

    # Save timing data.
    if timing:
        filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def load_localiser_prediction(
    dataset: str,
    pat_id: PatientID,
    localiser: ModelName,
    exists_only: bool = False) -> Union[np.ndarray, bool]:
    localiser = replace_ckpt_alias(localiser)

    # Load prediction.
    set = NRRDDataset(dataset)
    filepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, str(pat_id), *localiser, 'pred.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{set}', patient '{pat_id}', localiser '{localiser}'.")

    pred = np.load(filepath)['data']
    return pred

def load_localiser_predictions_timings(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    device: str = 'cuda',
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> DataFrame:
    localiser = replace_ckpt_alias(localiser)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser, f'timing-folds-{n_folds}-test-{test_fold}-device-{device}.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction timings not found for datasets '{datasets}', region '{region}', and localiser '{localiser}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath)

    return df

def load_localiser_centre(
    dataset: str,
    pat_id: PatientID,
    localiser: ModelName) -> Point3D:
    spacing = NRRDDataset(dataset).patient(pat_id).ct_spacing

    # Get localiser prediction.
    pred = load_localiser_prediction(dataset, pat_id, localiser)

    # Apply cropping for SpinalCord predictions that are "too long" on caudal end.
    # Otherwise the segmentation patch won't cover the top of the SpinalCord.
    region = localiser[0].split('-')[1]         # Infer region.
    if region == 'SpinalCord':
        pred = truncate_spine(pred, spacing)

    # Get localiser pred centre.
    ext_centre = get_extent_centre(pred)

    return ext_centre

def get_multi_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    model: Union[ModelName, Model],
    model_spacing: ImageSpacing3D,
    device: torch.device = torch.device('cpu')) -> np.ndarray:

    # Load model.
    if type(model) == tuple:
        model = MultiSegmenter.load(*model)
    model.eval()
    model.to(device)

    # Load patient CT data and spacing.
    set = NRRDDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    input_spacing = patient.ct_spacing

    # Resample input to model spacing.
    input_size = input.shape
    input = resample_3D(input, spacing=input_spacing, output_spacing=model_spacing) 

    # Apply 'naive' cropping.
    # crop_mm = (320, 520, 730)   # With 60 mm margin (30 mm either end) for each axis.
    crop_mm = (250, 400, 500)   # With 60 mm margin (30 mm either end) for each axis.
    crop = tuple(np.round(np.array(crop_mm) / model_spacing).astype(int))
    resampled_input_size = input.shape
    input = centre_crop_3D(input, crop)

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
    pred = one_hot(pred)
    pred = pred.moveaxis(-1, 0)
    
    # Apply postprocessing.
    pred = pred.cpu().numpy().astype(np.bool_)
    pred = largest_cc_4D(pred)

    # Crop/pad to the resampled size, i.e. before 'naive' cropping.
    pred = centre_pad_4D(pred, resampled_input_size)

    # Resample to original spacing.
    pred = resample_4D(pred, spacing=model_spacing, output_spacing=input_spacing)
    # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.
    crop_box = ((0, 0, 0), input_size)
    pred = crop_or_pad_4D(pred, crop_box)

    return pred

def get_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    loc_centre: Point3D,
    segmenter: Union[Model, ModelName],
    probs: bool = False,
    seg_spacing: ImageSpacing3D = (1, 1, 2),
    device: torch.device = torch.device('cpu')) -> np.ndarray:

    # Load model.
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)
    segmenter.eval()
    segmenter.to(device)

    # Load patient CT data and spacing.
    set = NRRDDataset(dataset)
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
    region = segmenter.name[0].split('-')[1]        # Infer region from model name.
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

def create_multi_segmenter_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[str, List[str]],
    model: Union[ModelName, Model],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None,
    **kwargs: Dict[str, Any]) -> None:
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
        model = MultiSegmenter.load(*model, n_gpus=n_gpus, region=model_region, **kwargs)

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = NRRDDataset(dataset)
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', model '{model.name}'.")

        # Make prediction.
        pred = get_multi_segmenter_prediction(dataset, pat_id, model, model_spacing, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model.name, 'pred.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_segmenter_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[str, List[str]],
    localiser: ModelName,
    segmenter: Union[Model, ModelName],
    device: Optional[torch.device] = None,
    probs: bool = False,
    raise_error: bool = False,
    savepath: Optional[str] = None) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, str)
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
        set = NRRDDataset(dataset)
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
            pred = get_segmenter_prediction(dataset, pat_id, loc_centre, segmenter, device=device)

        # Save segmentation.
        if probs:
            filename = 'pred-prob.npz'
        else:
            filename = 'pred.npz'
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'segmenter', dataset, pat_id, *localiser, *segmenter.name, filename)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def create_multi_segmenter_predictions(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: Union[ModelName, Model],
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_split_file: bool = False,
    use_timing: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    logging.arg_log('Making multi-segmenter predictions', ('dataset', 'region', 'model'), (dataset, region, model))
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    model_spacing = TrainingDataset(datasets[0]).params['output-spacing']     # Consistency is checked when building loaders in 'MultiLoader'.

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
    _, _, test_loader = MultiLoader.build_loaders(datasets, n_folds=n_folds, region=regions, test_fold=test_fold, use_split_file=use_loader_split_file) 

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, n_gpus=n_gpus, region=regions, **kwargs)

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

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
        filepath = os.path.join(config.directories.predictions, 'timing', 'multi-segmenter', encode(datasets), encode(regions), *model_name, f'folds-{n_folds}-test-{test_fold}-use-loader-split-file-{use_loader_split_file}-device-{device.type}-timing.csv')
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
    logging.info(f"Making segmenter predictions for NRRD datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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

            with timer.record(timing, data):
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
    exists_only: bool = False,
    use_model_manifest: bool = False) -> Union[np.ndarray, bool]:
    model = replace_ckpt_alias(model, use_manifest=use_model_manifest)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model, 'pred.npz')
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
    **kwargs) -> Union[Dict[str, np.ndarray], bool]:
    model_regions = arg_to_list(model_region, str)

    # Load prediction.
    pred = load_multi_segmenter_prediction(dataset, pat_id, model, **kwargs)
    if pred.shape[0] != len(model_regions) + 1:
        raise ValueError(f"Number of 'model_regions' ({model_regions}) should match number of channels in prediction '{pred.shape[0]}'.")

    # Convert to dict.
    data = {}
    for i, region in enumerate(model_region):
        region_pred = pred[i + 1]
        data[region] = region_pred

    return data

def load_multi_segmenter_prediction_timings(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    device: str = 'cuda',
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_split_file: bool = False) -> DataFrame:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    model = replace_ckpt_alias(model)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'timing', 'multi-segmenter', encode(datasets), encode(regions), *model, f'folds-{n_folds}-test-{test_fold}-use-loader-split-file-{use_loader_split_file}-device-{device}-timing.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Multi-segmenter prediction timings not found for dataset '{dataset}', region '{region}', model '{model}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath)

    return df

def load_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    localiser: ModelName,
    segmenter: ModelName,
    exists_only: bool = False,
    use_model_manifest: bool = False) -> Union[np.ndarray, bool]:
    localiser = replace_ckpt_alias(localiser, use_manifest=use_model_manifest)
    segmenter = replace_ckpt_alias(segmenter, use_manifest=use_model_manifest)

    # Load segmentation.
    filepath = os.path.join(config.directories.predictions, 'data', 'segmenter', dataset, str(pat_id), *localiser, *segmenter, 'pred.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{dataset}', patient '{pat_id}', segmenter '{segmenter}' with localiser '{localiser}'. Path: {filepath}")

    pred = np.load(filepath)['data']
    return pred

def load_segmenter_predictions_timings(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    device: str = 'cuda',
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> DataFrame:
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
    set = NRRDDataset(dataset)
    filepath = os.path.join(set.path, 'predictions', 'segmenter', *localiser, *segmenter, f'{pat_id}.npz') 
    np.savez_compressed(filepath, data=data)

def create_two_stage_predictions(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[Union[int, List[int], Literal['all']]] = None,
    timing: bool = True) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)
    if test_fold == 'all':
        test_folds = list(range(n_folds))
    elif type(test_fold) == int:
        test_folds = [test_fold]
    else:
        test_folds = test_fold
    logging.info(f"Making two-stage predictions for NRRD datasets '{datasets}', region '{region}', localiser '{localiser.name}', segmenter '{segmenter.name}', with {n_folds}-fold CV using test folds '{test_folds}'.")

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

                with loc_timer.record(timing, data):
                    create_localiser_prediction(dataset, pat_id, localiser, device=device)

                with seg_timer.record(timing, data):
                    create_segmenter_prediction(dataset, pat_id, localiser.name, segmenter, device=device)

        # Save timing data.
        if timing:
            filepath = os.path.join(config.directories.predictions, 'timing', 'localiser', encode(datasets), region, *localiser.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            loc_timer.save(filepath)
            filepath = os.path.join(config.directories.predictions, 'timing', 'segmenter', encode(datasets), region, *localiser.name, *segmenter.name, f'timing-folds-{n_folds}-test-{test_fold}-device-{device.type}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            seg_timer.save(filepath)
