from mymi.transforms.crop import crop_foreground_3D
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Literal, Optional, Union

from mymi import config
from mymi.datasets import NrrdDataset
from mymi.geometry import get_box, centre_of_extent, extent_mm
from mymi.loaders import Loader, MultiLoader
from mymi.metrics import dice, distances, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_ckpt_alias
from mymi.models.lightning_modules import Localiser, Segmenter
from mymi import logging
from mymi.predictions.datasets.nrrd import load_multi_segmenter_prediction_dict, load_segmenter_predictions
from mymi.regions import get_region_patch_size, get_region_tolerance, regions_to_list
from mymi.typing import ModelName, PatientRegions
from mymi.utils import append_row, arg_to_list, encode

def get_multi_segmenter_evaluation(
    dataset: str,
    region: PatientRegions,
    pat_id: str,
    model: ModelName) -> List[Dict[str, float]]:
    regions = arg_to_list(region, str)

    # Load ground truth and prediction.
    set = NrrdDataset(dataset)
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing
    labels = pat.region_data(region=regions)
    region_preds = load_multi_segmenter_prediction_dict(dataset, pat_id, model, regions)
 
    region_metrics = []
    for region, pred in region_preds.items():
        label = labels[region]

        # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
        if region == 'SpinalCord':
            z_min_pred = np.nonzero(pred)[2].min()
            z_min_label = np.nonzero(label)[2].min()
            z_min = np.max([z_min_label, z_min_pred])

            # Crop pred/label foreground voxels.
            crop = ((0, 0, z_min), label.shape)
            pred = crop_foreground_3D(pred, crop)
            label = crop_foreground_3D(label, crop)

        # Dice.
        metrics = {}
        metrics['dice'] = dice(pred, label)

        # Distances.
        if pred.sum() == 0 or label.sum() == 0:
            metrics['apl'] = np.nan
            metrics['hd'] = np.nan
            metrics['hd-95'] = np.nan
            metrics['msd'] = np.nan
            metrics['surface-dice'] = np.nan
        else:
            # Calculate distances for OAR tolerance.
            tols = [0, 0.5, 1, 1.5, 2, 2.5]
            tol = get_region_tolerance(region)
            if tol is not None:
                tols.append(tol)
            dists = distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[metric] = value

            # Add 'deepmind' comparison.
            dists = distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[f'dm-{metric}'] = value

        region_metrics.append(metrics)

    return region_metrics

def get_segmenter_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: ModelName,
    segmenter: ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_segmenter_predictions(dataset, pat_id, localiser, segmenter)
    set = NrrdDataset(dataset)
    label = set.patient(pat_id).region_data(region=region)[region].astype(np.bool_)

    # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
    if region == 'SpinalCord':
        z_min_pred = np.nonzero(pred)[2].min()
        z_min_label = np.nonzero(label)[2].min()
        z_min = np.max([z_min_label, z_min_pred])

        # Crop pred/label foreground voxels.
        crop = ((0, 0, z_min), label.shape)
        pred = crop_foreground_3D(pred, crop)
        label = crop_foreground_3D(label, crop)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing
    if pred.sum() == 0 or label.sum() == 0:
        data['apl'] = np.nan
        data['hd'] = np.nan
        data['hd-95'] = np.nan
        data['msd'] = np.nan
        data['surface-dice'] = np.nan
    else:
        # Calculate distances for OAR tolerance.
        tols = [0, 0.5, 1, 1.5, 2, 2.5]
        tol = get_region_tolerance(region)
        if tol is not None:
            tols.append(tol)
        dists = distances(pred, label, spacing, tol=tols)
        for metric, value in dists.items():
            data[metric] = value

    return data
    
def create_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_split_file: bool = False) -> None:
    datasets = arg_to_list(dataset, str)
    # 'regions' is used to determine which patients are loaded (those that have at least one of
    # the listed regions).
    regions = arg_to_list(region, str)
    model = replace_ckpt_alias(model)
    logging.arg_log('Evaluating multi-segmenter predictions for NRRD dataset', ('dataset', 'region', 'model'), (dataset, region, model))

    # Create dataframe.
    cols = {
        'fold': float,
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Build test loader.
    _, _, test_loader = MultiLoader.build_loaders(datasets, n_folds=n_folds, region=regions, test_fold=test_fold, use_split_file=use_loader_split_file) 

    # Add evaluations to dataframe.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Get metrics per region.
            region_metrics = get_multi_segmenter_evaluation(dataset, regions, pat_id, model)
            for region, metrics in zip(regions, region_metrics):
                for metric, value in metrics.items():
                    data = {
                        'fold': test_fold if test_fold is not None else np.nan,
                        'dataset': dataset,
                        'patient-id': pat_id,
                        'region': region,
                        'metric': metric,
                        'value': value
                    }
                    df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filename = f'folds-{n_folds}-test-{test_fold}-use-loader-split-file-{use_loader_split_file}.csv'
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', *model, encode(datasets), encode(regions), filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
def create_segmenter_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    logging.info(f"Evaluating segmenter predictions for NRRD datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Create dataframe.
    cols = {
        'fold': int,
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Build test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Add evaluations to dataframe.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            metrics = get_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter)
            for metric, value in metrics.items():
                data = {
                    'fold': test_fold,
                    'dataset': dataset,
                    'patient-id': pat_id,
                    'region': region,
                    'metric': metric,
                    'value': value
                }
                df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    exists_only: bool = False,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_split_file: bool = False) -> Union[np.ndarray, bool]:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    model = replace_ckpt_alias(model)
    filename = f'folds-{n_folds}-test-{test_fold}-use-loader-split-file-{use_loader_split_file}.csv'
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', *model, encode(datasets), encode(regions), filename)
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Multi-segmenter evaluation for dataset '{dataset}', model '{model}' not found. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df

def load_segmenter_evaluation(
    datasets: Union[str, List[str]],
    localiser: ModelName,
    segmenter: ModelName,
    exists_only: bool = False,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> Union[np.ndarray, bool]:
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Segmenter evaluation for dataset '{datasets}', localiser '{localiser}', segmenter '{segmenter}', {n_folds}-fold CV with test fold {test_fold} not found. Filepath: {filepath}.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def create_two_stage_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    logging.info(f"Evaluating two-stage predictions for NRRD datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test folds '{test_folds}'.")

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(n_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    for test_fold in tqdm(test_folds):
        # Create dataframe.
        cols = {
            'fold': int,
            'patient-id': str,
            'region': str,
            'metric': str,
            'value': float
        }
        loc_df = pd.DataFrame(columns=cols.keys())
        seg_df = pd.DataFrame(columns=cols.keys())

        # Build test loader.
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

        # Add evaluations to dataframe.
        for pat_desc_b in tqdm(iter(test_loader)):
            if type(pat_desc_b) == torch.Tensor:
                pat_desc_b = pat_desc_b.tolist()
            for pat_desc in pat_desc_b:
                dataset, pat_id = pat_desc.split(':')
                loc_df = create_localiser_evaluation(dataset, pat_id, region, localiser, df=loc_df)
                seg_df = create_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter, df=seg_df)

        # Add fold.
        loc_df['fold'] = test_fold
        seg_df['fold'] = test_fold

        # Set column types.
        loc_df = loc_df.astype(cols)
        seg_df = seg_df.astype(cols)

        # Save evaluations.
        filename = f'eval-folds-{n_folds}-test-{test_fold}'
        loc_filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
        seg_filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
        os.makedirs(os.path.dirname(loc_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(seg_filepath), exist_ok=True)
        loc_df.to_csv(loc_filepath, index=False)
        seg_df.to_csv(seg_filepath, index=False)
