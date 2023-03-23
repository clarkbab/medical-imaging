from mymi.transforms.crop import crop_foreground_3D
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Literal, Optional, Union

from mymi import config
from mymi.dataset.nrrd import NRRDDataset
from mymi.geometry import get_box, get_extent_centre
from mymi.loaders import Loader
from mymi.metrics import all_distances, dice, distances_deepmind, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_checkpoint_alias
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction.dataset.nrrd import load_localiser_prediction, load_segmenter_prediction
from mymi.regions import get_region_patch_size, get_region_tolerance
from mymi import types
from mymi.utils import append_row, encode

def get_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_localiser_prediction(dataset, pat_id, localiser)
    set = NRRDDataset(dataset)
    label = set.patient(pat_id).region_data(region=region)[region].astype(np.bool_)

    # If 'SpinalCord' prediction extends further than ground truth in caudal z direction, then crop prediction.
    if region == 'SpinalCord':
        z_min_pred = np.nonzero(pred)[2].min()
        z_min_label = np.nonzero(label)[2].min()
        if z_min_pred < z_min_label:
            # Crop pred/label foreground voxels.
            crop = ((0, 0, z_min_label), label.shape)
            pred = crop_foreground_3D(pred, crop)

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
        dists = all_distances(pred, label, spacing, tols)
        for metric, value in dists.items():
            data[metric] = value

        # Add 'deepmind' comparison.
        dists = distances_deepmind(pred, label, spacing, tols)
        for metric, value in dists.items():
            data[f'dm-{metric}'] = value

    # Extent distance.
    if pred.sum() == 0:
        ec_dist = (np.nan, np.nan, np.nan)
    else:
        ec_dist = extent_centre_distance(pred, label, spacing)

    data['extent-centre-dist-x'] = ec_dist[0]
    data['extent-centre-dist-y'] = ec_dist[1]
    data['extent-centre-dist-z'] = ec_dist[2]

    # Second stage patch distance.
    if pred.sum() == 0:
        e_dist = (np.nan, np.nan, np.nan)
    else:
        # Get second-stage patch min/max coordinates.
        centre = get_extent_centre(pred)
        size = get_region_patch_size(region, spacing)
        min, max = get_box(centre, size)

        # Clip second-stage patch to label size - if necessary.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.clip(max, a_min=None, a_max=label.shape)

        # Convert second-stage patch coordinates into a label of ones so we can use 'get_encaps_dist_mm'.
        patch_label = np.zeros_like(label)
        slices = tuple([slice(l, h + 1) for l, h in zip(min, max)])
        patch_label[slices] = 1

        # Get extent distance.
        e_dist = get_encaps_dist_mm(patch_label, label, spacing)

    data['encaps-dist-mm-x'] = e_dist[0]
    data['encaps-dist-mm-y'] = e_dist[1]
    data['encaps-dist-mm-z'] = e_dist[2]

    return data

def create_localiser_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_checkpoint_alias(*localiser)
    logging.info(f"Evaluating localiser predictions for NRRD datasets '{datasets}', region '{region}', localiser '{localiser}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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
            metrics = get_localiser_evaluation(dataset, pat_id, region, localiser)
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

    # Add fold.
    df['fold'] = test_fold

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_localiser_evaluation(
    datasets: Union[str, List[str]],
    localiser: types.ModelName,
    exists_only: bool = False,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> np.ndarray:
    localiser = replace_checkpoint_alias(*localiser)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Localiser evaluation for dataset '{datasets}', localiser '{localiser}', {n_folds}-fold CV with test fold {test_fold} not found. Filepath: {filepath}.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def get_segmenter_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)
    set = NRRDDataset(dataset)
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
        dists = all_distances(pred, label, spacing, tols)
        for metric, value in dists.items():
            data[metric] = value

        # Add 'deepmind' comparison.
        dists = distances_deepmind(pred, label, spacing, tols)
        for metric, value in dists.items():
            data[f'dm-{metric}'] = value

    return data
    
def create_segmenter_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_checkpoint_alias(*localiser)
    segmenter = replace_checkpoint_alias(*segmenter)
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

def load_segmenter_evaluation(
    datasets: Union[str, List[str]],
    localiser: types.ModelName,
    segmenter: types.ModelName,
    exists_only: bool = False,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> Union[np.ndarray, bool]:
    localiser = replace_checkpoint_alias(*localiser)
    segmenter = replace_checkpoint_alias(*segmenter)
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
    localiser: types.ModelName,
    segmenter: types.ModelName,
    n_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    # Get unique name.
    localiser = replace_checkpoint_alias(*localiser)
    segmenter = replace_checkpoint_alias(*segmenter)
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